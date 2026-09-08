"""Writing an episode out as a self-contained HTML page.

A 300-step Klondike is unpleasant to read as terminal scrollback: the boards
are the interesting part and they scroll past. A page that holds every board
and steps between them shows the same episode as something you can move back
and forth through, which is what looking for the move where an agent went
wrong actually requires.

Written from the standard library rather than through the `report` package.
The helpers there (`_escape`, `_table`, `HtmlReport`) are private and shaped
around `RunRecord` and tables of training metrics -- a different document
entirely -- so reusing them would mean widening their API and adding a package
dependency to gain `html.escape`.

The page carries the raw Unicode boards: it declares UTF-8, so unlike the
terminal it can show the real suit glyphs, and downgrading them here would
lose information the record still holds.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional, Union

from rl_card_lib.visualizer.episode import EpisodeRecord

_CSS = """
:root {
  --bg: #f7f7f8; --fg: #1c1c1e; --muted: #6b6b70;
  --panel: #ffffff; --line: #d8d8dc; --accent: #2d5fa4;
  --good: #1a7f45; --bad: #a4342d;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17181b; --fg: #e6e6e8; --muted: #9a9aa2;
    --panel: #1f2024; --line: #34353b; --accent: #7aa7e6;
    --good: #4cc27c; --bad: #e2796f;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.meta { color: var(--muted); margin-bottom: 20px; }
.meta b { color: var(--fg); font-weight: 600; }
.outcome { font-weight: 600; }
.outcome.solved { color: var(--good); }
.outcome.lost, .outcome.truncated { color: var(--bad); }
.cols { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; }
@media (max-width: 860px) { .cols { grid-template-columns: minmax(0, 1fr); } }
.panel {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 16px;
}
.controls { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
button {
  font: inherit; padding: 5px 12px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--line); background: var(--panel); color: var(--fg);
}
button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
button:disabled { opacity: .4; cursor: default; }
input[type=range] { flex: 1; min-width: 140px; accent-color: var(--accent); }
.counter { font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }
pre.board {
  margin: 0; overflow-x: auto; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: 13px; line-height: 1.45; white-space: pre;
}
.move { margin-bottom: 12px; font-size: 15px; }
.move .label { font-weight: 600; }
.nums { color: var(--muted); font-variant-numeric: tabular-nums; }
.tag {
  display: inline-block; margin-left: 6px; padding: 0 6px; border-radius: 4px;
  font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
  border: 1px solid var(--line); color: var(--muted);
}
.tag.illegal, .tag.stepcap { color: var(--bad); border-color: var(--bad); }
.tag.terminal { color: var(--good); border-color: var(--good); }
/* A repeat is not a failure on its own, so it keeps the muted base -- stated
   explicitly so every class `_flags` can emit has a rule of its own. */
.tag.repeat { color: var(--muted); border-color: var(--line); }
ol.moves { list-style: none; margin: 0; padding: 0; max-height: 60vh; overflow-y: auto; }
ol.moves li {
  padding: 5px 8px; border-radius: 5px; cursor: pointer;
  display: flex; gap: 8px; align-items: baseline;
}
ol.moves li:hover { background: var(--bg); }
ol.moves li:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
ol.moves li.on { background: var(--accent); color: #fff; }
ol.moves li.on .nums, ol.moves li.on .idx { color: #fff; opacity: .85; }
.idx { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 34px; }
.grow { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hint { color: var(--muted); font-size: 12px; margin-top: 12px; }
"""

# Frame data reaches the DOM as text and never as markup -- no innerHTML
# anywhere below. Escaping `<` in the payload keeps the *HTML parser* from
# seeing markup, but JS reads the escape back as `<`, so a label assigned
# through innerHTML would still run: an action label is game-supplied text and
# this page does not trust it.
_JS = """
const F = FRAMES;
let i = 0;
const board = document.getElementById('board');
const move = document.getElementById('move');
const slider = document.getElementById('slider');
const counter = document.getElementById('counter');
const prev = document.getElementById('prev');
const next = document.getElementById('next');
const items = [...document.querySelectorAll('ol.moves li')];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  node.textContent = text;
  return node;
}

function show(n) {
  i = Math.max(0, Math.min(F.length - 1, n));
  const f = F[i];
  board.textContent = f.board || '(this game does not render a board)';

  move.textContent = '';
  move.appendChild(el('span', 'label', f.action === null ? 'Opening deal' : f.label));
  if (f.action !== null) {
    (f.flags || []).forEach(t => move.appendChild(el('span', 'tag ' + t.cls, t.text)));
    move.appendChild(el('div', 'nums',
      `action ${f.action} \\u00b7 reward ${f.reward} \\u00b7 total ${f.total}`));
  }

  slider.value = i;
  counter.textContent = `${i} / ${F.length - 1}`;
  prev.disabled = i === 0;
  next.disabled = i === F.length - 1;
  items.forEach((li, k) => {
    li.classList.toggle('on', k === i);
    li.setAttribute('aria-current', k === i ? 'true' : 'false');
  });
  const on = items[i];
  if (on) on.scrollIntoView({ block: 'nearest' });
}

prev.onclick = () => show(i - 1);
next.onclick = () => show(i + 1);
slider.oninput = () => show(+slider.value);
items.forEach((li, k) => {
  li.onclick = () => show(k);
  // A click handler alone leaves the list unusable from the keyboard: the
  // items carry role="button", so Enter and Space have to activate them.
  li.onkeydown = e => {
    if (e.key === 'Enter' || e.key === ' ') { show(k); e.preventDefault(); }
  };
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') { show(i - 1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { show(i + 1); e.preventDefault(); }
  if (e.key === 'Home') { show(0); e.preventDefault(); }
  if (e.key === 'End') { show(F.length - 1); e.preventDefault(); }
});
show(0);
"""


def _flags(step) -> list:
    """The step's flags, each paired with the CSS class that styles it.

    The class is computed here and travels with the label rather than being
    derived in the page from the label's text. Deriving it there is how
    "step cap" came to ask for `.tag.cap` -- the stylesheet defines
    `.tag.stepcap`, so the flag silently lost its highlight and nothing failed.
    One definition, and `test_every_flag_class_is_styled` holds it to the
    stylesheet.
    """
    names = []
    if step.invalid:
        names.append("illegal")
    if step.repeated:
        names.append("repeat")
    if step.terminated:
        names.append("terminal")
    if step.truncated:
        names.append("step cap")
    return [{"text": name, "cls": name.replace(" ", "")} for name in names]


def _frames(record: EpisodeRecord) -> list:
    """The page's frames: the opening deal, then one per move.

    Frame 0 has `action: null` so the page can label it as the deal rather
    than as a move -- an episode is N moves but N+1 positions, and the
    starting position is the one you compare everything against.
    """
    frames = [{
        "action": None,
        "label": "Opening deal",
        "reward": "",
        "total": "",
        "flags": [],
        "board": record.opening_board,
    }]
    for step in record.steps:
        frames.append({
            "action": step.action,
            "label": step.action_label,
            "reward": f"{step.reward:+.3f}",
            "total": f"{step.total_reward:+.3f}",
            "flags": _flags(step),
            "board": step.board,
        })
    return frames


def episode_to_html(record: EpisodeRecord, *, title: Optional[str] = None) -> str:
    """Render an episode as one self-contained HTML document.

    No external assets and no network: the CSS, the script and every board are
    inlined, so the file can be mailed, committed next to a thesis chapter, or
    opened from a USB stick and still work.

    Args:
        record: The episode to render.
        title: Page title; a description of the episode when omitted.

    Returns:
        The complete HTML document.
    """
    frames = _frames(record)
    heading = title or f"{record.game} - seed {record.seed} - {record.agent}"

    lis = []
    for n, frame in enumerate(frames):
        label = html.escape(str(frame["label"]), quote=True)
        idx = "deal" if frame["action"] is None else str(n - 1)
        total = html.escape(str(frame["total"]), quote=True)
        # Focusable and announced as a button: the items are activated by
        # click, so without these a keyboard cannot reach them at all.
        lis.append(
            f'<li tabindex="0" role="button" aria-current="false">'
            f'<span class="idx">{idx}</span>'
            f'<span class="grow">{label}</span>'
            f'<span class="nums">{total}</span></li>'
        )

    # Every `<` becomes a string escape. `</script>` is the obvious way out of
    # a script element, but `<!--` and a nested `<script` also flip the HTML
    # parser into its escaped states, and an action label is game-supplied text
    # this module has no business trusting. JSON structure contains no `<` of
    # its own, so this only ever rewrites string contents, and `<` is read
    # back as `<` by the JS string literal.
    #
    # U+2028 and U+2029 go the same way. JSON permits them raw inside a string;
    # JavaScript counts them as line terminators, so before ES2019 they end the
    # literal mid-payload and the whole script fails to parse. Escaping costs
    # nothing and does not depend on the reader's engine.
    payload = (
        json.dumps(frames, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(heading, quote=True)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(heading, quote=True)}</h1>
  <p class="meta">
    game <b>{html.escape(str(record.game), quote=True)}</b> &middot;
    seed <b>{html.escape(str(record.seed), quote=True)}</b> &middot;
    agent <b>{html.escape(str(record.agent), quote=True)}</b> &middot;
    {record.step_count} steps &middot;
    return <b>{record.total_reward:+.3f}</b> &middot;
    <span class="outcome {html.escape(record.outcome, quote=True)}">{html.escape(record.outcome, quote=True)}</span>
  </p>
  <div class="cols">
    <div class="panel">
      <div class="controls">
        <button id="prev" type="button">&larr; Prev</button>
        <button id="next" type="button">Next &rarr;</button>
        <input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0">
        <span class="counter" id="counter"></span>
      </div>
      <div class="move" id="move"></div>
      <pre class="board" id="board"></pre>
      <p class="hint">Arrow keys step, Home and End jump to the deal and the last move.</p>
    </div>
    <div class="panel">
      <ol class="moves">{''.join(lis)}</ol>
    </div>
  </div>
</div>
<script>
{_JS.replace('FRAMES', payload)}
</script>
</body>
</html>
"""


def write_episode_html(
    record: EpisodeRecord,
    path: Union[str, Path],
    *,
    title: Optional[str] = None,
) -> Path:
    """Write `episode_to_html` to a file, creating parent directories.

    Args:
        record: The episode to render.
        path: Destination file.
        title: Page title; see `episode_to_html`.

    Returns:
        The path written, resolved.
    """
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8: the boards carry suit glyphs and the default encoding on
    # a Polish Windows is cp1250, which cannot write them (#47).
    target.write_text(episode_to_html(record, title=title), encoding="utf-8")
    return target.resolve()
