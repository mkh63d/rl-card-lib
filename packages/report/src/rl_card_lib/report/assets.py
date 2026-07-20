"""CSS and JavaScript for the HTML report, as string constants.

Kept inline rather than as separate files so the rendered page is a single
portable document: no CDN, no sibling stylesheet, nothing to break when the
file is copied somewhere else or opened without a network.

Light mode only, deliberately. The figures are matplotlib PNGs baked at render
time against a light surface, so a dark page would sit dark chrome under light
charts. This is a thesis appendix that gets printed.
"""

CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --plane: #f9f9f7;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --rule: #c3c2b7;
  --good: #006300;
  --critical: #d03b3b;
  --warn-bg: #fdf6e3;
  --border: rgba(11, 11, 11, 0.10);
  --q-learning: #2a78d6;
  --dqn: #008300;
  --double-dqn: #e87ba4;
  --ppo: #eda100;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--plane);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.55;
}

.wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px 96px; }

a { color: #1c5cab; }

/* ---- header ---- */
header.page {
  border-bottom: 1px solid var(--rule);
  background: var(--surface);
  padding: 40px 0 28px;
  margin-bottom: 0;
}
header.page h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: -0.01em; }
header.page p.lede { margin: 0; color: var(--ink-2); max-width: 70ch; }
.meta {
  display: flex; flex-wrap: wrap; gap: 8px 28px;
  margin-top: 18px; font-size: 13px; color: var(--muted);
}
.meta code {
  background: var(--plane); border: 1px solid var(--border);
  border-radius: 4px; padding: 1px 6px; color: var(--ink-2);
}

/* ---- nav ---- */
nav.toc {
  position: sticky; top: 0; z-index: 20;
  background: rgba(252, 252, 251, 0.96);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--rule);
  padding: 10px 0;
  margin-bottom: 36px;
}
nav.toc .wrap { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; }
nav.toc a {
  font-size: 13px; text-decoration: none; color: var(--ink-2);
  padding: 3px 8px; border-radius: 5px; white-space: nowrap;
}
nav.toc a:hover { background: var(--plane); color: var(--ink); }
nav.toc .sep { color: var(--rule); }

/* ---- sections ---- */
section { margin: 0 0 56px; scroll-margin-top: 62px; }
h2 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 17px; margin: 32px 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 18px; max-width: 78ch; }

.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px 22px; margin-bottom: 18px;
}

/* ---- stat tiles ---- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.tile .value { font-size: 26px; font-weight: 600; line-height: 1.2; margin-top: 2px; }
.tile .foot { font-size: 12px; color: var(--ink-2); margin-top: 2px; }

.delta-up { color: var(--good); }
.delta-down { color: var(--critical); }

/* ---- tables ---- */
.table-block { margin: 18px 0; }
.table-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
caption { text-align: left; font-size: 13px; color: var(--ink-2); padding: 0 0 8px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { font-weight: 600; color: var(--ink-2); background: var(--plane); position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #f5f5f2; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.none { color: var(--muted); font-style: italic; }

.toolbar { display: flex; gap: 6px; justify-content: flex-end; margin-bottom: 6px; }
.toolbar button, .toolbar a.btn {
  font: inherit; font-size: 12px; line-height: 1.4;
  background: var(--surface); color: var(--ink-2);
  border: 1px solid var(--rule); border-radius: 6px;
  padding: 3px 10px; cursor: pointer; text-decoration: none;
}
.toolbar button:hover, .toolbar a.btn:hover { background: var(--plane); color: var(--ink); }
.toolbar button:active { transform: translateY(1px); }

/* ---- figures ---- */
.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }
figure {
  margin: 0; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px 12px;
}
figure img { width: 100%; height: auto; display: block; }
figure figcaption { font-size: 12.5px; color: var(--ink-2); margin-top: 8px; }
figure .fig-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 8px; }
figure .fig-head h4 { margin: 0; font-size: 14px; }
details { margin-top: 10px; }
details summary { cursor: pointer; font-size: 12.5px; color: var(--ink-2); }
details summary:hover { color: var(--ink); }
details .table-block { margin-top: 10px; }
details table { font-size: 12.5px; }
details .table-scroll { max-height: 320px; overflow-y: auto; }

/* ---- callouts ---- */
.notes { background: var(--warn-bg); border: 1px solid #eadfbe; border-radius: 10px; padding: 12px 16px; margin: 16px 0; }
.notes h4 { margin: 0 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: #7a5c12; }
.notes ul { margin: 0; padding-left: 20px; }
.notes li { font-size: 13.5px; color: #5c4a1a; margin: 3px 0; }

.chip {
  display: inline-block; font-size: 11.5px; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--plane); color: var(--ink-2);
  vertical-align: middle;
}
.chip.swatch::before {
  content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--dot, var(--muted)); margin-right: 6px; vertical-align: baseline;
}
.chip.failed { background: #fbeaea; border-color: #e9c6c6; color: #8d2a2a; }

.empty { color: var(--muted); font-style: italic; }
footer.page { border-top: 1px solid var(--rule); padding-top: 18px; color: var(--muted); font-size: 13px; }

/* ---- print: thesis appendix ---- */
@media print {
  body { background: #fff; }
  nav.toc { display: none; }
  .toolbar { display: none; }
  figure, .card, .tile, .table-block { break-inside: avoid; page-break-inside: avoid; }
  section { break-before: auto; }
  details { display: none; }
  a { color: inherit; text-decoration: none; }
}
"""


# Export lives in the page so the report keeps working offline and after being
# copied elsewhere. CSV is the guaranteed path; PNG uses foreignObject, which
# is solid in Chrome/Firefox/Edge and historically flaky in Safari, so it
# fails loudly back to CSV rather than silently producing a blank image.
JS = """
(function () {
  "use strict";

  function cells(row) {
    return Array.prototype.slice.call(row.querySelectorAll("th,td"))
      .map(function (cell) { return (cell.innerText || "").trim(); });
  }

  function grid(table) {
    return Array.prototype.slice.call(table.querySelectorAll("tr")).map(cells);
  }

  function toCSV(table) {
    return grid(table).map(function (row) {
      return row.map(function (value) {
        return /[",\\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
      }).join(",");
    }).join("\\r\\n");
  }

  function toTSV(table) {
    return grid(table).map(function (row) { return row.join("\\t"); }).join("\\n");
  }

  function download(blob, filename) {
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function flash(button, message) {
    var original = button.textContent;
    button.textContent = message;
    setTimeout(function () { button.textContent = original; }, 1400);
  }

  function tableToPNG(table, filename, button) {
    var rect = table.getBoundingClientRect();
    var width = Math.ceil(rect.width) || 720;
    var height = Math.ceil(rect.height) || 240;
    var scale = 2;

    var clone = table.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
    var style =
      "font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:13.5px;" +
      "border-collapse:collapse;background:#fcfcfb;color:#0b0b0b;width:" + width + "px;";
    clone.setAttribute("style", style);
    Array.prototype.forEach.call(clone.querySelectorAll("th,td"), function (cell) {
      cell.setAttribute(
        "style",
        "padding:8px 12px;border-bottom:1px solid #e1e0d9;text-align:left;white-space:nowrap;"
      );
    });
    Array.prototype.forEach.call(clone.querySelectorAll("th"), function (cell) {
      cell.setAttribute(
        "style",
        cell.getAttribute("style") + "background:#f9f9f7;font-weight:600;color:#52514e;"
      );
    });

    var svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '">' +
      '<foreignObject width="100%" height="100%">' +
      new XMLSerializer().serializeToString(clone) +
      "</foreignObject></svg>";

    var image = new Image();
    image.onload = function () {
      try {
        var canvas = document.createElement("canvas");
        canvas.width = width * scale;
        canvas.height = height * scale;
        var context = canvas.getContext("2d");
        context.fillStyle = "#fcfcfb";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.scale(scale, scale);
        context.drawImage(image, 0, 0);
        canvas.toBlob(function (blob) {
          if (blob) { download(blob, filename); } else { flash(button, "use CSV"); }
        });
      } catch (err) {
        flash(button, "use CSV");
      }
    };
    image.onerror = function () { flash(button, "use CSV"); };
    image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }

  function nameOf(block) {
    return (block.getAttribute("data-name") || "table").replace(/[^a-z0-9_.-]+/gi, "_");
  }

  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-export]");
    if (!button) { return; }
    var block = button.closest(".table-block");
    var table = block && block.querySelector("table");
    if (!table) { return; }

    var kind = button.getAttribute("data-export");
    var name = nameOf(block);

    if (kind === "csv") {
      download(new Blob([toCSV(table)], { type: "text/csv;charset=utf-8" }), name + ".csv");
    } else if (kind === "copy") {
      var text = toTSV(table);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () { flash(button, "copied"); },
          function () { flash(button, "failed"); }
        );
      } else {
        flash(button, "unsupported");
      }
    } else if (kind === "png") {
      tableToPNG(table, name + ".png", button);
    }
  });
})();
"""

__all__ = ["CSS", "JS"]
