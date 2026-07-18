# Report Package - TODO

- [x] JSON export (`TrainingReport.to_json()`). YAML deliberately skipped: it
  would add a dependency for a format nothing downstream consumes yet.
- [x] Report helpers for other agent types — PPO section added; DoubleDQN is
  covered by the DQN section it inherits from; rule-based agents appear in the
  generic agent section.
- [x] Tests (`tests/test_report.py`).
