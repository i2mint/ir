# Calibration cases

`skills_min_score_cases.jsonl`: 14 positive (a prompt and the skill that should answer it) and 10 abstain-expected (off-topic) cases for the `skills` corpus. The floor is per-machine corpus state, so re-run it wherever the skills set differs:

```bash
ir calibrate-min-score skills misc/data/skills_min_score_cases.jsonl --mode lexical --persist
```

Gold ids are skill names; if a skill is renamed or removed, `ir.eval.validate_cases` flags it.
