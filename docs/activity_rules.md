# Activity labelling rules

Written by the drilling engineer during labelling (2026-11-06) and used for every activity label
(fields `activity_rules` in `data/labels/labels/*.json` refer to these rules). The same rules,
as class definitions, are in the activity prompt in `prototype/run_prototype.py`.

- Classify a time-log row by the work done, not by the reason for the time.
- Rig service and slip-and-cut drilling line: `other`.
- BHA pick-up or lay-down, with or without running in hole: `tripping`.
- "Fill pipe": `circulating`.
- Repair while waiting on a technician: `rig_repair`.
