# analysis/

Offline analysis of races, telemetry, rosbags, and training runs — lap-time
studies, trajectory/raceline comparisons, RL learning-curve analysis, failure
triage.

Keep **notebooks and scripts** here (they are small and reviewable). The data they
consume — rosbags, logs, checkpoints, wandb exports, rendered plots — is heavy and
**gitignored**; store it off-repo and load it by path/URL.

Suggested convention:

```text
analysis/
  <topic>/            # e.g. laptime_study/, raceline_compare/
    notebook.ipynb    # or a .py script
    README.md         # what it does + where the data lives (off-repo)
```
