# Business metrics and risk guardrails

This project uses synthetic applicants. The metrics below are the measurement contract,
not claimed production outcomes.

| Metric | Definition | Guardrail / decision |
| --- | --- | --- |
| Approval / review rate | Share of applications above the review threshold | Capacity for human review |
| Expected loss proxy | Predicted risk weighted by exposure | Portfolio monitoring |
| ROC AUC / average precision | Ranking quality on a time-aware holdout | Candidate acceptance gate |
| PSI | Distribution shift between baseline and current features | Retrain or investigate |
| Prediction latency | API request time at p50/p95 | User experience and SLA |
| Cost per 1,000 predictions | Compute and serving cost divided by volume | Scale decision |
| Fairness gap | Difference in approved/reviewed outcomes across groups | Responsible-use review |

Before using a real model, legal, risk and compliance owners must approve the target
population, features, adverse-action explanations, thresholds, monitoring baselines and
rollback policy. No synthetic metric in this repository should be presented as a client
result.
