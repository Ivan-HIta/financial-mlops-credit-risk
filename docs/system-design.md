# System design interview brief

## Lifecycle

```text
feature source -> validation -> reproducible train -> candidate registry
                                                   |
                                       offline quality + risk gates
                                                   |
                                      approval -> production API
                                                   |
                              logs -> drift/fairness/SLA monitoring
                                                   |
                                      retrain or rollback
```

The local JSON registry mirrors MLflow-style candidate, production and archived stages.
In a production implementation, MLflow Tracking records parameters, artifacts and
metrics, while the model registry provides versioned promotion and audit history.

## Reliability and safety

- Package the API as an immutable image and run multiple Kubernetes replicas.
- Expose health checks and keep a previous production artifact for rollback.
- Block promotion when validation, drift, fairness or explainability gates fail.
- Keep prediction logs free of direct identifiers and apply retention controls.
- Use Terraform for the cluster/service dependencies and CI for tests and image scans.

The design intentionally separates model quality from business approval: a statistically
strong model is not automatically an eligible credit decision system.
