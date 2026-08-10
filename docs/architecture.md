# MLOps architecture

```text
synthetic source -> feature contract -> time-aware training -> model artifact
                                             |                    |
                                      metrics/model card       FastAPI
                                                                    |
                                       prediction logs -> PSI/drift -> retraining decision
```

## Production controls to add

1. Store features and labels with point-in-time correctness.
2. Track runs in MLflow or Vertex AI Experiments.
3. Register approved models and require review before promotion.
4. Log prediction IDs, model version, latency and reason codes without PII.
5. Schedule drift checks and retraining through Airflow/Cloud Composer.
6. Compare champion/challenger models and support rollback.
7. Add fairness, calibration, adverse-action and human-review controls before use.

