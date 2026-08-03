# Predictive Maintenance for IoT on GCP

**Predict equipment failure ahead of time from streaming sensor data, with the training,
serving and deployment paths all written out.**

## What's here

Four modules, each a full layer of the system.

### `predictive_maintenance_main.py` — the core pipeline
`IoTSensorDataGenerator` synthesises realistic multi-sensor signals with injected
degradation, so the pipeline can be exercised without hardware. `TimeSeriesPreprocessor`
windows the history and labels each window by what happens in the following 48 hours, and
splits chronologically. `PredictiveMaintenanceModel` is the LSTM.
`RealTimePredictor`, `ModelOptimizer` and `MonitoringDashboard` cover inference, size
reduction and reporting.

### `model_training_optimization.py` — getting the model to production quality
`AdvancedFeatureEngineering`, `HyperparameterOptimizer` (Optuna), `EnsembleModel`,
`ModelCompression`, `AdvancedTrainingPipeline`, `ProductionModelManager` and
`PerformanceBenchmark`.

### `realtime_processing_system.py` — the serving path
FastAPI with Pydantic schemas (`SensorReading`, `PredictionResult`),
`RealTimePreprocessor`, `ModelInferenceEngine`, `StreamingPipeline`, `WebSocketManager`
for live push, plus `MetricsCollector`, `PerformanceMonitor` and `DataQualityChecker`.

### `gcp_deployment_script.py` — the cloud wiring
Pub/Sub ingestion, a Dataflow pipeline (`ProcessSensorData`, `AggregateMetrics`), a Cloud
Function predictor, Vertex AI endpoint with autoscaling, BigQuery storage, Cloud
Monitoring, and an `AutoScalingManager`.

## Design targets

- Predict failures roughly 48 hours ahead
- Accuracy in the low-to-mid nineties on the failure class
- Handle tens of thousands of events per day across a few hundred devices
- Sub-100 ms inference on the serving path
- Meaningful reduction in unplanned downtime

## On the numbers

The figures above are **design targets** that shaped the implementation — they are not
measured results. This repository ships no benchmark harness and no trained weights, so
nothing here reproduces them. They are recorded because they drove real decisions about
architecture and algorithm choice, not as claims about observed performance.

## Running it

```bash
pip install -r requirements.txt
python predictive_maintenance_main.py     # generates data, trains, evaluates
python realtime_processing_system.py      # FastAPI serving layer
```

`gcp_deployment_script.py` expects a configured GCP project and credentials. Model
artefacts (`.h5`, `sensor_scalers.pkl`) are produced by training and are not committed.

### How the labelling works

A window covering hours *t*..*t+23* is labelled 1 when a failure occurs in hours
*t+24*..*t+71*. The observation window and the prediction window never overlap, so the
model is genuinely forecasting rather than recognising a failure it can already see.

Splits are chronological, not shuffled. Consecutive windows overlap by 12 hours, so a
random split would place the same hours in both training and test; and the failure rate
rises over time in this data, so shuffling would let the model see the future. Scalers are
fitted on the training slice alone and reused for validation and test.

## Status

All four layers implemented. Runs end to end locally against synthetic sensor data; the GCP
path needs a real project to execute.

### Notes from a correctness pass

Three methodology defects were fixed, and they are the kind that inflate a score rather
than crash a run:

- **Labels sat inside the observation window.** A window was labelled by whether a failure
  occurred within it, which is detection after the fact. Nothing in the model could have
  forecast anything, whatever the accuracy said. Labels now look 48 hours past the end of
  the window.
- **The split was random.** `train_test_split(..., stratify=y)` on overlapping time windows
  puts the same hours in train and test, and the rising failure rate means shuffling leaks
  the future. Now chronological.
- **Scalers were fitted on everything** before splitting, leaking test-set minima and maxima
  into training. Now fitted on the training slice only.

Also: `deploy_ab_testing_setup` interpolated a bare `project_id` where the class stores
`self.project_id`, raising `NameError`; and `freq='H'` has been removed from pandas.

A split too short to contain one window plus its horizon now raises rather than returning
silently empty arrays.

## Licence

Licensed under the GNU Affero General Public License v3.0. See `LICENSE`.

In short: you may use, modify and redistribute this, including over a network,
provided your derivative is released under the same licence.
