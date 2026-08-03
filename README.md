# Predictive Maintenance for IoT on GCP

**Predict equipment failure ahead of time from streaming sensor data, with the training,
serving and deployment paths all written out.**

## What's here

Four modules, each a full layer of the system.

### `predictive_maintenance_main.py` — the core pipeline
`IoTSensorDataGenerator` synthesises realistic multi-sensor signals with injected
degradation, so the pipeline can be exercised without hardware. `TimeSeriesPreprocessor`
handles scaling and windowing into sequences. `PredictiveMaintenanceModel` is the LSTM.
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

## Status

All four layers implemented. Runs end to end locally against synthetic sensor data; the
GCP path needs a real project to execute.

## Licence

All rights reserved. Published for reading, not for reuse.
