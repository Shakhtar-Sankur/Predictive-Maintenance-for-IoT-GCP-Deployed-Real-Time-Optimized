"""Features built from the sensor stream, checked against hand-worked answers.

The sample frame is deliberate: two sensors, interleaved in time, with values
chosen so every derived column can be computed by hand. That is what makes it
possible to catch the mistake this kind of code invites — one sensor's history
leaking into another's rolling window.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from feature_engineering import AdvancedFeatureEngineering


@pytest.fixture
def frame():
    """Two sensors, six readings each, interleaved."""
    rows = []
    for index in range(6):
        rows.append({
            "sensor_id": "A", "timestamp": f"2026-01-01 0{index}:00:00",
            "temperature": 10.0 * (index + 1), "vibration": 2.0, "pressure": 5.0,
            "humidity": 4.0, "current": 3.0, "failure": 1 if index >= 4 else 0,
        })
        rows.append({
            "sensor_id": "B", "timestamp": f"2026-01-01 0{index}:00:00",
            "temperature": 100.0, "vibration": 4.0, "pressure": 2.0,
            "humidity": 10.0, "current": 1.0, "failure": 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def engineered(frame):
    return AdvancedFeatureEngineering().engineer_features(frame)


def test_the_original_frame_is_not_modified(frame):
    before = frame.copy(deep=True)
    AdvancedFeatureEngineering().engineer_features(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_the_interaction_features_are_what_they_say(engineered):
    row = engineered.iloc[0]
    assert row["temp_vibration_ratio"] == pytest.approx(row["temperature"] / (row["vibration"] + 1e-8))
    assert row["pressure_humidity_product"] == pytest.approx(row["pressure"] * row["humidity"])
    assert row["current_temp_interaction"] == pytest.approx(row["current"] * row["temperature"])


def test_dividing_by_a_zero_reading_does_not_produce_infinity():
    frame = pd.DataFrame([{
        "sensor_id": "A", "timestamp": "2026-01-01 00:00:00", "temperature": 50.0,
        "vibration": 0.0, "pressure": 1.0, "humidity": 1.0, "current": 1.0, "failure": 0,
    }])
    out = AdvancedFeatureEngineering().engineer_features(frame)
    assert np.isfinite(out["temp_vibration_ratio"]).all(), "a stationary sensor reads 0 vibration"


def test_rolling_windows_stay_inside_one_sensor(engineered):
    """Sensor B sits at a constant 100; A climbs 10..60. Mixing them would show."""
    b_rows = engineered[engineered["sensor_id"] == "B"]
    assert np.allclose(b_rows["temperature_rolling_mean_3"], 100.0), \
        "sensor A's readings leaked into sensor B's rolling mean"
    assert np.allclose(b_rows["temperature_rolling_std_3"], 0.0), \
        "a constant sensor cannot have a rolling standard deviation"


def test_the_rolling_mean_is_the_mean_of_the_last_three_readings(engineered):
    a_rows = engineered[engineered["sensor_id"] == "A"].reset_index(drop=True)
    # temperatures are 10, 20, 30, 40, 50, 60
    assert a_rows.loc[0, "temperature_rolling_mean_3"] == pytest.approx(10.0)
    assert a_rows.loc[1, "temperature_rolling_mean_3"] == pytest.approx(15.0)
    assert a_rows.loc[2, "temperature_rolling_mean_3"] == pytest.approx(20.0)
    assert a_rows.loc[5, "temperature_rolling_mean_3"] == pytest.approx(50.0)


def test_the_first_reading_has_a_zero_not_a_nan_standard_deviation(engineered):
    for column in [c for c in engineered.columns if c.endswith("_rolling_std_3")]:
        assert engineered[column].notna().all(), f"{column} still carries NaN into the model"


def test_the_hour_is_encoded_as_a_circle(engineered):
    """Midnight and 23:00 must be neighbours, which is the point of sin/cos."""
    assert np.allclose(engineered["hour_sin"] ** 2 + engineered["hour_cos"] ** 2, 1.0)
    midnight = engineered[pd.to_datetime(engineered["timestamp"]).dt.hour == 0].iloc[0]
    assert midnight["hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert midnight["hour_cos"] == pytest.approx(1.0)


def test_the_day_of_week_is_encoded_as_a_circle(engineered):
    assert np.allclose(engineered["dow_sin"] ** 2 + engineered["dow_cos"] ** 2, 1.0)


def test_percentile_flags_are_zero_or_one_and_not_all_the_same(engineered):
    flags = [c for c in engineered.columns if "_above_p" in c]
    assert flags, "no percentile features produced"
    for column in flags:
        assert set(engineered[column].unique()).issubset({0, 1})
    assert engineered["temperature_above_p50"].nunique() == 2, \
        "a median flag that never changes carries no information"


def test_feature_names_are_recorded_and_all_exist(engineered):
    engineer = AdvancedFeatureEngineering()
    out = engineer.engineer_features(engineered)
    assert engineer.feature_names
    missing = [name for name in engineer.feature_names if name not in out.columns]
    assert not missing, f"feature_names lists columns that were never built: {missing}"


def test_scalers_are_fitted_once_and_reused(frame):
    """Scaling validation data must reuse the training scaler, not refit it."""
    engineer = AdvancedFeatureEngineering()
    train = engineer.engineer_features(frame)
    engineer.prepare_features(train, fit_scalers=True)
    assert engineer.feature_scalers, "no scalers were kept"

    later = engineer.engineer_features(frame.copy())
    later["temperature"] = later["temperature"] * 3      # a shifted distribution
    scaled = engineer.prepare_features(later, fit_scalers=False)

    assert "temperature_scaled" in scaled.columns
    # Refitting would re-centre the shifted data; reusing the scaler must not.
    assert abs(float(scaled["temperature_scaled"].median())) > 0.5, \
        "the scaler was refitted on the new data instead of being reused"


def test_sensor_failure_rate_is_that_sensors_own_rate(engineered):
    """Documented behaviour, and a caution: this reads the label column.

    Sensor A fails in 2 of its 6 readings, B in none. The feature is computed
    over the whole frame, including the row it is attached to, so it must only
    be built inside a training split — see the README's note on leakage.
    """
    rates = engineered.groupby("sensor_id")["sensor_failure_rate"].first()
    assert rates["A"] == pytest.approx(2 / 6)
    assert rates["B"] == pytest.approx(0.0)
