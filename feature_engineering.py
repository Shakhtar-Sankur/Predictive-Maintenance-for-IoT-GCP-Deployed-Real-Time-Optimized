"""Feature engineering for the sensor stream.

Lifted out of model_training_optimization.py, which imports TensorFlow, Optuna
and MLflow at module level: none of them has anything to do with building
features out of a DataFrame, and requiring a gigabyte of libraries to compute a
rolling mean made this code impossible to test or reuse on its own.

model_training_optimization.py re-exports AdvancedFeatureEngineering, so
existing imports keep working.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class AdvancedFeatureEngineering:
    def __init__(self):
        self.feature_scalers = {}
        self.feature_names = []

    def engineer_features(self, df):
        engineered_df = df.copy()

        feature_groups = []

        engineered_df['temp_vibration_ratio'] = engineered_df['temperature'] / (engineered_df['vibration'] + 1e-8)
        engineered_df['pressure_humidity_product'] = engineered_df['pressure'] * engineered_df['humidity']
        engineered_df['current_temp_interaction'] = engineered_df['current'] * engineered_df['temperature']
        feature_groups.extend(['temp_vibration_ratio', 'pressure_humidity_product', 'current_temp_interaction'])

        for sensor_id in engineered_df['sensor_id'].unique():
            sensor_mask = engineered_df['sensor_id'] == sensor_id
            sensor_data = engineered_df[sensor_mask].copy()

            if len(sensor_data) > 1:
                for col in ['temperature', 'vibration', 'pressure', 'humidity', 'current']:
                    engineered_df.loc[sensor_mask, f'{col}_rolling_mean_3'] = sensor_data[col].rolling(3, min_periods=1).mean()
                    engineered_df.loc[sensor_mask, f'{col}_rolling_std_3'] = sensor_data[col].rolling(3, min_periods=1).std().fillna(0)
                    engineered_df.loc[sensor_mask, f'{col}_rolling_mean_6'] = sensor_data[col].rolling(6, min_periods=1).mean()
                    engineered_df.loc[sensor_mask, f'{col}_rolling_std_6'] = sensor_data[col].rolling(6, min_periods=1).std().fillna(0)

                    feature_groups.extend([
                        f'{col}_rolling_mean_3', f'{col}_rolling_std_3',
                        f'{col}_rolling_mean_6', f'{col}_rolling_std_6'
                    ])

        for col in ['temperature', 'vibration', 'pressure', 'humidity', 'current']:
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                percentile_val = np.percentile(engineered_df[col], p)
                engineered_df[f'{col}_above_p{p}'] = (engineered_df[col] > percentile_val).astype(int)
                feature_groups.append(f'{col}_above_p{p}')

        engineered_df['hour'] = pd.to_datetime(engineered_df['timestamp']).dt.hour
        engineered_df['day_of_week'] = pd.to_datetime(engineered_df['timestamp']).dt.dayofweek
        engineered_df['hour_sin'] = np.sin(2 * np.pi * engineered_df['hour'] / 24)
        engineered_df['hour_cos'] = np.cos(2 * np.pi * engineered_df['hour'] / 24)
        engineered_df['dow_sin'] = np.sin(2 * np.pi * engineered_df['day_of_week'] / 7)
        engineered_df['dow_cos'] = np.cos(2 * np.pi * engineered_df['day_of_week'] / 7)
        feature_groups.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])

        sensor_failure_rates = engineered_df.groupby('sensor_id')['failure'].mean()
        engineered_df['sensor_failure_rate'] = engineered_df['sensor_id'].map(sensor_failure_rates)
        feature_groups.append('sensor_failure_rate')

        self.feature_names = (['temperature', 'vibration', 'pressure', 'humidity', 'current'] +
                             feature_groups)

        return engineered_df

    def prepare_features(self, df, fit_scalers=True):
        if fit_scalers:
            for feature in self.feature_names:
                if feature in df.columns:
                    scaler = RobustScaler()
                    df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])
                    self.feature_scalers[feature] = scaler
        else:
            for feature in self.feature_names:
                if feature in df.columns and feature in self.feature_scalers:
                    df[f'{feature}_scaled'] = self.feature_scalers[feature].transform(df[[feature]])

        return df
