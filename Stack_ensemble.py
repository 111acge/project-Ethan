import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
import os


class KerasLSTMWrapper(BaseEstimator, RegressorMixin):
    """
    包装Keras LSTM模型使其符合sklearn的接口
    """

    def __init__(self, lookback_points=48, n_features=4):
        self.lookback_points = lookback_points
        self.n_features = n_features
        self.model = None

    def _build_model(self):
        model = Sequential([
            LSTM(units=64, activation='relu', return_sequences=True,
                 input_shape=(self.lookback_points, self.n_features)),
            LSTM(units=32, activation='relu'),
            Dense(48)  # 假设预测未来48个时间点
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, y):
        X_reshaped = X.reshape(-1, self.lookback_points, self.n_features)
        self.model = self._build_model()
        self.model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        X_reshaped = X.reshape(-1, self.lookback_points, self.n_features)
        return self.model.predict(X_reshaped)


class StackingEnsemblePredictor:
    """
    基于Stacking的集成预测器
    使用XGBoost、Random Forest和LSTM作为基础模型，XGBoost作为元模型
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T'):
        """
        初始化 Stacking 集成预测器
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 基础模型配置
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )),
            ('xgb', XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                n_jobs=-1,
                random_state=42
            )),
            ('lstm', KerasLSTMWrapper(
                lookback_points=self.lookback_points,
                n_features=4
            ))
        ]

        # 元模型使用 XGBoost
        meta_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=4,
            n_jobs=-1
        )

        # 创建 Stacking 模型
        self.model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            verbose=0
        )

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        预处理时间序列数据
        """
        try:
            print("Starting data preprocessing...")

            # 基础数据准备
            df = df[['timestamp', 'temperature', 'load']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # 添加时间特征
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # 重采样数据
            df.set_index('timestamp', inplace=True)
            df = df.resample(self.sample_interval).mean().interpolate(method='time')
            df.reset_index(inplace=True)

            # 标准化数据
            features = ['hour_sin', 'hour_cos', 'temperature', 'load']
            scaled_data = self.scaler.fit_transform(df[features])

            return df, scaled_data

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _create_sequences(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列的输入序列和目标序列
        """
        try:
            X, y = [], []
            for i in range(len(scaled_data) - self.lookback_points - self.forecast_points):
                sequence = scaled_data[i:(i + self.lookback_points)].flatten()  # 展平特征
                target = scaled_data[(i + self.lookback_points):(i + self.lookback_points + self.forecast_points), -1]

                X.append(sequence)
                y.append(target)

            return np.array(X), np.array(y)

        except Exception as e:
            logging.error(f"Error in sequence creation: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        执行预测
        """
        try:
            print("Starting data preprocessing...")
            processed_df, scaled_data = self._preprocess_data(df)
            if processed_df is None or scaled_data is None:
                print("Error: Data preprocessing failed")
                return None

            # 创建序列
            X, y = self._create_sequences(scaled_data)
            if len(X) == 0 or len(y) == 0:
                print("Error: No sequences created")
                return None

            print(f"Training Stacking Ensemble with input shape: {X.shape}, output shape: {y.shape}")

            # 训练模型
            print("Training Stacking Ensemble model...")
            print("This may take a while as it involves cross-validation and multiple models...")

            start_time = time.time()
            try:
                self.model.fit(X, y)
                training_time = time.time() - start_time
                print(f"Model training completed in {training_time:.2f} seconds")
            except Exception as e:
                print(f"Error: Model training failed - {str(e)}")
                return None

            # 预测
            print("Generating predictions...")
            try:
                last_window = scaled_data[-self.lookback_points:].flatten().reshape(1, -1)
                future_prediction = self.model.predict(last_window)
            except Exception as e:
                print(f"Error: Prediction failed - {str(e)}")
                return None

            # 反标准化预测结果
            try:
                dummy_features = np.zeros((self.forecast_points, scaled_data.shape[1]))
                dummy_features[:, -1] = future_prediction
                final_predictions = self.scaler.inverse_transform(dummy_features)[:, -1]
            except Exception as e:
                print(f"Error: Failed to inverse transform predictions - {str(e)}")
                return None

            # 创建时间索引和预测结果 DataFrame
            try:
                last_timestamp = processed_df['timestamp'].iloc[-1]
                future_timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=self.forecast_points + 1,
                    freq=self.sample_interval
                )[1:]

                predictions_df = pd.DataFrame({
                    'timestamp': future_timestamps,
                    'predicted_load': final_predictions
                })
            except Exception as e:
                print(f"Error: Failed to create prediction dataframe - {str(e)}")
                return None

            # 可视化结果
            try:
                self._plot_predictions(processed_df, predictions_df, save_dir)
            except Exception as e:
                print(f"Warning: Failed to create plots - {str(e)}")
                print("Continuing without plotting...")

            # 打印预测统计和模型信息
            self._print_prediction_summary(final_predictions)
            self._print_model_information()

            return predictions_df

        except Exception as e:
            print(f"Error in prediction process: {str(e)}")
            return None

    def _plot_predictions(self, historical_df: pd.DataFrame,
                          predictions_df: pd.DataFrame,
                          save_dir: Optional[str] = None) -> None:
        """
        绘制预测结果图表
        """
        plt.figure(figsize=(15, 6))

        # 绘制历史数据
        historical_data = historical_df['load'].iloc[-self.lookback_points:]
        historical_timestamps = historical_df['timestamp'].iloc[-self.lookback_points:]
        plt.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)

        # 绘制预测数据
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='Stacking Ensemble Predicted Load', color='red', alpha=0.6)

        plt.title(f'{self.forecast_hours}-hour Stacking Ensemble Load Forecast')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'stacking_ensemble_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print("Stacking Ensemble prediction plot saved")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """
        打印预测结果统计信息
        """
        print("\nStacking Ensemble Prediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")

    def _print_model_information(self) -> None:
        """
        打印模型组成信息
        """
        print("\nStacking Ensemble Model Configuration:")
        print("Base Models:")
        for name, estimator in self.model.estimators_:
            print(f"- {name}: {type(estimator).__name__}")

        print("\nMeta Model:")
        print(f"- {type(self.model.final_estimator_).__name__}")
        print(f"- Cross-validation folds: {self.model.cv}")