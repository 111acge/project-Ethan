import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import logging
from datetime import datetime
import os


class XGBoostEnsemblePredictor:
    """
    基于XGBoost的集成预测器
    使用多个XGBoost模型进行不同时间尺度的预测
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T', n_estimators: int = 100):
        """
        初始化预测器

        参数:
        lookback_hours: int - 用于预测的历史数据时长（小时）
        forecast_hours: int - 预测时长（小时）
        sample_interval: str - 数据采样间隔，默认30分钟
        n_estimators: int - XGBoost模型中的树的数量
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 初始化XGBoost模型参数
        self.model_params = {
            'n_estimators': n_estimators,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1  # 使用所有可用的CPU核心
        }

        # 创建多个预测模型用于不同的预测时间段
        self.models = [XGBRegressor(**self.model_params) for _ in range(3)]

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        预处理时间序列数据
        """
        try:
            print("Starting data preprocessing...")

            # 准备基础数据
            df = df[['timestamp', 'temperature', 'load']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # 添加时间特征
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['week'] = df['timestamp'].dt.isocalendar().week

            # 添加周期性特征
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

            # 添加滞后特征
            df['load_lag_1h'] = df['load'].shift(2)  # 1小时前
            df['load_lag_24h'] = df['load'].shift(48)  # 24小时前
            df['load_lag_7d'] = df['load'].shift(336)  # 7天前

            # 添加滑动窗口特征
            df['load_rolling_mean_6h'] = df['load'].rolling(window=12).mean()
            df['load_rolling_std_6h'] = df['load'].rolling(window=12).std()
            df['temp_rolling_mean_6h'] = df['temperature'].rolling(window=12).mean()

            # 重采样数据
            df.set_index('timestamp', inplace=True)
            df = df.resample(self.sample_interval).mean()
            # 使用前向填充和后向填充替代插值
            df = df.fillna(method='ffill').fillna(method='bfill')
            df.reset_index(inplace=True)

            # 处理缺失值
            df = df.fillna(method='bfill')

            # 选择特征
            features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                        'temperature', 'temp_rolling_mean_6h',
                        'load_lag_1h', 'load_lag_24h', 'load_lag_7d',
                        'load_rolling_mean_6h', 'load_rolling_std_6h']

            # 标准化数据
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
                X.append(scaled_data[i:(i + self.lookback_points)])
                y.append(scaled_data[(i + self.lookback_points):(i + self.lookback_points + self.forecast_points), -1])

            return np.array(X), np.array(y)

        except Exception as e:
            logging.error(f"Error in sequence creation: {str(e)}")
            raise

    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练集成模型
        """
        try:
            print("Training XGBoost ensemble models...")

            # 将预测时间段分为三段，每个模型负责一段
            segment_length = self.forecast_points // 3

            for i, model in enumerate(self.models):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length if i < 2 else self.forecast_points

                # 为每个模型准备训练数据
                X_flat = X.reshape(X.shape[0], -1)  # 展平特征
                y_segment = y[:, start_idx:end_idx]

                # 使用时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=5)
                for train_idx, val_idx in tscv.split(X_flat):
                    model.fit(
                        X_flat[train_idx],
                        y_segment[train_idx],
                        verbose=0
                    )

                print(f"Model {i + 1} training completed")

        except Exception as e:
            logging.error(f"Error in ensemble training: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        执行完整的预测流程
        """
        try:
            # 数据预处理
            processed_df, scaled_data = self._preprocess_data(df)

            # 创建序列
            X, y = self._create_sequences(scaled_data)
            print(f"Created sequences with shape: X={X.shape}, y={y.shape}")

            # 训练模型
            self._train_ensemble(X, y)

            # 准备预测数据
            last_window = scaled_data[-self.lookback_points:].reshape(1, -1)

            # 生成预测
            print("Generating predictions...")
            predictions = []
            segment_length = self.forecast_points // 3

            for i, model in enumerate(self.models):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length if i < 2 else self.forecast_points
                segment_pred = model.predict(last_window)
                predictions.extend(segment_pred[0])

            # 反标准化预测结果
            dummy_features = np.zeros((len(predictions), scaled_data.shape[1]))
            dummy_features[:, -1] = predictions
            final_predictions = self.scaler.inverse_transform(dummy_features)[:, -1]

            # 创建时间索引
            last_timestamp = processed_df['timestamp'].iloc[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp,
                periods=self.forecast_points + 1,
                freq=self.sample_interval
            )[1:]

            # 创建预测结果DataFrame
            predictions_df = pd.DataFrame({
                'timestamp': future_timestamps,
                'predicted_load': final_predictions
            })

            # 可视化结果
            self._plot_predictions(processed_df, predictions_df, save_dir)

            # 输出预测统计信息
            self._print_prediction_summary(final_predictions)

            return predictions_df

        except Exception as e:
            logging.error(f"Error in prediction process: {str(e)}")
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
                 label='XGBoost Predicted Load', color='red', alpha=0.6)

        plt.title(f'{self.forecast_hours}-hour XGBoost Ensemble Load Forecast')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'xgboost_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print(f"XGBoost prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """
        打印预测结果的统计信息
        """
        print("\nXGBoost Ensemble Prediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")