import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging
import os
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Lambda
from keras import backend as K


class BNNPredictor:
    """
    基于MC Dropout的贝叶斯神经网络时间序列预测器
    主要用于预测电力负载，考虑预测的不确定性
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T'):
        """
        初始化预测器

        参数:
        lookback_hours: int - 用于预测的历史数据时长（小时）
        forecast_hours: int - 预测时长（小时）
        sample_interval: str - 数据采样间隔，默认30分钟
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 初始化模型
        self.model = None
        self.dropout_rate = 0.2
        self.num_monte_carlo = 100

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        预处理时间序列数据
        """
        try:
            print("Starting data preprocessing for BNN...")

            # 准备基础数据
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
                sequence = scaled_data[i:(i + self.lookback_points)].flatten()
                target = scaled_data[(i + self.lookback_points):(i + self.lookback_points + self.forecast_points), -1]

                X.append(sequence)
                y.append(target)

            return np.array(X), np.array(y)

        except Exception as e:
            logging.error(f"Error in sequence creation: {str(e)}")
            raise

    def _build_mc_dropout_model(self, input_shape: int):
        """
        构建带有MC Dropout的神经网络模型
        """
        model = keras.Sequential([
            Dense(128, input_shape=(input_shape,), activation='relu'),
            Dropout(self.dropout_rate),
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.forecast_points, activation='linear')
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae'])

        return model

    def predict(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        执行完整的预测流程
        """
        try:
            # 数据预处理
            processed_df, scaled_data = self._preprocess_data(df)

            # 创建序列
            X, y = self._create_sequences(scaled_data)
            print(f"Training MC Dropout Network with input shape: {X.shape}, output shape: {y.shape}")

            # 构建并训练模型
            if self.model is None:
                self.model = self._build_mc_dropout_model(X.shape[1])
                print("Training MC Dropout model...")
                history = self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2,
                                         verbose=0)

                # 只打印最终的损失值
                final_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                print(f"Training completed - Final loss: {final_loss:.4f}, Validation loss: {final_val_loss:.4f}")

            # 准备最后一个时间窗口用于预测
            last_window = scaled_data[-self.lookback_points:].flatten().reshape(1, -1)

            # 使用MC Dropout进行多次预测
            print("Generating predictions with MC Dropout sampling...")
            predictions_list = []
            for _ in range(self.num_monte_carlo):
                predictions_list.append(self.model(last_window, training=True))
            predictions_array = np.array(predictions_list)

            # 计算均值和标准差
            predictions_mean = np.mean(predictions_array, axis=0)[0]
            predictions_std = np.std(predictions_array, axis=0)[0]

            # 反标准化预测结果
            dummy_features = np.zeros((self.forecast_points, scaled_data.shape[1]))

            # 处理均值
            dummy_features[:, -1] = predictions_mean
            final_predictions_mean = self.scaler.inverse_transform(dummy_features)[:, -1]

            # 处理置信区间
            dummy_features[:, -1] = predictions_mean + 2 * predictions_std
            final_predictions_upper = self.scaler.inverse_transform(dummy_features)[:, -1]

            dummy_features[:, -1] = predictions_mean - 2 * predictions_std
            final_predictions_lower = self.scaler.inverse_transform(dummy_features)[:, -1]

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
                'predicted_load': final_predictions_mean,
                'prediction_lower': final_predictions_lower,
                'prediction_upper': final_predictions_upper
            })

            # 可视化结果
            self._plot_predictions(processed_df, predictions_df, save_dir)

            # 输出预测统计信息
            self._print_prediction_summary(predictions_df)

            return predictions_df

        except Exception as e:
            logging.error(f"Error in prediction process: {str(e)}")
            return None

    def _plot_predictions(self, historical_df: pd.DataFrame,
                          predictions_df: pd.DataFrame,
                          save_dir: Optional[str] = None) -> None:
        """
        绘制预测结果图表，包括置信区间
        """
        plt.figure(figsize=(15, 6))

        # 绘制历史数据
        historical_data = historical_df['load'].iloc[-self.lookback_points:]
        historical_timestamps = historical_df['timestamp'].iloc[-self.lookback_points:]
        plt.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)

        # 绘制预测数据及置信区间
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='BNN Predicted Load', color='red', alpha=0.6)
        plt.fill_between(predictions_df['timestamp'],
                         predictions_df['prediction_lower'],
                         predictions_df['prediction_upper'],
                         color='red', alpha=0.2,
                         label='95% Confidence Interval')

        plt.title(f'{self.forecast_hours}-hour BNN Load Forecast with Uncertainty')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'bnn_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print(f"BNN prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions_df: pd.DataFrame) -> None:
        """
        打印预测结果的统计信息，包括不确定性估计
        """
        predictions = predictions_df['predicted_load'].values
        uncertainty = (predictions_df['prediction_upper'].values -
                       predictions_df['prediction_lower'].values) / 2

        print("\nBNN Prediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")
        print(f"Average Uncertainty: ±{uncertainty.mean():.2f} MW")