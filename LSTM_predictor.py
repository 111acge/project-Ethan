import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging
from datetime import datetime
import os


class LSTMPredictor:
    """
    基于Keras LSTM的时间序列预测器
    主要用于预测电力负载
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
        self.model = None

    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        构建LSTM模型

        参数:
        input_shape: Tuple[int, int] - 输入数据的形状 (时间步长, 特征数)
        """
        self.model = Sequential([
            LSTM(units=64, activation='relu', return_sequences=True,
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32, activation='relu'),
            Dropout(0.2),
            Dense(units=self.forecast_points)
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("LSTM model structure:")
        self.model.summary()

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
                # 保持时间步维度
                sequence = scaled_data[i:(i + self.lookback_points)]
                target = scaled_data[(i + self.lookback_points):(i + self.lookback_points + self.forecast_points), -1]

                X.append(sequence)
                y.append(target)

            return np.array(X), np.array(y)

        except Exception as e:
            logging.error(f"Error in sequence creation: {str(e)}")
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
            print(f"Training LSTM with input shape: {X.shape}, output shape: {y.shape}")

            # 构建模型
            if self.model is None:
                self._build_model(input_shape=(X.shape[1], X.shape[2]))

            # 设置回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]

            # 训练模型
            print("Training LSTM model...")
            history = self.model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )

            # 准备最后一个时间窗口用于预测
            last_window = scaled_data[-self.lookback_points:].reshape(1, self.lookback_points, -1)

            # 生成预测
            print("Generating predictions...")
            future_prediction = self.model.predict(last_window)[0]

            # 反标准化预测结果
            dummy_features = np.zeros((self.forecast_points, scaled_data.shape[1]))
            dummy_features[:, -1] = future_prediction
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
            self._plot_predictions(processed_df, predictions_df, history, save_dir)

            # 输出预测统计信息
            self._print_prediction_summary(final_predictions)

            return predictions_df

        except Exception as e:
            logging.error(f"Error in prediction process: {str(e)}")
            return None

    def _plot_predictions(self, historical_df: pd.DataFrame,
                          predictions_df: pd.DataFrame,
                          history: any,
                          save_dir: Optional[str] = None) -> None:
        """
        绘制预测结果图表和训练历史
        """
        # 创建一个2x1的子图布局
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 绘制预测结果
        historical_data = historical_df['load'].iloc[-self.lookback_points:]
        historical_timestamps = historical_df['timestamp'].iloc[-self.lookback_points:]

        ax1.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)
        ax1.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='LSTM Predicted Load', color='red', alpha=0.6)
        ax1.set_title(f'{self.forecast_hours}-hour LSTM Load Forecast')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Load (MW)')
        ax1.legend()
        ax1.grid(True)
        plt.setp(ax1.get_xticklabels(), rotation=45)

        # 绘制训练历史
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Training History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'lstm_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print(f"LSTM prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """
        打印预测结果的统计信息
        """
        print("\nLSTM Prediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")