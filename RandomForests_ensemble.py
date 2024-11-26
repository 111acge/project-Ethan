import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
import os


class RandomForestsPredictor:
    """
    基于sklearn RandomForest的时间序列预测器
    主要用于预测电力负载，支持温度作为额外的回归变量
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

        # 初始化RandomForest模型
        self.model = RandomForestRegressor(
            n_estimators=100,  # 树的数量
            max_depth=None,  # 树的最大深度，None表示不限制
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,  # 使用所有可用的CPU核心
            verbose=1
        )

        # 存储最佳参数
        self.best_params = None

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        预处理时间序列数据

        参数:
        df: DataFrame - 包含 'timestamp', 'temperature', 'load' 列的数据框

        返回:
        Tuple[pd.DataFrame, np.ndarray] - 处理后的数据框和标准化后的特征数据
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
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

            # 重采样数据
            df.set_index('timestamp', inplace=True)
            df = df.resample(self.sample_interval).mean().interpolate(method='time')
            df.reset_index(inplace=True)

            # 创建滚动特征
            df['load_lag_24h'] = df['load'].shift(48)  # 24小时滞后
            df['load_lag_1w'] = df['load'].shift(336)  # 一周滞后
            df['temp_lag_24h'] = df['temperature'].shift(48)  # 24小时温度滞后

            # 创建移动平均特征
            df['load_ma_24h'] = df['load'].rolling(window=48).mean()
            df['temp_ma_24h'] = df['temperature'].rolling(window=48).mean()

            # 填充缺失值
            df = df.fillna(method='bfill')

            # 标准化特征
            features = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                        'temperature', 'temp_lag_24h', 'temp_ma_24h',
                        'load', 'load_lag_24h', 'load_lag_1w', 'load_ma_24h']

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
                # 将所有特征展平作为输入
                sequence = scaled_data[i:(i + self.lookback_points)].flatten()
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

        参数:
        df: DataFrame - 输入数据
        save_dir: Optional[str] - 保存预测结果的目录

        返回:
        Optional[pd.DataFrame] - 预测结果数据框
        """
        try:
            # 数据预处理
            processed_df, scaled_data = self._preprocess_data(df)

            # 创建序列
            X, y = self._create_sequences(scaled_data)
            print(f"Training Random Forest with input shape: {X.shape}, output shape: {y.shape}")

            # 训练模型
            print("Training Random Forest model...")
            self.model.fit(X, y)
            print("Model training completed")

            # 准备最后一个时间窗口用于预测
            last_window = scaled_data[-self.lookback_points:].flatten().reshape(1, -1)

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
            self._plot_predictions(processed_df, predictions_df, save_dir)

            # 输出预测统计信息
            self._print_prediction_summary(final_predictions)

            # 输出特征重要性
            self._print_feature_importance()

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
                 label='Random Forest Predicted Load', color='red', alpha=0.6)

        plt.title(f'{self.forecast_hours}-hour Random Forest Load Forecast')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'random_forest_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Random Forest prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """
        打印预测结果的统计信息
        """
        print("\nRandom Forest Prediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")

    def _print_feature_importance(self) -> None:
        """
        打印特征重要性信息
        """
        feature_importance = self.model.feature_importances_
        print("\nFeature Importance:")
        for importance in feature_importance:
            print(f"Feature importance: {importance:.4f}")