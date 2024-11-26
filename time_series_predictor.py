import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from typing import Optional, Tuple, List
import logging
from datetime import datetime
import os


class TimeSeriesPredictor:
    """
    基于相似模式的时间序列预测器
    主要用于预测电力负载，但也可扩展用于其他时间序列预测任务
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T', n_neighbors: int = 5):
        """
        初始化预测器

        参数:
        lookback_hours: int - 用于预测的历史数据时长（小时）
        forecast_hours: int - 预测时长（小时）
        sample_interval: str - 数据采样间隔，默认30分钟
        n_neighbors: int - 用于预测的相似模式数量
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.n_neighbors = n_neighbors
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

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

            # 确保数据按时间排序
            df = df[['timestamp', 'temperature', 'load']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # 重采样数据
            df.set_index('timestamp', inplace=True)
            df = df.resample(self.sample_interval).mean().interpolate(method='time')
            df.reset_index(inplace=True)

            # 添加时间特征
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek

            # 标准化数据
            features = ['hour', 'dayofweek', 'temperature', 'load']
            scaled_data = self.scaler.fit_transform(df[features])

            return df, scaled_data

        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _create_sequences(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列的输入序列和目标序列

        参数:
        scaled_data: np.ndarray - 标准化后的特征数据

        返回:
        Tuple[np.ndarray, np.ndarray] - 输入序列和对应的目标序列
        """
        try:
            X = []
            y = []

            for i in range(len(scaled_data) - self.lookback_points - self.forecast_points):
                X.append(scaled_data[i:(i + self.lookback_points)])
                y.append(scaled_data[(i + self.lookback_points):(i + self.lookback_points + self.forecast_points), -1])

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error in sequence creation: {str(e)}")
            raise

    def _predict_similar_pattern(self, test_window: np.ndarray,
                                 train_windows: np.ndarray,
                                 train_targets: np.ndarray) -> np.ndarray:
        """
        基于相似模式进行预测

        参数:
        test_window: np.ndarray - 用于预测的测试窗口
        train_windows: np.ndarray - 训练数据窗口
        train_targets: np.ndarray - 训练数据目标值

        返回:
        np.ndarray - 预测结果
        """
        try:
            # 计算与所有训练窗口的相似度
            similarities = np.array([
                euclidean(test_window.flatten(), train_window.flatten())
                for train_window in train_windows
            ])

            # 找到最相似的k个窗口
            similar_indices = np.argsort(similarities)[:self.n_neighbors]
            weights = 1 / (similarities[similar_indices] + 1e-6)
            weights /= np.sum(weights)

            # 加权平均预测
            return np.average(train_targets[similar_indices], weights=weights, axis=0)

        except Exception as e:
            self.logger.error(f"Error in pattern prediction: {str(e)}")
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

            # 划分训练集
            train_size = int(len(X) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]

            print("Preparing final prediction...")
            # 使用最后一个时间窗口进行预测
            last_window = scaled_data[-self.lookback_points:]
            future_prediction = self._predict_similar_pattern(last_window, X_train, y_train)

            # 准备反标准化
            dummy_features = np.zeros((self.forecast_points, scaled_data.shape[1]))
            dummy_features[:, -1] = future_prediction

            # 反标准化预测结果
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
            self.logger.error(f"Error in prediction process: {str(e)}")
            return None

    def _plot_predictions(self, historical_df: pd.DataFrame,
                          predictions_df: pd.DataFrame,
                          save_dir: Optional[str] = None) -> None:
        """
        绘制预测结果图表

        参数:
        historical_df: pd.DataFrame - 历史数据
        predictions_df: pd.DataFrame - 预测数据
        save_dir: Optional[str] - 保存图表的目录
        """
        plt.figure(figsize=(15, 6))

        # 绘制最后一个时间窗口的历史数据
        historical_data = historical_df['load'].iloc[-self.lookback_points:]
        historical_timestamps = historical_df['timestamp'].iloc[-self.lookback_points:]
        plt.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)

        # 绘制预测数据
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='Predicted Load', color='red', alpha=0.6)

        plt.title(f'{self.forecast_hours}-hour Load Forecast ({self.sample_interval[:-1]} minute intervals)')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'time_series_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved in {save_dir} dir")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """
        打印预测结果的统计信息

        参数:
        predictions: np.ndarray - 预测结果数组
        """
        print("\nPrediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")


