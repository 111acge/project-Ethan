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
    基于Random Forest的集成预测器
    主要用于预测电力负载，支持温度作为额外的回归变量
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T'):
        """
        初始化预测器
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 初始化 Random Forest 模型
        self.model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            warm_start=False,
            max_samples=None
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
        执行完整的预测流程
        """
        try:
            # 数据预处理
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

            print(f"Training Random Forest with input shape: {X.shape}, output shape: {y.shape}")

            # 训练模型
            print("Training Random Forest model...")
            start_time = time.time()

            try:
                self.model.fit(X, y)
                training_time = time.time() - start_time
                print(f"Model training completed in {training_time:.2f} seconds")
            except Exception as e:
                print(f"Error: Model training failed - {str(e)}")
                return None

            # 准备最后一个时间窗口用于预测
            try:
                last_window = scaled_data[-self.lookback_points:].flatten().reshape(1, -1)
                if last_window.shape[1] != X.shape[1]:
                    print(f"Error: Input shape mismatch. Expected {X.shape[1]}, got {last_window.shape[1]}")
                    return None
            except Exception as e:
                print(f"Error: Failed to prepare prediction window - {str(e)}")
                return None

            # 生成预测
            print("Generating predictions...")
            try:
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

            # 创建时间索引
            try:
                last_timestamp = processed_df['timestamp'].iloc[-1]
                future_timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=self.forecast_points + 1,
                    freq=self.sample_interval
                )[1:]
            except Exception as e:
                print(f"Error: Failed to create timestamps - {str(e)}")
                return None

            # 创建预测结果DataFrame
            predictions_df = pd.DataFrame({
                'timestamp': future_timestamps,
                'predicted_load': final_predictions
            })

            # 可视化结果
            try:
                self._plot_predictions(processed_df, predictions_df, save_dir)
            except Exception as e:
                print(f"Warning: Failed to create plots - {str(e)}")
                print("Continuing without plotting...")

            # 输出预测统计信息和特征重要性
            self._print_prediction_summary(final_predictions)
            self._print_feature_importance()

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
            print("Random Forest prediction plot saved")
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
        打印特征重要性信息和额外的模型信息
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                print("\nFeature Importance Analysis:")
                importances = self.model.feature_importances_
                std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

                for i, (importance, std_dev) in enumerate(zip(importances, std)):
                    print(f"Feature {i}: Importance = {importance:.4f} (±{std_dev:.4f})")

                print("\nModel Performance Metrics:")
                if hasattr(self.model, 'oob_score_'):
                    print(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        except Exception as e:
            print(f"Warning: Could not print feature importance - {str(e)}")