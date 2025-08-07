import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Average
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
import os


class RandomForestsPredictor:
    """
    基于TensorFlow的Random Forests预测器实现
    主要用于预测电力负载
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T', n_estimators: int = 12,
                 subsample_size: float = 0.8):
        """初始化预测器"""
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.n_estimators = n_estimators
        self.subsample_size = subsample_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 检测GPU
        if tf.config.list_physical_devices('GPU'):
            logging.info("RandomForests using GPU")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # 初始化模型
        self.models = []
        self.ensemble_model = None
        self.feature_importance = None

    def _build_tree_model(self) -> tf.keras.Model:
        """构建单个树模型"""
        with tf.device('/device:GPU:0'):
            model = Sequential([
                Dense(128, input_shape=(self.lookback_points * 4,)),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                Dropout(0.2),

                Dense(64),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                Dropout(0.2),

                Dense(32),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                Dropout(0.2),

                Dense(self.forecast_points)
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

            return model

    def _build_ensemble_model(self):
        """构建集成模型"""
        inputs = Input(shape=(self.lookback_points * 4,))
        outputs = []

        # 创建n_estimators个独立的树模型
        for _ in range(self.n_estimators):
            tree_model = self._build_tree_model()
            self.models.append(tree_model)
            outputs.append(tree_model(inputs))

        # 平均所有树的输出
        if len(outputs) > 1:
            avg_output = Average()(outputs)
        else:
            avg_output = outputs[0]

        self.ensemble_model = Model(inputs=inputs, outputs=avg_output)
        self.ensemble_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成bootstrap样本"""
        n_samples = int(len(X) * self.subsample_size)
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        return X[indices], y[indices]

    def _calculate_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """计算特征重要性"""
        importances = np.zeros(X.shape[1])

        # 使用所有树的输入权重平均值作为特征重要性
        for model in self.models:
            weights = model.layers[0].get_weights()[0]
            importances += np.mean(np.abs(weights), axis=1)

        # 归一化
        importances = importances / (len(self.models) * np.sum(importances))
        return importances

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """预处理数据"""
        try:
            logging.info("Starting data preprocessing...")

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
        """创建序列数据"""
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

    def predict(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """执行预测"""
        try:
            # 数据预处理
            processed_df, scaled_data = self._preprocess_data(df)

            # 创建序列
            X, y = self._create_sequences(scaled_data)
            logging.info(f"Training RandomForests with input shape: {X.shape}, output shape: {y.shape}")

            # 构建模型
            if not self.ensemble_model:
                self._build_ensemble_model()

            # 分别训练每棵树
            logging.info("Training individual trees...")
            for i, model in enumerate(self.models):
                X_boot, y_boot = self._bootstrap_sample(X, y)
                model.fit(
                    X_boot, y_boot,
                    epochs=10,
                    batch_size=128,
                    verbose=0
                )
                if (i + 1) % 3 == 0:
                    logging.info(f"Trained {i + 1}/{self.n_estimators} trees")

            # 计算特征重要性
            self.feature_importance = self._calculate_feature_importance(X)

            # 准备预测
            last_window = scaled_data[-self.lookback_points:].flatten().reshape(1, -1)
            future_prediction = self.ensemble_model.predict(last_window)[0]

            # 反标准化
            dummy_features = np.zeros((self.forecast_points, scaled_data.shape[1]))
            dummy_features[:, -1] = future_prediction
            final_predictions = self.scaler.inverse_transform(dummy_features)[:, -1]

            # 生成时间索引
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

            # 输出预测统计信息和特征重要性
            self._print_prediction_summary(final_predictions)
            self._print_feature_importance()

            return predictions_df

        except Exception as e:
            logging.error(f"Error in prediction process: {str(e)}")
            return None

    def _plot_predictions(self, historical_df: pd.DataFrame,
                          predictions_df: pd.DataFrame,
                          save_dir: Optional[str] = None) -> None:
        """绘制预测结果"""
        plt.figure(figsize=(15, 6))

        historical_data = historical_df['load'].iloc[-self.lookback_points:]
        historical_timestamps = historical_df['timestamp'].iloc[-self.lookback_points:]
        plt.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)

        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='RandomForests Predicted Load', color='red', alpha=0.6)

        plt.title(f'{self.forecast_hours}-hour RandomForests Load Forecast')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'randomforests_prediction.png'),
                        dpi=300, bbox_inches='tight')
            logging.info("RandomForests prediction plot saved")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """打印预测结果统计信息"""
        logging.info("TF-RandomForests Prediction Summary:")
        logging.info(f"Mean Load: {predictions.mean():.2f} MW")
        logging.info(f"Standard Deviation: {predictions.std():.2f} MW")
        logging.info(f"Maximum Load: {predictions.max():.2f} MW")
        logging.info(f"Minimum Load: {predictions.min():.2f} MW")

    def _print_feature_importance(self) -> None:
        """打印特征重要性信息"""
        if self.feature_importance is not None:
            logging.info("Feature Importance Analysis:")
            features = ['hour_sin', 'hour_cos', 'temperature', 'load']
            for feature, importance in zip(features, self.feature_importance):
                logging.info(f"{feature}: {importance:.4f}")

    def predict_with_split(self, X_train, y_train, X_test, test_df: pd.DataFrame, save_dir: Optional[str] = None) -> \
            Optional[pd.DataFrame]:
        """
        使用预先划分的训练集和测试集进行预测和评估

        参数:
        X_train: np.ndarray - 训练特征
        y_train: np.ndarray - 训练目标
        X_test: np.ndarray - 测试特征
        test_df: pd.DataFrame - 包含测试数据的原始DataFrame
        save_dir: Optional[str] - 保存目录

        返回:
        Optional[pd.DataFrame] - 包含预测结果的DataFrame
        """
        try:
            # 构建模型（如果尚未构建）
            if not self.ensemble_model:
                self._build_ensemble_model()

            # 训练回调函数
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # 分别训练每棵树
            logging.info("Training individual trees...")
            for i, model in enumerate(self.models):
                # 生成bootstrap样本
                X_boot, y_boot = self._bootstrap_sample(X_train, y_train)
                model.fit(
                    X_boot, y_boot,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                if (i + 1) % 3 == 0:
                    logging.info(f"Trained {i + 1}/{self.n_estimators} trees")

            # 计算特征重要性
            self.feature_importance = self._calculate_feature_importance(X_train)

            # 生成预测
            logging.info("Generating predictions on test data...")
            predictions = self.ensemble_model.predict(X_test)

            # 创建预测结果DataFrame
            predictions_df = pd.DataFrame({
                'timestamp': test_df['timestamp'],
                'predicted_load': predictions.flatten()[:len(test_df)]
            })

            # 可视化结果
            self._plot_predictions_with_split(test_df, predictions_df, save_dir)

            # 输出预测统计信息和特征重要性
            self._print_prediction_summary(predictions.flatten())
            self._print_feature_importance()

            return predictions_df

        except Exception as e:
            logging.error(f"Error in prediction process with split: {str(e)}")
            return None

    def _plot_predictions_with_split(self, test_df: pd.DataFrame,
                                     predictions_df: pd.DataFrame,
                                     save_dir: Optional[str] = None) -> None:
        """
        绘制基于训练测试集划分的预测结果图表

        参数:
        test_df: pd.DataFrame - 测试数据
        predictions_df: pd.DataFrame - 预测结果
        save_dir: Optional[str] - 保存目录
        """
        # 设置图表样式
        plt.style.use('classic')
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'figure.dpi': 300
        })

        # 创建单一图表
        plt.figure(figsize=(15, 6))

        # 绘制预测结果与实际值的对比
        plt.plot(test_df['timestamp'], test_df['load'],
                 label='Actual Load', color='blue', alpha=0.6)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='RandomForests Predicted Load', color='red', alpha=0.6)
        plt.title('RandomForests Load Forecast with Train-Test Split')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'randomforests_prediction_split.png'),
                        dpi=300, bbox_inches='tight')
            logging.info(f"RandomForests prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()