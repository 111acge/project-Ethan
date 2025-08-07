import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D
from tensorflow.keras.layers import MaxPooling1D, Flatten, Concatenate, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging
import os


class StackCNNXGBoostPredictor:
    """
    基于CNN和XGBoost的Stack2集成预测器
    用于预测电力负载
    """

    def __init__(self, lookback_hours: int = 48, forecast_hours: int = 48,
                 sample_interval: str = '30T'):
        """初始化预测器"""
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.lookback_points = int(lookback_hours * 60 / int(sample_interval[:-1]))
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 检测GPU
        if tf.config.list_physical_devices('GPU'):
            logging.info("Stack2 using GPU")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.model = None

    def _build_cnn_branch(self, input_layer):
        """构建CNN分支"""
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Dropout(0.2)(x)
        return x

    def _build_xgboost_branch(self, input_layer):
        """构建XGBoost分支"""
        x = Dense(1000)(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        return x

    def _build_model(self):
        """构建完整的Stack2模型"""
        # CNN分支输入
        cnn_input = Input(shape=(self.lookback_points, 4))
        cnn_output = self._build_cnn_branch(cnn_input)

        # XGBoost分支输入
        xgb_input = Input(shape=(self.lookback_points * 4,))
        xgb_output = self._build_xgboost_branch(xgb_input)

        # 合并两个分支
        merged = Concatenate()([cnn_output, xgb_output])

        # 元学习器
        x = Dense(128)(merged)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.forecast_points)(x)

        # 构建模型
        model = Model(inputs=[cnn_input, xgb_input], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return model

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """数据预处理"""
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

    def _create_sequences(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建序列数据"""
        try:
            X_cnn, X_xgb, y = [], [], []
            for i in range(len(scaled_data) - self.lookback_points - self.forecast_points):
                sequence = scaled_data[i:(i + self.lookback_points)]
                target = scaled_data[(i + self.lookback_points):(i + self.lookback_points + self.forecast_points), -1]

                X_cnn.append(sequence)
                X_xgb.append(sequence.flatten())
                y.append(target)

            return np.array(X_cnn), np.array(X_xgb), np.array(y)

        except Exception as e:
            logging.error(f"Error in sequence creation: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """执行预测"""
        try:
            # 数据预处理
            processed_df, scaled_data = self._preprocess_data(df)
            X_cnn, X_xgb, y = self._create_sequences(scaled_data)

            logging.info(
                f"Training Stack2 with input shapes: CNN={X_cnn.shape}, XGB={X_xgb.shape}, output shape={y.shape}")

            # 构建模型
            if self.model is None:
                self.model = self._build_model()

            # 训练模型
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = self.model.fit(
                [X_cnn, X_xgb], y,
                epochs=100,
                batch_size=128,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )

            # 预测
            last_sequence = scaled_data[-self.lookback_points:]
            last_sequence_cnn = last_sequence.reshape(1, self.lookback_points, 4)
            last_sequence_xgb = last_sequence.flatten().reshape(1, -1)

            future_prediction = self.model.predict([last_sequence_cnn, last_sequence_xgb])[0]

            # 反标准化
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
        """绘制预测结果和训练历史"""
        plt.style.use('classic')
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'figure.dpi': 300
        })

        # 创建2x1的子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 绘制预测结果
        historical_data = historical_df['load'].iloc[-self.lookback_points:]
        historical_timestamps = historical_df['timestamp'].iloc[-self.lookback_points:]

        ax1.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)
        ax1.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='Stack2 Predicted Load', color='red', alpha=0.6)
        ax1.set_title(f'{self.forecast_hours}-hour Stack2 Load Forecast')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Load (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 绘制训练历史
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Training History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'stack2_prediction.png'),
                        dpi=300, bbox_inches='tight')
            logging.info(f"Stack2 prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """打印预测结果统计信息"""
        logging.info("Stack2 Prediction Summary:")
        logging.info(f"Mean Load: {predictions.mean():.2f} MW")
        logging.info(f"Standard Deviation: {predictions.std():.2f} MW")
        logging.info(f"Maximum Load: {predictions.max():.2f} MW")
        logging.info(f"Minimum Load: {predictions.min():.2f} MW")

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
            if self.model is None:
                self.model = self._build_model()

            # 准备模型输入数据
            # 注：我们需要为CNN和XGBoost分支准备不同格式的输入
            X_cnn = []
            X_xgb = []

            # 假设X_train和X_test是已经处理好的特征
            # 需要进一步处理以适配模型输入需求

            # 重新处理训练数据
            for i in range(len(X_train)):
                # 假设每条数据是一个展平的时间窗口
                # 需要将其重塑为CNN需要的形状
                sequence = X_train[i].reshape(-1, 4)  # 尝试重塑为(lookback_points, 4)
                X_cnn.append(sequence)
                X_xgb.append(X_train[i])  # XGBoost分支使用展平的特征

            X_cnn_train = np.array(X_cnn)
            X_xgb_train = np.array(X_xgb)

            # 处理测试数据
            X_cnn_test = []
            X_xgb_test = []
            for i in range(len(X_test)):
                sequence = X_test[i].reshape(-1, 4)
                X_cnn_test.append(sequence)
                X_xgb_test.append(X_test[i])

            X_cnn_test = np.array(X_cnn_test)
            X_xgb_test = np.array(X_xgb_test)

            # 设置早停回调函数
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # 训练模型
            logging.info("Training Stack2 (CNN+XGBoost) model with split data...")
            history = self.model.fit(
                [X_cnn_train, X_xgb_train], y_train,
                epochs=100,
                batch_size=128,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            logging.info("Model training completed")

            # 生成预测
            logging.info("Generating predictions on test data...")
            predictions = self.model.predict([X_cnn_test, X_xgb_test])

            # 创建预测结果DataFrame
            predictions_df = pd.DataFrame({
                'timestamp': test_df['timestamp'],
                'predicted_load': predictions.flatten()[:len(test_df)]
            })

            # 可视化结果
            self._plot_predictions_with_split(test_df, predictions_df, history, save_dir)

            # 输出预测统计信息
            self._print_prediction_summary(predictions.flatten())

            return predictions_df

        except Exception as e:
            logging.error(f"Error in prediction process with split: {str(e)}")
            return None

    def _plot_predictions_with_split(self, test_df: pd.DataFrame,
                                     predictions_df: pd.DataFrame,
                                     history: object,
                                     save_dir: Optional[str] = None) -> None:
        """
        绘制基于训练测试集划分的预测结果图表

        参数:
        test_df: pd.DataFrame - 测试数据
        predictions_df: pd.DataFrame - 预测结果
        history: object - 训练历史（未使用）
        save_dir: Optional[str] - 保存目录
        """
        # # 设置图表样式
        # plt.style.use('classic')
        # plt.rcParams.update({
        #     'font.family': 'DejaVu Sans',
        #     'font.size': 10,
        #     'axes.labelsize': 12,
        #     'axes.titlesize': 12,
        #     'figure.dpi': 300
        # })

        # 创建单一图表
        plt.figure(figsize=(15, 6))

        # 绘制预测结果与实际值的对比
        plt.plot(test_df['timestamp'], test_df['load'],
                 label='Actual Load', color='blue', alpha=0.6)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='Stack2 Predicted Load', color='red', alpha=0.6)
        plt.title('Stack2 (CNN+XGBoost) Load Forecast with Train-Test Split')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'stack2_prediction_split.png'),
                        dpi=300, bbox_inches='tight')
            logging.info(f"Stack2 (CNN+XGBoost) prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()