import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging


class ProphetPredictor:
    """
    基于Facebook Prophet的时间序列预测器
    主要用于预测电力负载，支持温度作为额外的回归变量
    """

    def __init__(self, forecast_hours: int = 48, sample_interval: str = '30T'):
        """
        初始化预测器

        参数:
        forecast_hours: int - 预测时长（小时）
        sample_interval: str - 数据采样间隔，默认30分钟
        """
        self.forecast_hours = forecast_hours
        self.sample_interval = sample_interval
        self.forecast_points = int(forecast_hours * 60 / int(sample_interval[:-1]))
        self.best_params = None

        # 定义参数网格
        self.param_grid = {
            'changepoint_prior_scale': [0.01, 0.1],
            'seasonality_prior_scale': [0.1, 1.0],
            'seasonality_mode': ['multiplicative'],
            'changepoint_range': [0.9],
            'holidays_prior_scale': [0.1]
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理时间序列数据

        参数:
        df: DataFrame - 包含 'timestamp', 'temperature', 'load' 列的数据框

        返回:
        DataFrame - 处理后的Prophet格式数据框
        """
        try:
            print("Starting data preprocessing for Prophet...")

            # 准备Prophet格式数据
            prophet_df = df[['timestamp', 'load', 'temperature']].copy()
            prophet_df.columns = ['ds', 'y', 'temperature']

            # 确保数据按时间排序
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

            # 重采样数据
            prophet_df.set_index('ds', inplace=True)
            prophet_df = prophet_df.resample(self.sample_interval).mean().interpolate(method='time')
            prophet_df.reset_index(inplace=True)

            return prophet_df

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _grid_search(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """
        网格搜索最佳参数

        参数:
        train_df: DataFrame - 训练数据
        val_df: DataFrame - 验证数据

        返回:
        Dict - 最佳参数
        """
        all_params = [dict(zip(self.param_grid.keys(), v))
                      for v in itertools.product(*self.param_grid.values())]
        print(f"Evaluating {len(all_params)} parameter combinations")

        best_params = None
        best_score = float('inf')

        for params in all_params:
            try:
                print(f"\nTrying parameters: {params}")

                model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    seasonality_mode=params['seasonality_mode'],
                    changepoint_range=params['changepoint_range'],
                    holidays_prior_scale=params['holidays_prior_scale'],
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True
                )

                # 添加温度作为回归变量
                model.add_regressor('temperature')

                # 训练模型
                model.fit(train_df)

                # 验证
                forecast = model.predict(val_df)
                mse = mean_squared_error(val_df['y'], forecast['yhat'])
                mae = mean_absolute_error(val_df['y'], forecast['yhat'])

                if mse < best_score:
                    best_score = mse
                    best_params = params
                    print(f'New best parameters found: MSE={mse:.2f}, MAE={mae:.2f}')

            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue

        return best_params

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
            prophet_df = self._preprocess_data(df)

            # 划分训练集和验证集
            train_size = int(len(prophet_df) * 0.8)
            train_df = prophet_df[:train_size]
            val_df = prophet_df[train_size:]

            # 网格搜索最佳参数
            if not self.best_params:
                self.best_params = self._grid_search(train_df, val_df)

            if self.best_params:
                print("\nTraining final model with best parameters...")
                final_model = Prophet(
                    changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
                    seasonality_prior_scale=self.best_params['seasonality_prior_scale'],
                    seasonality_mode=self.best_params['seasonality_mode'],
                    changepoint_range=self.best_params['changepoint_range'],
                    holidays_prior_scale=self.best_params['holidays_prior_scale'],
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True
                )
                final_model.add_regressor('temperature')
                final_model.fit(prophet_df)

                # 创建未来时间序列
                future = final_model.make_future_dataframe(periods=self.forecast_points,
                                                           freq=self.sample_interval)

                # 使用最后一个已知温度值
                future['temperature'] = df['temperature'].iloc[-1]

                # 生成预测
                print("Generating predictions...")
                forecast = final_model.predict(future)
                predictions = forecast['yhat'].tail(self.forecast_points).values

                # 创建时间戳
                last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                future_timestamps = pd.date_range(
                    start=last_timestamp,
                    periods=self.forecast_points + 1,
                    freq=self.sample_interval
                )[1:]

                # 创建预测结果DataFrame
                predictions_df = pd.DataFrame({
                    'timestamp': future_timestamps,
                    'predicted_load': predictions
                })

                # 可视化结果
                self._plot_predictions(df, predictions_df, save_dir)

                # 输出预测统计信息
                self._print_prediction_summary(predictions)

                return predictions_df

            return None

        except Exception as e:
            logging.error(f"Error in prediction process: {str(e)}")
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

        # 绘制历史数据（最后48小时）
        historical_data = historical_df['load'].iloc[-48:]
        historical_timestamps = historical_df['timestamp'].iloc[-48:]
        plt.plot(historical_timestamps, historical_data,
                 label='Historical Data', color='blue', alpha=0.6)

        # 绘制预测数据
        plt.plot(predictions_df['timestamp'], predictions_df['predicted_load'],
                 label='Prophet Predicted Load', color='red', alpha=0.6)

        plt.title(f'{self.forecast_hours}-hour Prophet Load Forecast')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'prophet_prediction.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Prophet prediction plot saved in {save_dir}")
        else:
            plt.show()

        plt.close()

    def _print_prediction_summary(self, predictions: np.ndarray) -> None:
        """
        打印预测结果的统计信息

        参数:
        predictions: np.ndarray - 预测结果数组
        """
        print("\nProphet Prediction Summary:")
        print(f"Mean Load: {predictions.mean():.2f} MW")
        print(f"Standard Deviation: {predictions.std():.2f} MW")
        print(f"Maximum Load: {predictions.max():.2f} MW")
        print(f"Minimum Load: {predictions.min():.2f} MW")