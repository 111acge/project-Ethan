import os
import sys
import logging
import time
from contextlib import contextmanager
from functools import wraps

# 以下是文件
from data_preprocessor import NSWDataPreprocessor
from visualizer import DataVisualizer
from time_series_predictor import TimeSeriesPredictor
from prophet_predictor import ProphetPredictor
from MLP_predictor import MLPPredictor
from BNN_predictor import BNNPredictor
from CNN_predictor import CNNPredictor
from LSTM_predictor import LSTMPredictor
from XGBoost_ensemble import XGBoostEnsemblePredictor
from RandomForests_ensemble import RandomForestsPredictor


# 输出Tee
class Tee:
    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file
        self.encoding = stdout.encoding

    def write(self, data):
        try:
            self.stdout.write(data)
            self.file.write(data)
            # 确保立即刷新缓冲区
            self.stdout.flush()
            self.file.flush()
        except Exception as e:
            print(f"Tee write error: {e}")

    def flush(self):
        try:
            self.stdout.flush()
            self.file.flush()
        except Exception as e:
            print(f"Tee flush error: {e}")

# 定义装饰器
def log_function_name(func):
    @wraps(func)  # 保留原函数的元信息（如名称、文档字符串等）
    def wrapper(*args, **kwargs):
        print(f"{'='*10}{func.__name__} is running {'='*20}")  # 打印函数名称
        result = func(*args, **kwargs)  # 执行原函数并保存返回值
        print(f"{'='*10}{func.__name__} is done {'='*20}\n")  # 打印函数完成信息
        return result  # 返回原函数的结果
    return wrapper

@log_function_name
def check_data_files():
    """
    检查必要的数据文件是否存在
    """
    data_files = {
        'temperature': 'data/NSW/temperature_nsw.csv',
        'demand': 'data/NSW/totaldemand_nsw.csv'
    }

    for name, path in data_files.items():
        if not os.path.exists(path):
            logging.error(f"Error: {name} data file not found: {path}")
            return False
    return True

@log_function_name
def process_data():
    """
    数据处理步骤
    """
    try:
        # 创建预处理器实例
        preprocessor = NSWDataPreprocessor(window_size=1440)  # 24小时

        # 准备数据
        X_train, X_test, y_train, y_test, feature_names, df = preprocessor.prepare_data(
            'data/NSW/temperature_nsw.csv',
            'data/NSW/totaldemand_nsw.csv'
        )

        if X_train is not None:
            print("\nData preprocessing completed:")
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Number of features: {len(feature_names)}")
            print("\nFeature list:")
            for i, feature in enumerate(feature_names):
                print(f"{i + 1}. {feature}")

            return X_train, X_test, y_train, y_test, feature_names, preprocessor, df

        return None

    except Exception as e:
        logging.error(f"Error during data processing: {str(e)}")
        return None

@log_function_name
def create_visualizations(df, feature_names, save_dir='data-preprocessor'):
    """
    创建数据可视化
    """
    visualizer = DataVisualizer(save_dir)

    print("Starting data visualization...")

    # 绘制每日负载模式
    visualizer.plot_daily_pattern(df)

    # 绘制温度与负载关系
    visualizer.plot_temp_load_relationship(df)

    print("Data visualization completed successfully")

@log_function_name
def time_series(df):
    """
    示例使用方法
    """
    # 创建预测器实例
    predictor = TimeSeriesPredictor(
        lookback_hours=48,
        forecast_hours=48,
        sample_interval='30T',
        n_neighbors=5
    )

    # 读取示例数据
    try:
        save_dir = 'model-time_series'

        # 执行预测
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            predictions.to_csv(os.path.join(save_dir, 'time_series_predictions.csv'), index=False)
            print("Predictions saved to CSV file")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")



@log_function_name
def prophet_series(df):
    """
    使用Prophet模型进行预测
    """
    # 创建Prophet预测器实例
    predictor = ProphetPredictor(
        forecast_hours=48,
        sample_interval='30T'
    )

    try:
        save_dir = 'model-prophet'
        os.makedirs(save_dir, exist_ok=True)

        # 执行预测
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            predictions.to_csv(os.path.join(save_dir, 'prophet_predictions.csv'), index=False)
            print("Prophet predictions saved to CSV file")

    except Exception as e:
        logging.error(f"Error in prophet execution: {str(e)}")

@log_function_name
def mlp_prediction(df):
    """
    使用MLP模型进行预测
    """
    # 创建预测器实例
    predictor = MLPPredictor(
        lookback_hours=48,  # 使用过去48小时的数据
        forecast_hours=48,  # 预测未来48小时
        sample_interval='30T',  # 30分钟间隔
    )

    try:
        # 创建保存目录
        save_dir = 'modle-mlp'
        os.makedirs(save_dir, exist_ok=True)

        # 执行预测
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            # 保存预测结果到CSV
            predictions.to_csv(os.path.join(save_dir, 'mlp_predictions.csv'), index=False)
            print("MLP predictions saved to CSV file")
            return predictions

    except Exception as e:
        logging.error(f"Error in MLP prediction: {str(e)}")


@log_function_name
def bnn_prediction(df):
    """
    使用BNN模型进行预测
    """
    # 创建预测器实例
    predictor = BNNPredictor(
        lookback_hours=48,  # 使用过去48小时的数据
        forecast_hours=48,  # 预测未来48小时
        sample_interval='30T'  # 30分钟间隔
    )

    try:
        # 创建保存目录
        save_dir = 'model-bnn'
        os.makedirs(save_dir, exist_ok=True)

        # 执行预测
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            # 保存预测结果到CSV
            predictions.to_csv(os.path.join(save_dir, 'bnn_predictions.csv'), index=False)
            print("BNN predictions saved to CSV file")
            return predictions

    except Exception as e:
        logging.error(f"Error in BNN prediction: {str(e)}")

@log_function_name
def cnn_prediction(df):
    """
    CNN模型预测
    """
    predictor = CNNPredictor(
        lookback_hours=48,
        forecast_hours=48,
        sample_interval='30T'
    )

    try:
        save_dir = 'model-cnn'
        os.makedirs(save_dir, exist_ok=True)

        print("Starting CNN prediction process...")
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            predictions.to_csv(os.path.join(save_dir, 'cnn_predictions.csv'), index=False)
            print("CNN predictions saved to CSV file")
            return predictions

    except Exception as e:
        logging.error(f"Error in CNN prediction: {str(e)}")
        print(f"CNN prediction failed with error: {str(e)}")

@log_function_name
def lstm_prediction(df):
    """
    使用LSTM模型进行预测
    """
    # 创建预测器实例
    predictor = LSTMPredictor(
        lookback_hours=48,  # 使用过去48小时的数据
        forecast_hours=48,  # 预测未来48小时
        sample_interval='30T'  # 30分钟间隔
    )

    try:
        # 创建保存目录
        save_dir = 'model-lstm'
        os.makedirs(save_dir, exist_ok=True)

        # 执行预测
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            # 保存预测结果到CSV
            predictions.to_csv(os.path.join(save_dir, 'lstm_predictions.csv'), index=False)
            print("LSTM predictions saved to CSV file")
            return predictions

    except Exception as e:
        logging.error(f"Error in LSTM prediction: {str(e)}")

@log_function_name
def xgboost_prediction(df):
    """
    Using XGBoost Ensemble model for prediction
    """
    # Initialize predictor
    predictor = XGBoostEnsemblePredictor(
        lookback_hours=48,
        forecast_hours=48,
        sample_interval='30T'
    )

    try:
        # Create save directory
        save_dir = 'model-xgboost'
        os.makedirs(save_dir, exist_ok=True)

        # Execute prediction
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            # Save predictions to CSV
            predictions.to_csv(os.path.join(save_dir, 'xgboost_predictions.csv'), index=False)
            print("XGBoost ensemble predictions saved to CSV file")
            return predictions

    except Exception as e:
        logging.error(f"Error in XGBoost prediction: {str(e)}")

@log_function_name
def random_forest_prediction(df):
    """
    随机森林预测
    """
    # 创建预测器实例
    predictor = RandomForestsPredictor(
        lookback_hours=48,
        forecast_hours=48,
        sample_interval='30T'
    )

    try:
        save_dir = 'model-random_forest'
        os.makedirs(save_dir, exist_ok=True)

        # 执行预测
        predictions = predictor.predict(df, save_dir=save_dir)

        if predictions is not None:
            predictions.to_csv(os.path.join(save_dir, 'random_forest_predictions.csv'), index=False)
            print("Random Forest predictions saved to CSV file")

    except Exception as e:
        logging.error(f"Error in Random Forest prediction: {str(e)}")

def main():
    """
    主函数，用于协调整个程序的执行流程
    """
    # 检查数据文件
    if not check_data_files():
        return


    # 数据处理
    results = process_data()
    if results is None:
        logging.error("Data processing failed")
        return

    X_train, X_test, y_train, y_test, feature_names, preprocessor, df = results


    # # 创建可视化
    # create_visualizations(df, feature_names)
    #
    # # 模型-时间序列-负载
    # time_series(df)
    #
    # # 模型-prophet
    # prophet_series(df)
    #
    # # 模型-MLP预测
    # mlp_prediction(df)
    #
    # # 模型-BNN预测
    # bnn_prediction(df)
    #
    # # 模型-CNN预测
    # cnn_prediction(df)
    #
    # # 模型-LSTM预测
    # lstm_prediction(df)
    #
    # # 模型-XGBoost集成预测
    # xgboost_prediction(df)

    # 随机森林预测
    random_forest_prediction(df)


if __name__ == "__main__":

    @contextmanager
    def tee_output(filename):
        try:
            os.makedirs('logs', exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as file:
                tee = Tee(sys.stdout, file)
                old_stdout = sys.stdout
                sys.stdout = tee
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
        except Exception as e:
            print(f"Error in tee_output: {e}")


    # 使用方式
    current_time = time.strftime("%Y%m%d_%H%M%S")
    with tee_output(f'logs/output_{current_time}.txt'):
        main()