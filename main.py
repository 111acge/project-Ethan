"""
main.py
主程序入口文件
"""

import os
import sys
import logging
import time
from contextlib import contextmanager
from functools import wraps

# 导入所需模块
from data_preprocessor import NSWDataPreprocessor
from visualizer import DataVisualizer
from model_comparator import ModelComparator, run_comparison

# Tee类用于同时输出到控制台和文件
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


# 定义输出重定向的上下文管理器
@contextmanager
def tee_output(filename):
    """
    将输出同时重定向到文件和控制台的上下文管理器
    """
    try:
        os.makedirs('logs', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as file:
            tee = Tee(sys.stdout, file)
            old_stdout = sys.stdout
            sys.stdout = tee
            yield  # 这里是关键，需要yield
            sys.stdout = old_stdout
    except Exception as e:
        print(f"Error in tee_output: {e}")
        sys.stdout = sys.__stdout__  # 确保发生错误时也能恢复stdout


# 定义装饰器
def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{'='*10}{func.__name__} is running {'='*20}")
        result = func(*args, **kwargs)
        print(f"{'='*10}{func.__name__} is done {'='*20}\n")
        return result
    return wrapper


@log_function
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


@log_function
def process_data():
    """
    数据处理步骤
    """
    try:
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


@log_function
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


@log_function
def compare_models(df):
    """
    比较不同模型的性能
    """
    print("Starting model comparison...")

    comparator = ModelComparator()
    comparator.compare_models(df)

    print("Model comparison completed")


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

    # 创建可视化
    create_visualizations(df, feature_names)

    # 执行模型比较
    compare_models(df)


if __name__ == "__main__":
    # 设置运行时间
    current_time = time.strftime("%Y%m%d_%H%M%S")

    # 使用重定向输出
    with tee_output(f'logs/output_{current_time}.txt'):
        main()