import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os
from datetime import datetime


class DataVisualizer:
    """
    用于数据可视化的类，提供多种图表绘制功能
    """

    def __init__(self, save_dir: str):
        """
        初始化可视化工具

        参数:
        save_dir: str - 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._setup_plot_style()

    def _setup_plot_style(self):
        """
        设置图表样式
        """
        # 使用内置的风格而不是seaborn
        plt.style.use('classic')
        sns.set_theme(style="whitegrid")  # 使用seaborn的主题设置
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def _save_plot(self, name: str):
        """
        保存图表

        参数:
        name: str - 图表名称
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}.png"
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")

    def plot_time_series(self, df: pd.DataFrame, columns: List[str], title: str):
        """
        绘制时间序列数据

        参数:
        df: pd.DataFrame - 包含时间序列数据的DataFrame
        columns: List[str] - 需要绘制的列名列表
        title: str - 图表标题
        """
        plt.figure(figsize=(15, 7))
        for column in columns:
            plt.plot(df['timestamp'], df[column], label=column, alpha=0.7)

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        self._save_plot(f"time_series_{columns[0]}")

    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str]):
        """
        绘制特征相关性热力图

        参数:
        df: pd.DataFrame - 包含特征的DataFrame
        features: List[str] - 需要分析相关性的特征列表
        """
        plt.figure(figsize=(12, 10))
        correlation = df[features].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        self._save_plot('correlation_heatmap')

    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str]):
        """
        绘制特征分布图

        参数:
        df: pd.DataFrame - 包含特征的DataFrame
        features: List[str] - 需要分析分布的特征列表
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // 2

        plt.figure(figsize=(15, 5 * n_rows))
        for i, feature in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=feature, kde=True)
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Count')

        plt.tight_layout()
        self._save_plot('feature_distributions')

    def plot_daily_pattern(self, df: pd.DataFrame):
        """
        绘制每日负载模式图

        参数:
        df: pd.DataFrame - 包含负载数据的DataFrame
        """
        daily_df = df.copy()
        daily_df['hour'] = daily_df['timestamp'].dt.hour
        daily_pattern = daily_df.groupby('hour')['load'].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(12, 6))
        plt.plot(daily_pattern['hour'], daily_pattern['mean'], 'b-', label='Mean Load')
        plt.fill_between(
            daily_pattern['hour'],
            daily_pattern['mean'] - daily_pattern['std'],
            daily_pattern['mean'] + daily_pattern['std'],
            alpha=0.2,
            label='±1 std'
        )

        plt.title('Average Daily Load Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 24, 2))
        self._save_plot('daily_pattern')

    def plot_temp_load_relationship(self, df: pd.DataFrame):
        """
        绘制温度与负载关系图

        参数:
        df: pd.DataFrame - 包含温度和负载数据的DataFrame
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='temperature', y='load', alpha=0.5)

        # 添加趋势线
        z = np.polyfit(df['temperature'], df['load'], 2)
        p = np.poly1d(z)
        temp_range = np.linspace(df['temperature'].min(), df['temperature'].max(), 100)
        plt.plot(temp_range, p(temp_range), 'r--', label='Trend Line')

        plt.title('Temperature vs Load Relationship')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        self._save_plot('temp_load_relationship')

    def plot_weekly_pattern(self, df: pd.DataFrame):
        """
        绘制每周负载模式图

        参数:
        df: pd.DataFrame - 包含负载数据的DataFrame
        """
        weekly_df = df.copy()
        weekly_df['day_of_week'] = weekly_df['timestamp'].dt.day_name()
        weekly_pattern = weekly_df.groupby('day_of_week')['load'].agg(['mean', 'std']).reset_index()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern['day_of_week'] = pd.Categorical(weekly_pattern['day_of_week'], categories=days_order, ordered=True)
        weekly_pattern = weekly_pattern.sort_values('day_of_week')

        plt.figure(figsize=(12, 6))
        plt.plot(days_order, weekly_pattern['mean'], 'b-', marker='o', label='Mean Load')
        plt.fill_between(
            range(7),
            weekly_pattern['mean'] - weekly_pattern['std'],
            weekly_pattern['mean'] + weekly_pattern['std'],
            alpha=0.2,
            label='±1 std'
        )

        plt.title('Average Weekly Load Pattern')
        plt.xlabel('Day of Week')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(7), days_order, rotation=45)
        self._save_plot('weekly_pattern')

    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray):
        """
        绘制特征重要性图

        参数:
        feature_names: List[str] - 特征名称列表
        importance_scores: np.ndarray - 特征重要性分数
        """
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(sorted_idx.shape[0]) + .5

        plt.figure(figsize=(10, len(feature_names) / 2))
        plt.barh(pos, importance_scores[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Feature Importance Score')
        plt.title('Feature Importance')
        self._save_plot('feature_importance')