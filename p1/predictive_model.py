#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享单车需求预测模型
使用XGBoost算法预测未来时段的单车租赁需求和归还情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import timedelta
import warnings
import os

from utils import (setup_matplotlib, ensure_dir_exists, load_csv_file, 
                   calculate_distance, log_info, get_time_period_label)
from config import (DATA_CONFIG, RESULTS_CONFIG, PREPROCESSING_CONFIG,
                    TIME_ANALYSIS_CONFIG, SPATIAL_ANALYSIS_CONFIG,
                    PREDICTION_CONFIG, VISUALIZATION_CONFIG)

warnings.filterwarnings('ignore')

class DemandPredictor:
    """
    需求预测器类，用于预测共享单车的使用需求
    
    Attributes:
        config: 配置参数对象
        data: 原始数据
        model: 训练好的XGBoost模型
        time_series_data: 时间序列数据
        features: 用于预测的特征列表
    """
    
    def __init__(self):
        """初始化需求预测器"""
        # 合并所有配置
        self.config = {
            **DATA_CONFIG,
            **RESULTS_CONFIG,
            **PREPROCESSING_CONFIG,
            **TIME_ANALYSIS_CONFIG,
            **SPATIAL_ANALYSIS_CONFIG,
            **PREDICTION_CONFIG,
            **VISUALIZATION_CONFIG,
            'RANDOM_SEED': 42,
            'PLOT_SIZE': (15, 6),
            'PLOT_DPI': 300,
            'MODEL': {
                'XGB_PARAMS': PREDICTION_CONFIG['XGBOOST_PARAMS'],
                'N_SPLITS': PREDICTION_CONFIG['TIME_SERIES_SPLITS'],
                'TRAIN_TEST_SPLIT': 0.8,
                'LAG_FEATURES': [1, 2, 3, 6, 12, 24],
                'ROLLING_WINDOWS': [3, 6, 12],
                'FUTURE_HOURS': 24,
                'TOP_LOCATIONS': 10
            }
        }
        
        self.data = None
        self.model = None
        self.time_series_data = None
        self.features = None
        
        # 设置Matplotlib
        setup_matplotlib()
        
        # 创建必要的目录
        ensure_dir_exists(self.config['RESULTS_DIR'])
    
    def load_and_preprocess_data(self):
        """
        加载并预处理数据
        
        Returns:
            self: 当前实例，支持链式调用
        """
        log_info("1. 加载并预处理数据...")
        
        self.data = load_csv_file(self.config['PROCESSED_DATA_PATH'])
        if self.data is None:
            raise FileNotFoundError(f"无法找到数据文件: {self.config['PROCESSED_DATA_PATH']}")
        
        log_info(f"数据加载完成，共{len(self.data)}条记录")
        return self
    
    def feature_engineering(self):
        """
        特征工程：从时间和空间数据中提取特征
        
        Returns:
            self: 当前实例，支持链式调用
        """
        log_info("2. 进行特征工程...")
        
        df = self.data.copy()
        
        # 将时间列转换为datetime格式
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # 添加时间相关特征
        df['hour'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek  # 0=周一, 6=周日
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['day_of_month'] = df['start_time'].dt.day
        df['month'] = df['start_time'].dt.month
        
        # 添加时段标签
        df['time_period'] = df['hour'].apply(
            lambda hour: get_time_period_label(hour)
        )
        
        # 编码时段标签
        le = LabelEncoder()
        df['time_period_encoded'] = le.fit_transform(df['time_period'])
        
        # 添加骑行时长特征
        df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        
        # 添加距离特征（欧几里得距离）
        df['distance'] = calculate_distance(
            df['start_location_x'], df['start_location_y'],
            df['end_location_x'], df['end_location_y']
        )
        
        self.data = df
        log_info("特征工程完成")
        return self
    
    def create_time_series_features(self, target_col='demand_count', freq='1H'):
        """
        创建时间序列特征，按指定频率聚合数据
        
        Args:
            target_col: 目标列名称
            freq: 时间聚合频率
            
        Returns:
            self: 当前实例，支持链式调用
        """
        log_info("3. 创建时间序列特征...")
        
        # 按时间频率聚合数据
        time_series = self.data.set_index('start_time').resample(freq).size().reset_index(name=target_col)
        
        # 添加时间相关特征
        time_series['hour'] = time_series['start_time'].dt.hour
        time_series['day_of_week'] = time_series['start_time'].dt.dayofweek
        time_series['is_weekend'] = time_series['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        time_series['day_of_month'] = time_series['start_time'].dt.day
        time_series['month'] = time_series['start_time'].dt.month
        
        # 添加滞后特征
        for lag in self.config['MODEL']['LAG_FEATURES']:
            time_series[f'lag_{lag}'] = time_series[target_col].shift(lag)
        
        # 添加滚动统计特征
        for window in self.config['MODEL']['ROLLING_WINDOWS']:
            time_series[f'rolling_mean_{window}'] = time_series[target_col].rolling(window=window).mean()
            time_series[f'rolling_std_{window}'] = time_series[target_col].rolling(window=window).std()
        
        # 移除包含NaN值的行
        time_series = time_series.dropna()
        
        self.time_series_data = time_series
        log_info(f"时间序列特征创建完成，共{len(time_series)}条记录")
        return self
    
    def build_xgboost_model(self, X_train, y_train):
        """
        构建XGBoost模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练目标数据
            
        Returns:
            model: 训练好的XGBoost模型
        """
        # 从配置中获取模型参数
        xgb_params = self.config['MODEL']['XGB_PARAMS']
        
        model = xgb.XGBRegressor(**xgb_params)
        
        # 直接在整个训练集上训练模型
        model.fit(
            X_train, y_train,
            verbose=False
        )
        
        return model
    
    def train_model(self):
        """
        训练需求预测模型
        
        Returns:
            tuple: 测试集特征、测试集目标、测试集预测结果
        """
        log_info("4. 训练预测模型...")
        
        if self.time_series_data is None:
            raise ValueError("时间序列特征未创建，请先调用create_time_series_features")
        
        # 准备训练数据
        self.features = [
            'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month'
        ] + [f'lag_{lag}' for lag in self.config['MODEL']['LAG_FEATURES']] + \
        [f'rolling_mean_{window}' for window in self.config['MODEL']['ROLLING_WINDOWS']] + \
        [f'rolling_std_{window}' for window in self.config['MODEL']['ROLLING_WINDOWS']]
        
        X = self.time_series_data[self.features]
        y = self.time_series_data['demand_count']
        
        # 划分训练集和测试集（时间序列数据需要按时间顺序划分）
        split_point = int(len(X) * self.config['MODEL']['TRAIN_TEST_SPLIT'])
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        log_info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练模型
        log_info("4.2 训练XGBoost模型...")
        self.model = self.build_xgboost_model(X_train, y_train)
        
        # 评估模型
        log_info("4.3 评估模型性能...")
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        metrics = self._calculate_metrics(y_test, y_pred)
        
        log_info("模型评估结果：")
        for metric_name, value in metrics.items():
            log_info(f"  {metric_name}: {value}")
        
        return X_test, y_test, y_pred
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        计算模型评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            dict: 评估指标字典
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "均方误差 (MSE)": f"{mse:.2f}",
            "均方根误差 (RMSE)": f"{rmse:.2f}",
            "平均绝对误差 (MAE)": f"{mae:.2f}",
            "决定系数 (R²)": f"{r2:.4f}"
        }
    
    def predict_future_demand(self, future_hours=24):
        """
        预测未来指定小时数的需求
        
        Args:
            future_hours: 预测未来的小时数
            
        Returns:
            future_df: 包含预测结果的数据框
        """
        log_info(f"5. 预测未来{future_hours}小时的需求...")
        
        if self.model is None or self.time_series_data is None:
            raise ValueError("模型未训练，请先调用train_model")
        
        # 获取最后一个已知时间点
        last_time = self.time_series_data['start_time'].iloc[-1]
        
        # 创建未来时间点
        future_times = [last_time + timedelta(hours=i) for i in range(1, future_hours + 1)]
        
        # 创建未来预测数据框
        future_df = pd.DataFrame({'start_time': future_times})
        
        # 添加时间相关特征
        future_df['hour'] = future_df['start_time'].dt.hour
        future_df['day_of_week'] = future_df['start_time'].dt.dayofweek
        future_df['is_weekend'] = future_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        future_df['day_of_month'] = future_df['start_time'].dt.day
        future_df['month'] = future_df['start_time'].dt.month
        
        # 获取历史数据用于滞后特征
        historical_demand = self.time_series_data['demand_count'].values.tolist()
        
        # 预测每小时的需求
        future_predictions = []
        temp_history = historical_demand.copy()
        
        for i in range(future_hours):
            # 创建当前预测的特征
            current_features = {
                'hour': future_df.loc[i, 'hour'],
                'day_of_week': future_df.loc[i, 'day_of_week'],
                'is_weekend': future_df.loc[i, 'is_weekend'],
                'day_of_month': future_df.loc[i, 'day_of_month'],
                'month': future_df.loc[i, 'month'],
            }
            
            # 添加滞后特征
            for lag in self.config['MODEL']['LAG_FEATURES']:
                current_features[f'lag_{lag}'] = temp_history[-lag]
            
            # 添加滚动统计特征
            for window in self.config['MODEL']['ROLLING_WINDOWS']:
                current_features[f'rolling_mean_{window}'] = np.mean(temp_history[-window:])
                current_features[f'rolling_std_{window}'] = np.std(temp_history[-window:])
            
            # 转换为模型输入格式
            X_pred = pd.DataFrame([current_features])[self.features]
            
            # 进行预测
            prediction = self.model.predict(X_pred)[0]
            future_predictions.append(max(0, prediction))  # 确保预测值不为负
            
            # 更新历史数据，用于下一个预测
            temp_history.append(prediction)
        
        # 将预测结果添加到数据框
        future_df['predicted_demand'] = future_predictions
        future_df['predicted_demand'] = future_df['predicted_demand'].round(0)  # 取整
        
        log_info(f"未来{future_hours}小时需求预测完成")
        return future_df
    
    def visualize_predictions(self, y_test, y_pred, future_df):
        """
        可视化预测结果
        
        Args:
            y_test: 测试集实际值
            y_pred: 测试集预测值
            future_df: 未来需求预测数据框
            
        Returns:
            self: 当前实例，支持链式调用
        """
        log_info("6. 可视化预测结果...")
        
        # 1. 测试集预测 vs 实际值
        plt.figure(figsize=self.config['PLOT_SIZE'])
        plt.plot(range(len(y_test)), y_test.values, label='实际需求', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label='预测需求', color='red', alpha=0.7)
        plt.title('测试集：实际需求 vs 预测需求')
        plt.xlabel('时间点')
        plt.ylabel('需求数量')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.config["RESULTS_DIR"]}/prediction_test_set.png', 
                   dpi=self.config['PLOT_DPI'], bbox_inches='tight')
        plt.close()
        
        # 2. 未来需求预测
        plt.figure(figsize=self.config['PLOT_SIZE'])
        
        # 用户提供的颜色组
        second_color_group = [(58,13,96), (101,68,150), (151,141,190), (198,196,223), (235,236,242), 
                            (252,230,201), (253,193,117), (228,137,30), (181,90,6), (127,59,8)]
        # 转换为RGB格式
        second_color_group_rgb = [(r/255, g/255, b/255) for r, g, b in second_color_group]
        
        plt.plot(future_df['start_time'], future_df['predicted_demand'], 
                 color=second_color_group_rgb[0], marker='o', linestyle='-')
        plt.title(f'未来{len(future_df)}小时的需求预测')
        plt.xlabel('时间')
        plt.ylabel('预测需求数量')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig(f'{self.config["RESULTS_DIR"]}/future_demand_prediction.png', 
                   dpi=self.config['PLOT_DPI'], bbox_inches='tight')
        plt.close()
        
        # 3. 特征重要性
        plt.figure(figsize=(12, 8))
        
        # 使用用户提供的第一组颜色创建渐变
        first_color_group = [(1,121,135), (43,156,161), (123,178,176), (164,198,187), (219,233,227), 
                           (255,247,238), (242,218,190), (199,163,126), (125,103,75), (51,45,21)]
        # 转换为RGB格式
        first_color_group_rgb = [(r/255, g/255, b/255) for r, g, b in first_color_group]
        
        # 绘制特征重要性，使用渐变颜色
        ax = plt.gca()
        xgb.plot_importance(self.model, max_num_features=10, height=0.8, ax=ax, 
                          color=first_color_group_rgb[1])
        plt.title('XGBoost模型特征重要性')
        plt.savefig(f'{self.config["RESULTS_DIR"]}/feature_importance.png', 
                   dpi=self.config['PLOT_DPI'], bbox_inches='tight')
        plt.close()
        
        log_info("预测结果可视化完成")
        return self
    
    def spatial_demand_analysis(self, top_locations=10):
        """
        分析高需求区域的空间分布
        
        Args:
            top_locations: 分析的高需求区域数量
            
        Returns:
            top_locations_df: 高需求区域数据
        """
        log_info(f"7. 分析高需求区域的空间需求 (Top {top_locations})...")
        
        # 统计起点和终点的使用频率
        start_counts = self.data.groupby(['start_location_x', 'start_location_y']).size().reset_index(name='start_count')
        
        end_counts = self.data.groupby(['end_location_x', 'end_location_y']).size().reset_index(name='end_count')
        
        # 合并起点和终点数据
        locations = pd.merge(
            start_counts, end_counts,
            left_on=['start_location_x', 'start_location_y'],
            right_on=['end_location_x', 'end_location_y'],
            how='outer'
        )
        
        # 重命名列
        locations.rename(columns={'start_location_x': 'location_x', 
                                 'start_location_y': 'location_y'}, inplace=True)
        locations.drop(['end_location_x', 'end_location_y'], axis=1, inplace=True)
        
        # 填充缺失值
        locations['start_count'] = locations['start_count'].fillna(0)
        locations['end_count'] = locations['end_count'].fillna(0)
        
        # 计算总使用次数
        locations['total_count'] = locations['start_count'] + locations['end_count']
        
        # 获取使用频率最高的地点
        top_locations_df = locations.sort_values(
            by='total_count', ascending=False
        ).head(top_locations)
        
        # 可视化高需求区域
        plt.figure(figsize=(12, 10))
        
        # 使用更明显的渐变色组，从浅灰到深紫，再到亮黄和橙色
        color_group = [(200, 200, 200), (235, 236, 242), (198, 196, 223), (151, 141, 190), (101, 68, 150),
                      (252, 230, 201), (253, 193, 117), (228, 137, 30), (181, 90, 6), (127, 59, 8)]
        # 转换为RGB格式
        color_group_rgb = [(r/255, g/255, b/255) for r, g, b in color_group]
        
        # 使用颜色映射为所有起点创建渐变效果
        # 根据位置的使用频率分配颜色
        locations['normalized_count'] = (locations['total_count'] - locations['total_count'].min()) / (locations['total_count'].max() - locations['total_count'].min())
        
        # 创建自定义颜色映射
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [color_group_rgb[0], color_group_rgb[1], color_group_rgb[2]])
        
        plt.scatter(locations['location_x'], locations['location_y'], 
                   alpha=0.4, label='所有起点', c=locations['normalized_count'], cmap=cmap, s=locations['total_count']/10)
        plt.scatter(top_locations_df['location_x'], top_locations_df['location_y'], 
                   s=top_locations_df['total_count']/5, label='高需求区域', 
                   color=color_group_rgb[6], alpha=0.8, edgecolor='black', linewidth=1)
        plt.title(f'共享单车高需求区域（Top {top_locations}）')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.config["RESULTS_DIR"]}/high_demand_locations.png', 
                   dpi=self.config['PLOT_DPI'], bbox_inches='tight')
        plt.close()
        
        log_info(f"高需求区域分析完成，共识别{top_locations}个高需求区域")
        return top_locations_df
    
    def generate_report(self, y_test, y_pred, future_df):
        """
        生成预测结果报告
        
        Args:
            y_test: 测试集实际值
            y_pred: 测试集预测值
            future_df: 未来需求预测数据框
            
        Returns:
            self: 当前实例，支持链式调用
        """
        log_info("8. 生成预测结果报告...")
        
        # 计算评估指标
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # 生成报告内容
        report = f"""
共享单车需求预测模型报告
=========================

1. 模型概述
----------
使用算法：XGBoost
预测目标：未来时段单车租赁需求
预测单位：每小时

2. 模型评估指标
--------------
均方误差 (MSE): {metrics['均方误差 (MSE)']}
均方根误差 (RMSE): {metrics['均方根误差 (RMSE)']}
平均绝对误差 (MAE): {metrics['平均绝对误差 (MAE)']}
决定系数 (R²): {metrics['决定系数 (R²)']}

3. 未来24小时需求预测
--------------------
"""
        
        # 添加未来预测数据
        report += future_df[['start_time', 'predicted_demand']].to_string(index=False)
        
        # 添加预测统计信息
        report += f"""

4. 预测统计信息
--------------
预测时间范围: {future_df['start_time'].iloc[0].strftime('%Y-%m-%d %H:%M')} 至 {future_df['start_time'].iloc[-1].strftime('%Y-%m-%d %H:%M')}
平均预测需求: {future_df['predicted_demand'].mean():.2f}
最高预测需求: {future_df['predicted_demand'].max()}
最低预测需求: {future_df['predicted_demand'].min()}

5. 模型使用建议
--------------
- 该模型主要基于历史时间模式进行预测
- 建议结合天气、节假日等外部因素进一步优化
- 定期使用新数据重新训练模型以提高预测准确性
"""
        
        # 保存报告
        report_path = f'{self.config["RESULTS_DIR"]}/prediction_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        log_info(f"预测报告已保存至: {report_path}")
        return self
    
    def run(self):
        """
        执行完整的需求预测流程
        
        Returns:
            self: 当前实例
        """
        log_info("开始执行共享单车需求预测流程...")
        
        try:
            self.load_and_preprocess_data()
            self.feature_engineering()
            self.create_time_series_features()
            
            # 训练模型并获取结果
            X_test, y_test, y_pred = self.train_model()
            
            # 预测未来需求
            future_df = self.predict_future_demand(
                future_hours=self.config['MODEL']['FUTURE_HOURS']
            )
            
            # 可视化结果
            self.visualize_predictions(y_test, y_pred, future_df)
            
            # 空间需求分析
            self.spatial_demand_analysis(
                top_locations=self.config['MODEL']['TOP_LOCATIONS']
            )
            
            # 生成报告
            self.generate_report(y_test, y_pred, future_df)
            
            log_info("共享单车需求预测流程执行完成！")
            log_info(f"所有结果已保存至: {self.config['RESULTS_DIR']}")
            return self
            
        except Exception as e:
            log_info(f"需求预测过程中发生错误：{str(e)}")
            raise
    
    def get_time_period(self, hour: int) -> str:
        """
        根据小时获取时间段标签
        
        Args:
            hour: 小时数
            
        Returns:
            时间段标签：'早高峰'、'晚高峰'、'非高峰'
        """
        morning_start, morning_end = self.config['TIME_PERIODS']['MORNING_PEAK']
        evening_start, evening_end = self.config['TIME_PERIODS']['EVENING_PEAK']
        
        if morning_start <= hour < morning_end:
            return '早高峰'
        elif evening_start <= hour < evening_end:
            return '晚高峰'
        else:
            return '非高峰'
    
    def analyze_demand_by_time_period(self):
        """
        按时间段分析起点订单数
        
        Returns:
            dict: 不同时间段的起点订单统计
        """
        log_info("按时间段分析起点订单数...")
        
        # 确保数据已加载
        if self.data is None:
            self.load_and_preprocess_data()
        
        # 添加时间段列
        self.data['time_period'] = self.data['start_hour'].apply(self.get_time_period)
        
        # 按时间段和起点位置统计订单数
        demand_by_period = {}
        
        # 遍历所有时间段
        for period in ['早高峰', '晚高峰', '非高峰']:
            # 筛选该时间段的数据
            period_data = self.data[self.data['time_period'] == period]
            
            # 统计每个起点的订单数
            start_counts = period_data.groupby(['start_location_x', 'start_location_y']).size().reset_index(name='demand_count')
            
            # 排序
            start_counts = start_counts.sort_values(by='demand_count', ascending=False)
            
            demand_by_period[period] = start_counts
        
        log_info("按时间段分析完成")
        return demand_by_period
    
    def analyze_return_by_time_period(self):
        """
        按时间段分析终点归还数
        
        Returns:
            dict: 不同时间段的终点归还统计
        """
        log_info("按时间段分析终点归还数...")
        
        # 确保数据已加载
        if self.data is None:
            self.load_and_preprocess_data()
        
        # 添加时间段列
        self.data['time_period'] = self.data['end_hour'].apply(self.get_time_period)
        
        # 按时间段和终点位置统计归还数
        return_by_period = {}
        
        # 遍历所有时间段
        for period in ['早高峰', '晚高峰', '非高峰']:
            # 筛选该时间段的数据
            period_data = self.data[self.data['time_period'] == period]
            
            # 统计每个终点的归还数
            end_counts = period_data.groupby(['end_location_x', 'end_location_y']).size().reset_index(name='return_count')
            
            # 排序
            end_counts = end_counts.sort_values(by='return_count', ascending=False)
            
            return_by_period[period] = end_counts
        
        log_info("按时间段归还分析完成")
        return return_by_period
    
    def get_top_stations(self, n=20):
        """
        获取主要起始站点和归还站点
        
        Args:
            n: 要获取的站点数量
            
        Returns:
            tuple: (top_start_stations, top_return_stations)
        """
        log_info(f"获取前{n}个主要起始站点和归还站点...")
        
        # 确保数据已加载
        if self.data is None:
            self.load_and_preprocess_data()
        
        # 统计所有起始站点的订单数
        all_start_counts = self.data.groupby(['start_location_x', 'start_location_y']).size().reset_index(name='demand_count')
        top_start_stations = all_start_counts.sort_values(by='demand_count', ascending=False).head(n)
        
        # 统计所有归还站点的归还数
        all_return_counts = self.data.groupby(['end_location_x', 'end_location_y']).size().reset_index(name='return_count')
        top_return_stations = all_return_counts.sort_values(by='return_count', ascending=False).head(n)
        
        log_info(f"获取主要站点完成")
        return top_start_stations, top_return_stations
    
    def get_top_stations_by_period(self, n=20):
        """
        按时间段获取主要起始站点和归还站点
        
        Args:
            n: 要获取的站点数量
            
        Returns:
            tuple: (demand_by_period, return_by_period)
        """
        log_info(f"按时间段获取前{n}个主要起始站点和归还站点...")
        
        # 获取按时间段的需求和归还统计
        demand_by_period = self.analyze_demand_by_time_period()
        return_by_period = self.analyze_return_by_time_period()
        
        # 获取每个时间段的前n个站点
        top_demand_by_period = {}
        top_return_by_period = {}
        
        for period in ['早高峰', '晚高峰', '非高峰']:
            top_demand_by_period[period] = demand_by_period[period].head(n)
            top_return_by_period[period] = return_by_period[period].head(n)
        
        log_info(f"按时间段获取主要站点完成")
        return top_demand_by_period, top_return_by_period
    
    def export_morning_peak_stations(self, n=20):
        """
        导出早高峰主要需求站点表格
        
        Args:
            n: 要导出的站点数量
            
        Returns:
            DataFrame: 早高峰主要需求站点数据
        """
        log_info(f"导出早高峰前{n}个主要需求站点表格...")
        
        # 获取按时间段的需求统计
        demand_by_period = self.analyze_demand_by_time_period()
        
        # 获取早高峰前n个站点
        morning_peak_stations = demand_by_period['早高峰'].head(n)
        
        # 重命名列以便更好的可读性
        morning_peak_stations = morning_peak_stations.rename(columns={
            'start_location_x': '经度',
            'start_location_y': '纬度',
            'demand_count': '需求量'
        })
        
        # 保存为CSV文件
        output_path = os.path.join(self.config['RESULTS_DIR'], 'morning_peak_demand_stations.csv')
        morning_peak_stations.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        log_info(f"早高峰主要需求站点表格已导出至: {output_path}")
        return morning_peak_stations
    
    def predict_station_demand(self, stations, future_hours=24, is_return=False):
        """
        预测指定站点在未来不同时段的需求量
        
        Args:
            stations: 要预测的站点列表，包含经度和纬度
            future_hours: 预测未来的小时数
            is_return: 是否预测归还量（True表示预测归还量，False表示预测需求量）
            
        Returns:
            DataFrame: 站点预测结果
        """
        log_info(f"预测指定站点未来{future_hours}小时的{'归还量' if is_return else '需求量'}...")
        
        # 确保数据已加载
        if self.data is None:
            self.load_and_preprocess_data()
        
        # 确保模型已训练
        if self.model is None:
            self.feature_engineering()
            self.create_time_series_features()
            self.train_model()
        
        # 获取最后一个已知时间点
        last_time = self.time_series_data['start_time'].iloc[-1]
        
        # 创建未来时间点
        future_times = [last_time + timedelta(hours=i) for i in range(1, future_hours + 1)]
        
        # 创建未来预测数据框
        future_df = pd.DataFrame({'start_time': future_times})
        
        # 添加时间相关特征
        future_df['hour'] = future_df['start_time'].dt.hour
        future_df['day_of_week'] = future_df['start_time'].dt.dayofweek
        future_df['is_weekend'] = future_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        future_df['day_of_month'] = future_df['start_time'].dt.day
        future_df['month'] = future_df['start_time'].dt.month
        future_df['time_period'] = future_df['hour'].apply(self.get_time_period)
        
        # 获取历史数据用于滞后特征
        historical_demand = self.time_series_data['demand_count'].values.tolist()
        
        # 预测每小时的总需求
        future_predictions = []
        temp_history = historical_demand.copy()
        
        for i in range(future_hours):
            # 创建当前预测的特征
            current_features = {
                'hour': future_df.loc[i, 'hour'],
                'day_of_week': future_df.loc[i, 'day_of_week'],
                'is_weekend': future_df.loc[i, 'is_weekend'],
                'day_of_month': future_df.loc[i, 'day_of_month'],
                'month': future_df.loc[i, 'month'],
            }
            
            # 添加滞后特征
            for lag in self.config['MODEL']['LAG_FEATURES']:
                current_features[f'lag_{lag}'] = temp_history[-lag]
            
            # 添加滚动统计特征
            for window in self.config['MODEL']['ROLLING_WINDOWS']:
                current_features[f'rolling_mean_{window}'] = np.mean(temp_history[-window:])
                current_features[f'rolling_std_{window}'] = np.std(temp_history[-window:])
            
            # 转换为模型输入格式
            X_pred = pd.DataFrame([current_features])[self.features]
            
            # 进行预测
            prediction = self.model.predict(X_pred)[0]
            future_predictions.append(max(0, prediction))  # 确保预测值不为负
            
            # 更新历史数据，用于下一个预测
            temp_history.append(prediction)
        
        # 将预测结果添加到数据框
        future_df['total_predicted'] = future_predictions
        future_df['total_predicted'] = future_df['total_predicted'].round(0)  # 取整
        
        # 计算每个站点的历史需求比例
        if is_return:
            # 统计归还站点的历史归还数
            station_counts = self.data.groupby(['end_location_x', 'end_location_y']).size().reset_index(name='count')
        else:
            # 统计起始站点的历史需求量
            station_counts = self.data.groupby(['start_location_x', 'start_location_y']).size().reset_index(name='count')
        
        # 只保留要预测的站点
        station_mapping = {}
        total_historical = 0
        
        for _, station in stations.iterrows():
            lon_col = 'end_location_x' if is_return else 'start_location_x'
            lat_col = 'end_location_y' if is_return else 'start_location_y'
            
            # 处理不同的列名情况
            if lon_col in station:
                lon = station[lon_col]
                lat = station[lat_col]
            elif '经度' in station:
                lon = station['经度']
                lat = station['纬度']
            else:
                # 如果是新的列名，尝试获取第一个包含坐标的列
                lon = station.iloc[0]
                lat = station.iloc[1]
            
            # 查找该站点的历史数据
            station_data = station_counts[
                (station_counts[lon_col] == lon) & 
                (station_counts[lat_col] == lat)
            ]
            
            if not station_data.empty:
                count = station_data.iloc[0]['count']
            else:
                count = 1  # 如果没有历史数据，给一个最小的计数
            
            station_mapping[(lon, lat)] = count
            total_historical += count
        
        # 为每个站点分配预测需求（基于历史需求比例）
        results = []
        
        for (lon, lat), count in station_mapping.items():
            # 计算该站点的历史需求比例
            if total_historical > 0:
                ratio = count / total_historical
            else:
                ratio = 1.0 / len(station_mapping)
            
            for _, row in future_df.iterrows():
                # 根据历史比例分配预测需求
                predicted_value = row['total_predicted'] * ratio
                
                results.append({
                    'start_time': row['start_time'],
                    'time_period': row['time_period'],
                    '经度': lon,
                    '纬度': lat,
                    'predicted_value': round(predicted_value)
                })
        
        # 创建结果数据框
        prediction_results = pd.DataFrame(results)
        
        # 根据类型重命名列
        if is_return:
            prediction_results = prediction_results.rename(columns={'predicted_value': 'predicted_return'})
        else:
            prediction_results = prediction_results.rename(columns={'predicted_value': 'predicted_demand'})
        
        log_info(f"站点{'归还量' if is_return else '需求量'}预测完成")
        return prediction_results
    
    def export_prediction_results(self, n=20, future_hours=24):
        """
        导出不同时段不同站点的需求量和归还量预测结果
        
        Args:
            n: 主要站点数量
            future_hours: 预测未来的小时数
            
        Returns:
            tuple: (demand_predictions, return_predictions)
        """
        log_info(f"导出不同时段不同站点的预测结果...")
        
        # 获取主要起始站点和归还站点
        top_start_stations, top_return_stations = self.get_top_stations(n=n)
        
        # 预测起始站点的需求量
        demand_predictions = self.predict_station_demand(top_start_stations, future_hours=future_hours, is_return=False)
        
        # 保存需求预测结果
        demand_output_path = os.path.join(self.config['RESULTS_DIR'], 'station_demand_predictions.csv')
        demand_predictions.to_csv(demand_output_path, index=False, encoding='utf-8-sig')
        
        # 预测归还站点的归还量
        return_predictions = self.predict_station_demand(top_return_stations, future_hours=future_hours, is_return=True)
        
        # 保存归还预测结果
        return_output_path = os.path.join(self.config['RESULTS_DIR'], 'station_return_predictions.csv')
        return_predictions.to_csv(return_output_path, index=False, encoding='utf-8-sig')
        
        log_info(f"预测结果已导出至: {demand_output_path} 和 {return_output_path}")
        return demand_predictions, return_predictions
    
    def run_assignment(self, n=20, future_hours=24):
        """
        执行完整的作业要求：分析不同时段的使用频率分布，识别高需求区域，预测未来需求
        
        Args:
            n: 主要站点数量
            future_hours: 预测未来的小时数
        """
        log_info("开始执行作业要求...")
        
        try:
            # 1. 加载数据
            self.load_and_preprocess_data()
            
            # 2. 分析不同时段的使用频率分布
            log_info("1. 分析不同时段的使用频率分布...")
            demand_by_period = self.analyze_demand_by_time_period()
            return_by_period = self.analyze_return_by_time_period()
            
            # 3. 识别高需求区域（主要起始站点和归还站点）
            log_info(f"2. 识别高需求区域...")
            top_start_stations, top_return_stations = self.get_top_stations(n=n)
            
            # 4. 导出早高峰主要需求站点表格
            log_info(f"3. 导出早高峰主要需求站点表格...")
            morning_peak_stations = self.export_morning_peak_stations(n=n)
            
            # 5. 训练模型（如果需要）
            log_info("4. 训练预测模型...")
            self.feature_engineering()
            self.create_time_series_features()
            self.train_model()
            
            # 6. 预测不同时段不同站点的需求量和归还量
            log_info(f"5. 预测未来{future_hours}小时的需求量和归还量...")
            demand_predictions, return_predictions = self.export_prediction_results(n=n, future_hours=future_hours)
            
            log_info("作业要求执行完成！")
            log_info(f"所有结果已保存至: {self.config['RESULTS_DIR']}")
            
            return {
                'demand_by_period': demand_by_period,
                'return_by_period': return_by_period,
                'top_start_stations': top_start_stations,
                'top_return_stations': top_return_stations,
                'morning_peak_stations': morning_peak_stations,
                'demand_predictions': demand_predictions,
                'return_predictions': return_predictions
            }
            
        except Exception as e:
            log_info(f"作业执行过程中发生错误：{str(e)}")
            raise

def main():
    """主函数，执行需求预测流程"""
    try:
        predictor = DemandPredictor()
        predictor.run()
    except Exception as e:
        log_info(f"程序执行失败：{str(e)}")
        raise

if __name__ == "__main__":
    main()