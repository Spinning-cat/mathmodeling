#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件：集中管理所有项目参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据文件路径
DATA_CONFIG = {
    'RAW_DATA_PATH': PROJECT_ROOT / '题C-附件-mobike_shanghai_dataset.csv',
    'PROCESSED_DATA_PATH': PROJECT_ROOT / 'data' / 'processed_data.csv',
}

# 结果文件路径
RESULTS_CONFIG = {
    'RESULTS_DIR': PROJECT_ROOT / 'results',
    'TIME_ANALYSIS_REPORT': PROJECT_ROOT / 'results' / 'time_analysis_report.txt',
    'SPATIAL_ANALYSIS_REPORT': PROJECT_ROOT / 'results' / 'spatial_analysis_report.txt',
    'PREDICTION_REPORT': PROJECT_ROOT / 'results' / 'prediction_report.txt',
    'FINAL_REPORT': PROJECT_ROOT / 'results' / 'final_report.md',
}

# 数据预处理参数
PREPROCESSING_CONFIG = {
    # 上海地区经纬度范围
    'LONGITUDE_RANGE': (120.84, 122.10),
    'LATITUDE_RANGE': (30.65, 31.86),
    # 骑行时长过滤条件（分钟）
    'DURATION_MIN': 1,
    'DURATION_MAX': 120,
    # 时段划分
    'TIME_PERIODS': {
        'MORNING_PEAK': (7, 9),
        'EVENING_PEAK': (17, 19),
    },
}

# 时间分析参数
TIME_ANALYSIS_CONFIG = {
    'CHART_DPI': 300,
    'FIGSIZE': (12, 6),
    'HOURLY_PATTERN_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'hourly_pattern.png',
    'WEEKDAY_WEEKEND_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'weekday_weekend_pattern.png',
    'DAILY_TREND_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'daily_trend.png',
    'DURATION_DISTRIBUTION_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'duration_distribution.png',
}

# 空间分析参数
SPATIAL_ANALYSIS_CONFIG = {
    'CHART_DPI': 300,
    'FIGSIZE': (12, 10),
    'KMEANS_N_CLUSTERS': 10,
    'KMEANS_RANDOM_STATE': 42,
    'CLUSTERING_VISUALIZATION_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'clustering_visualization.png',
    'DEMAND_HEATMAP_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'demand_heatmap.png',
    'SPATIAL_DISTRIBUTION_MORNING_PEAK_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'spatial_distribution_morning_peak.png',
    'SPATIAL_DISTRIBUTION_EVENING_PEAK_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'spatial_distribution_evening_peak.png',
    'SPATIAL_DISTRIBUTION_OFF_PEAK_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'spatial_distribution_off_peak.png',
}

# 预测模型参数
PREDICTION_CONFIG = {
    # 时间序列参数
    'AGGREGATION_FREQ': '1H',  # 按小时聚合
    'FUTURE_HOURS_TO_PREDICT': 24,  # 预测未来24小时
    # XGBoost模型参数
    'XGBOOST_PARAMS': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
    },
    'TIME_SERIES_SPLITS': 5,  # 时间序列交叉验证折数
    # 可视化参数
    'CHART_DPI': 300,
    'FIGSIZE': (15, 6),
    'PREDICTION_TEST_SET_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'prediction_test_set.png',
    'FUTURE_DEMAND_PREDICTION_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'future_demand_prediction.png',
    'FEATURE_IMPORTANCE_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'feature_importance.png',
    'HIGH_DEMAND_LOCATIONS_FILE': RESULTS_CONFIG['RESULTS_DIR'] / 'high_demand_locations.png',
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'CHART_DPI': 300,
    'FONT_FAMILY': ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
    'FONT_SIZE': 12,
}
