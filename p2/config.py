#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：共享单车调度优化模型配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
P21_DIR = Path(__file__).parent

# 数据文件路径（从p1目录读取）
P1_RESULTS_DIR = PROJECT_ROOT / 'p1' / 'results'
P1_DATA_DIR = PROJECT_ROOT / 'p1' / 'data'

DATA_CONFIG = {
    # p1的输出文件
    'ALL_CLUSTERS_STATS': P1_RESULTS_DIR / 'dbscan_cluster_stats_new.csv',  # 所有站点统计
    'TOP30_CLUSTERS': P1_RESULTS_DIR / 'dbscan_top30_clusters.csv',  # 30个主要站点
    'TOP30_COUNT': P1_RESULTS_DIR / 'top30_count.csv',  # 30个站点24小时借还统计
    'STATION_DEMAND_GAP': P1_RESULTS_DIR / 'station_demand_gap.csv',  # 早高峰需求预测
    'PROCESSED_DATA': P1_DATA_DIR / 'processed_data.csv',  # 清洗后的原始数据
}

# 结果输出目录
RESULTS_CONFIG = {
    'RESULTS_DIR': P21_DIR / 'results',
    'SCHEDULING_PLAN': P21_DIR / 'results' / 'optimal_scheduling_plan.csv',
    'STATION_INVENTORY': P21_DIR / 'results' / 'station_initial_inventory.csv',
    'SCHEDULING_REPORT': P21_DIR / 'results' / 'scheduling_optimization_report.md',
    'VISUALIZATION': P21_DIR / 'results' / 'scheduling_visualization.png',
}

# 调度优化参数
SCHEDULING_CONFIG = {
    # 初始库存计算参数
    'INITIAL_INVENTORY_METHOD': 'historical_avg',  # 'historical_avg', 'demand_based', 'fixed_ratio'
    'INVENTORY_SCALE_FACTOR': 1.5,  # 进一步增加初始库存缩放因子
    
    # 库存约束参数
    'MIN_INVENTORY_THRESHOLD_RATIO': 0.2,  # 最低库存阈值比例（相对于初始库存）
    'MAX_SUPPLY_RATIO': 0.8,  # 最大可调度量比例（初始库存 - 最低库存）
    
    # 调度基础参数
    'K_NEIGHBORS': 10,  # 邻居站点数量
    'SOLVER_TIME_LIMIT': 300,  # 求解器超时时间（秒）
    'MAX_DISTANCE': 3.5,  # 增加最大调度距离，增加调度路径选择
    
    # 调度成本与约束参数
    'VEHICLE_CAPACITY': 25,  # 单车载量（辆）
    'DISTANCE_COST_PER_KM': 1.0,  # 每公里调度成本（元）
    'FIXED_COST_PER_TRIP': 8.0,  # 每次调度的固定成本（元）
    'PENALTY_COEFFICIENT': 300.0,  # 未满足需求惩罚系数
    'DEMAND_SATISFACTION_RATIO': 0.90,  # 需求满足率
    'MAX_VEHICLES': 200,  # 全局最大调度车辆数
    'MAX_BIKES_PER_TRIP': 30,  # 单路径最大调度量
    'MAX_OUTFLOW_PER_STATION': 100,  # 单点最大调出量
    'SCHEDULE_WINDOW': 6,  # 调度时间窗口（小时）
    
    # 需求满足参数
    'TIME_PERIOD': 'morning_peak',  # 'morning_peak', 'evening_peak', 'full_day'
    
    # 优化算法参数
    'SOLVER': 'pulp',  # 'pulp', 'scipy', 'gurobi'
    'TIMEOUT': 300,  # 求解器超时时间（秒）
    'SOLVER_GAP': 0.10,  # 最优解差距容忍度
    'SOLVER_THREADS': 4,  # 并行计算线程数
}

# 可视化参数
VISUALIZATION_CONFIG = {
    'DPI': 300,
    'FIGSIZE': (14, 10),
    'MAP_BOUNDS': {
        'lon_min': 121.2,
        'lon_max': 121.6,
        'lat_min': 31.1,
        'lat_max': 31.5,
    },
}

