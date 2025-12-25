#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块：从p1结果文件中加载所需数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_CONFIG


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        """初始化数据加载器"""
        self.all_clusters = None
        self.top30_clusters = None
        self.top30_count = None
        self.station_demand_gap = None
        self.processed_data = None
    
    def load_all_clusters(self) -> pd.DataFrame:
        """
        加载所有站点（聚类）的统计信息
        
        Returns:
            包含所有站点信息的DataFrame
        """
        print("正在加载所有站点统计信息...")
        file_path = DATA_CONFIG['ALL_CLUSTERS_STATS']
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(str(file_path))
        # 过滤掉噪声点（cluster == -1）
        df = df[df['cluster'] != -1].copy()
        
        print(f"  ✓ 加载完成，共 {len(df)} 个有效站点")
        self.all_clusters = df
        return df
    
    def load_top30_clusters(self) -> pd.DataFrame:
        """
        加载30个主要站点信息
        
        Returns:
            包含30个主要站点信息的DataFrame
        """
        print("正在加载30个主要站点信息...")
        file_path = DATA_CONFIG['TOP30_CLUSTERS']
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(str(file_path))
        print(f"  ✓ 加载完成，共 {len(df)} 个主要站点")
        self.top30_clusters = df
        return df
    
    def load_top30_count(self) -> pd.DataFrame:
        """
        加载30个主要站点的24小时借还统计
        
        Returns:
            包含时间序列借还统计的DataFrame
        """
        print("正在加载30个主要站点时间序列统计...")
        file_path = DATA_CONFIG['TOP30_COUNT']
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(str(file_path))
        print(f"  ✓ 加载完成，共 {len(df)} 条记录")
        self.top30_count = df
        return df
    
    def load_station_demand_gap(self) -> pd.DataFrame:
        """
        加载30个主要站点早高峰需求预测
        
        Returns:
            包含早高峰需求预测的DataFrame
        """
        print("正在加载早高峰需求预测...")
        file_path = DATA_CONFIG['STATION_DEMAND_GAP']
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(str(file_path))
        print(f"  ✓ 加载完成，共 {len(df)} 条记录")
        self.station_demand_gap = df
        return df
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        加载清洗后的原始数据
        
        Returns:
            清洗后的原始骑行数据
        """
        print("正在加载原始数据...")
        file_path = DATA_CONFIG['PROCESSED_DATA']
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        df = pd.read_csv(str(file_path))
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        print(f"  ✓ 加载完成，共 {len(df)} 条记录")
        self.processed_data = df
        return df
    
    def load_all(self) -> dict:
        """
        加载所有需要的数据
        
        Returns:
            包含所有数据的字典
        """
        print("\n=== 开始加载数据 ===")
        
        all_clusters = self.load_all_clusters()
        top30_clusters = self.load_top30_clusters()
        top30_count = self.load_top30_count()
        station_demand_gap = self.load_station_demand_gap()
        processed_data = self.load_processed_data()
        
        print("\n=== 数据加载完成 ===\n")
        
        return {
            'all_clusters': all_clusters,
            'top30_clusters': top30_clusters,
            'top30_count': top30_count,
            'station_demand_gap': station_demand_gap,
            'processed_data': processed_data,
        }


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的欧几里得距离
    
    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度
    
    Returns:
        距离（单位：km，近似值）
    """
    # 使用简化的距离计算（适用于小范围）
    # 1度纬度 ≈ 111 km, 1度经度 ≈ 111*cos(lat) km
    lat_avg = (lat1 + lat2) / 2
    lat_diff = (lat2 - lat1) * 111
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians(lat_avg))
    
    return np.sqrt(lat_diff**2 + lon_diff**2)

