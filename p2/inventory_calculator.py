#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始库存计算模块：根据历史数据计算所有站点的初始库存
"""

import pandas as pd
import numpy as np
from data_loader import calculate_distance
from sklearn.neighbors import KDTree


class InventoryCalculator:
    """初始库存计算器"""
    
    def __init__(self, config):
        """
        初始化库存计算器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
    
    def calculate_initial_inventory(self, all_clusters: pd.DataFrame, 
                                   top30_count: pd.DataFrame,
                                   processed_data: pd.DataFrame,
                                   top30_clusters: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有站点的初始库存，其中前30个为主要站点

        方法：基于历史平均借还量计算初始库存，与p2.py保持一致
        
        Args:
            all_clusters: 所有1810个聚类站点信息
            top30_count: 30个主要站点的时间序列统计
            processed_data: 原始数据
            top30_clusters: 30个主要站点信息（用于标记高需求站点）

        Returns:
            包含所有站点初始库存的DataFrame
        """
        print("\n=== 开始计算初始库存 ===")
        
        scale_factor = self.config['INVENTORY_SCALE_FACTOR']
        
        # 1. 计算30个主要站点的初始库存
        print("1. 计算30个主要站点的初始库存...")
        
        # 计算每个主要站点的平均借还量
        top30_avg_rent = top30_count.groupby('cluster_id')['rent_count'].mean()
        top30_avg_return = top30_count.groupby('cluster_id')['return_count'].mean()
        top30_avg_total = top30_avg_rent + top30_avg_return
        
        # 初始库存 = 平均借还量 * 缩放因子
        top30_inventory = (top30_avg_total * scale_factor).round().astype(int)
        
        # 创建主要站点库存字典
        top30_inventory_dict = top30_inventory.to_dict()
        
        # 2. 为所有站点计算初始库存
        print("2. 计算所有站点的初始库存...")
        
        # 创建主要站点ID集合
        top30_cluster_ids = set(top30_clusters['cluster'].tolist())
        
        # 初始化所有站点的库存列表
        inventory_list = []
        
        for idx, cluster in all_clusters.iterrows():
            cluster_id = cluster['cluster']
            
            # 判断是否为主要站点
            is_main_station = cluster_id in top30_cluster_ids
            
            # 根据是否为主要站点计算初始库存
            if is_main_station:
                initial_inventory = top30_inventory_dict.get(cluster_id, 0)
            else:
                # 非主要站点使用简化计算：基于订单总数的一定比例
                order_count = cluster['sum_total_count']
                initial_inventory = max(10, int(order_count * 0.5 * scale_factor))  # 至少10辆
                
            # 生成station_id
            station_id = f"S{str(idx+1).zfill(4)}"  # 使用4位数字以支持1810个站点
            
            # 确定站点类型和容量
            order_count = cluster['sum_total_count']
            if order_count >= 100:
                station_type = "large"
                capacity = max(200, initial_inventory * 1.2)  # 确保容量足够大
            elif order_count >= 50:
                station_type = "medium"
                capacity = max(150, initial_inventory * 1.2)
            else:
                station_type = "small"
                capacity = max(100, initial_inventory * 1.2)
            
            # 计算高峰需求和总需求（与p2.py一致）
            if is_main_station:
                # 主要站点使用更精确的计算
                base_demand = int(order_count * 0.8)  # 基础需求
            else:
                # 非主要站点使用简化计算
                base_demand = max(5, int(order_count * 0.3))  # 至少5辆
                
            peak_demand = base_demand  # 高峰需求
            total_demand = base_demand  # 总需求
            
            inventory_list.append({
                'station_id': station_id,
                'cluster_id': cluster_id,
                'center_x': cluster['center_x'],
                'center_y': cluster['center_y'],
                'sum_total_count': cluster['sum_total_count'],
                'station_count': cluster['station_count'],
                'station_type': station_type,
                'capacity': capacity,
                'peak_demand': peak_demand,
                'total_demand': total_demand,
                'initial_inventory': initial_inventory,
                'is_main_station': is_main_station,
            })
        
        inventory_df = pd.DataFrame(inventory_list)
        
        print(f"\n=== 初始库存计算完成 ===")
        print(f"总站点数: {len(inventory_df)}")
        print(f"主要站点数: {inventory_df['is_main_station'].sum()}")
        print(f"平均初始库存: {inventory_df['initial_inventory'].mean():.1f}")
        print(f"总初始库存: {inventory_df['initial_inventory'].sum()}")
        print(f"最大初始库存: {inventory_df['initial_inventory'].max()}")
        print(f"最小初始库存: {inventory_df['initial_inventory'].min()}\n")
        
        return inventory_df

