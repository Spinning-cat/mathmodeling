import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.neighbors import KDTree

def build_station_timeseries():
    # 设置文件路径
    data_path = "E:\\QI002\\Documents\\trae_projects\\math model HW\\data\\processed_data.csv"
    top30_path = "E:\\QI002\\Documents\\trae_projects\\math model HW\\results\\dbscan_top30_clusters.csv"
    cluster_stations_path = "E:\\QI002\\Documents\\trae_projects\\math model HW\\results\\dbscan_cluster_stations_new.csv"
    output_path = "E:\\QI002\\Documents\\trae_projects\\math model HW\\results\\top30_count.csv"
    
    print("1. 读取数据...")
    # 读取前30个主要簇的信息
    top30_clusters = pd.read_csv(top30_path)
    main_clusters = top30_clusters['cluster'].tolist()
    print(f"   ✓ 读取前30个主要簇信息，簇ID: {main_clusters}")
    
    # 读取所有簇的站点信息
    all_cluster_stations = pd.read_csv(cluster_stations_path)
    print(f"   ✓ 读取所有簇的站点信息，共 {len(all_cluster_stations)} 个站点")
    
    # 只保留前30个簇的站点
    main_stations = all_cluster_stations[all_cluster_stations['dbscan_cluster'].isin(main_clusters)]
    print(f"   ✓ 筛选出前30个簇的站点，共 {len(main_stations)} 个站点")
    
    # 读取原始骑行数据
    df = pd.read_csv(data_path)
    print(f"   ✓ 读取原始骑行数据，共 {len(df)} 条记录")
    
    # 转换时间字段为datetime类型
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    print("2. 使用KDTree进行空间映射...")
    # 准备KDTree数据
    station_coords = main_stations[['location_x', 'location_y']].values
    station_clusters = main_stations['dbscan_cluster'].values
    
    # 创建KDTree
    kdtree = KDTree(station_coords)
    
    # 将订单的起始位置映射到最近的簇
    start_coords = df[['start_location_x', 'start_location_y']].values
    _, start_indices = kdtree.query(start_coords, k=1)
    df['start_cluster'] = station_clusters[start_indices.flatten()]
    
    # 将订单的结束位置映射到最近的簇
    end_coords = df[['end_location_x', 'end_location_y']].values
    _, end_indices = kdtree.query(end_coords, k=1)
    df['end_cluster'] = station_clusters[end_indices.flatten()]
    
    print("3. 构造时间片并统计...")
    # 计算时间片编号（1-48）
    df['start_time_slot'] = (df['start_time'].dt.hour * 2 + df['start_time'].dt.minute // 30) + 1
    df['end_time_slot'] = (df['end_time'].dt.hour * 2 + df['end_time'].dt.minute // 30) + 1
    
    # 统计租赁量（按起始簇和起始时间片）
    rent_stats = df.groupby(['start_cluster', 'start_time_slot']).size().reset_index(name='rent_count')
    rent_stats = rent_stats.rename(columns={'start_cluster': 'cluster_id', 'start_time_slot': 'time_slot'})
    
    # 统计归还量（按结束簇和结束时间片）
    return_stats = df.groupby(['end_cluster', 'end_time_slot']).size().reset_index(name='return_count')
    return_stats = return_stats.rename(columns={'end_cluster': 'cluster_id', 'end_time_slot': 'time_slot'})
    
    # 合并租赁量和归还量
    merged_stats = pd.merge(rent_stats, return_stats, on=['cluster_id', 'time_slot'], how='outer')
    merged_stats = merged_stats.fillna(0)
    merged_stats[['rent_count', 'return_count']] = merged_stats[['rent_count', 'return_count']].astype(int)
    
    print("4. 生成全量时间片覆盖...")
    # 创建所有簇和时间片的组合
    all_clusters = pd.Series(main_clusters, name='cluster_id')
    all_time_slots = pd.Series(range(1, 49), name='time_slot')
    all_combinations = all_clusters.to_frame().merge(all_time_slots, how='cross')
    
    # 合并以确保所有组合都存在
    final_stats = pd.merge(all_combinations, merged_stats, on=['cluster_id', 'time_slot'], how='left')
    final_stats = final_stats.fillna(0)
    final_stats[['rent_count', 'return_count']] = final_stats[['rent_count', 'return_count']].astype(int)
    
    print("5. 保存结果...")
    # 保存到CSV文件
    final_stats[['cluster_id', 'time_slot', 'rent_count', 'return_count']].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ 站点级时间序列已保存至 {output_path}")
    print(f"   ✓ 共 {len(final_stats)} 条记录")
    
    # 打印前几行数据查看结果
    print("\n结果示例：")
    print(final_stats.head())
    
    # 打印一些统计信息
    print("\n统计信息：")
    print(f"   - 主要簇数量：{len(main_clusters)}")
    print(f"   - 时间片数量：48 (30分钟/片)")
    print(f"   - 总记录数：{len(final_stats)} (30簇 × 48时间片)")
    print(f"   - 平均每簇每时间片租赁量：{final_stats['rent_count'].mean():.2f}")
    print(f"   - 平均每簇每时间片归还量：{final_stats['return_count'].mean():.2f}")

if __name__ == "__main__":
    build_station_timeseries()
