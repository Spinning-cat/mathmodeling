import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree

# 定义结果目录
RESULTS_DIR = "E:/QI002/Documents/trae_projects/math model HW/results"

# 1. 加载数据
print("正在加载数据...")

# 加载原始骑行数据
data_path = "E:/QI002/Documents/trae_projects/math model HW/data/processed_data.csv"
df = pd.read_csv(data_path)

# 加载前30个主要簇的数据
top30_path = "E:/QI002/Documents/trae_projects/math model HW/results/dbscan_top30_clusters.csv"
top30_clusters = pd.read_csv(top30_path)
main_clusters = top30_clusters['cluster'].tolist()

print(f"共加载 {len(df)} 条骑行记录")
print(f"选择的前30个主要簇：{main_clusters}")

# 2. 预处理时间数据
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# 3. 加载簇站点映射关系（将每个订单的起始/结束位置映射到对应的簇）
print("正在加载簇站点映射关系...")
cluster_stations_path = "E:/QI002/Documents/trae_projects/math model HW/results/dbscan_cluster_stations_new.csv"
cluster_stations = pd.read_csv(cluster_stations_path)

# 只保留主要簇的站点
cluster_stations = cluster_stations[cluster_stations['dbscan_cluster'].isin(main_clusters)]

print(f"共找到 {len(cluster_stations)} 个属于主要簇的站点")

# 4. 将订单映射到对应的簇
print("正在将订单映射到对应的簇...")

# 使用KDTree加速最近邻查找
station_coords = cluster_stations[['location_x', 'location_y']].values
tree = KDTree(station_coords)

# 添加起始簇
start_coords = df[['start_location_x', 'start_location_y']].values
distances, indices = tree.query(start_coords)
df['start_cluster'] = cluster_stations.iloc[indices.flatten()]['dbscan_cluster'].values

# 添加结束簇
end_coords = df[['end_location_x', 'end_location_y']].values
distances, indices = tree.query(end_coords)
df['end_cluster'] = cluster_stations.iloc[indices.flatten()]['dbscan_cluster'].values

# 只保留属于主要簇的订单
df = df[(df['start_cluster'].isin(main_clusters)) & (df['end_cluster'].isin(main_clusters))]

print(f"过滤后共保留 {len(df)} 条属于主要簇的订单")

# 5. 构造站点级时间序列
print("正在构造站点级时间序列...")

# 定义时间片长度（30分钟）
time_slot_minutes = 30

# 创建所有时间片
all_time_slots = []
for hour in range(24):
    for minute in [0, 30]:
        time_slot = hour * 2 + (minute // 30) + 1  # 时间片编号从1开始
        all_time_slots.append(time_slot)

# 计算每个订单的时间片
df['start_time_slot'] = (df['start_time'].dt.hour * 2) + (df['start_time'].dt.minute // time_slot_minutes) + 1
df['end_time_slot'] = (df['end_time'].dt.hour * 2) + (df['end_time'].dt.minute // time_slot_minutes) + 1

# 6. 统计租赁量和归还量
print("正在统计租赁量和归还量...")

# 统计租赁量（按起始簇和起始时间片）
rent_stats = df.groupby(['start_cluster', 'start_time_slot']).size().reset_index(name='rent_count')
rent_stats.rename(columns={'start_cluster': 'cluster_id', 'start_time_slot': 'time_slot'}, inplace=True)

# 统计归还量（按结束簇和结束时间片）
return_stats = df.groupby(['end_cluster', 'end_time_slot']).size().reset_index(name='return_count')
return_stats.rename(columns={'end_cluster': 'cluster_id', 'end_time_slot': 'time_slot'}, inplace=True)

# 合并租赁量和归还量
merged_stats = pd.merge(rent_stats, return_stats, on=['cluster_id', 'time_slot'], how='outer')
merged_stats.fillna(0, inplace=True)

# 确保所有簇和时间片都有记录
all_combinations = pd.DataFrame(
    [(cluster, time_slot) for cluster in main_clusters for time_slot in all_time_slots],
    columns=['cluster_id', 'time_slot']
)

final_stats = pd.merge(all_combinations, merged_stats, on=['cluster_id', 'time_slot'], how='left')
final_stats.fillna(0, inplace=True)

# 将计数转换为整数
final_stats['rent_count'] = final_stats['rent_count'].astype(int)
final_stats['return_count'] = final_stats['return_count'].astype(int)

# 7. 保存结果
print("正在保存结果...")

output_path = os.path.join(RESULTS_DIR, 'top30_count.csv')
final_stats.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"结果已保存至 {output_path}")
print(f"共生成 {len(final_stats)} 条记录")
print(f"每条记录包含：cluster_id, time_slot, rent_count, return_count")

# 验证结果
print("\n验证结果：")
print(f"- 簇数量：{len(final_stats['cluster_id'].unique())}")
print(f"- 时间片数量：{len(final_stats['time_slot'].unique())}")
print(f"- 总租赁量：{final_stats['rent_count'].sum()}")
print(f"- 总归还量：{final_stats['return_count'].sum()}")

print("\n站点级时间序列构造完成！")
