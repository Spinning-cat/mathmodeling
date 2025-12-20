import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置工作目录
os.chdir('e:\\QI002\\Documents\\trae_projects\\math model HW')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 从文件中读取数据
demand_df = pd.read_csv('results/station_demand_predictions.csv')
return_df = pd.read_csv('results/station_return_predictions.csv')

# 提取起始站点（去重，只保留唯一的经纬度对）
demand_stations = demand_df[['经度', '纬度']].drop_duplicates().reset_index(drop=True)
# 提取归还站点（去重，只保留唯一的经纬度对）
return_stations = return_df[['经度', '纬度']].drop_duplicates().reset_index(drop=True)

# 创建图表
plt.figure(figsize=(12, 10))

# 绘制起始站点（使用红色，标记为'D'）
plt.scatter(
    demand_stations['经度'], 
    demand_stations['纬度'], 
    color='red', 
    s=100, 
    marker='D', 
    label='主要起始站点'
)

# 绘制归还站点（使用蓝色，标记为'O'）
plt.scatter(
    return_stations['经度'], 
    return_stations['纬度'], 
    color='blue', 
    s=100, 
    marker='o', 
    label='主要归还站点'
)

# 给起始站点添加标注
for i, (lon, lat) in enumerate(zip(demand_stations['经度'], demand_stations['纬度'])):
    plt.annotate(
        f'{i+1}',  # 标注内容：站点编号
        (lon, lat),  # 标注位置
        fontsize=10, 
        fontweight='bold',
        color='red',
        xytext=(5, 5),  # 文本偏移量
        textcoords='offset points'
    )

# 给归还站点添加标注
for i, (lon, lat) in enumerate(zip(return_stations['经度'], return_stations['纬度'])):
    plt.annotate(
        f'{i+1}',  # 标注内容：站点编号
        (lon, lat),  # 标注位置
        fontsize=10, 
        fontweight='bold',
        color='blue',
        xytext=(5, 5),  # 文本偏移量
        textcoords='offset points'
    )

# 设置图表标题和坐标轴标签
plt.title('主要起始站点和归还站点分布', fontsize=16, fontweight='bold')
plt.xlabel('经度', fontsize=14)
plt.ylabel('纬度', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 调整网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 调整坐标轴范围，确保所有点都在视图内
total_lon = pd.concat([demand_stations['经度'], return_stations['经度']])
total_lat = pd.concat([demand_stations['纬度'], return_stations['纬度']])

# 添加一些边距
lon_min, lon_max = total_lon.min() - 0.001, total_lon.max() + 0.001
lat_min, lat_max = total_lat.min() - 0.001, total_lat.max() + 0.001

plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)

# 保存图表
save_path = 'results/station_distribution.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"站点分布图已保存至: {save_path}")
print(f"\n站点统计:")
print(f"起始站点数量: {len(demand_stations)}")
print(f"归还站点数量: {len(return_stations)}")

# 检查是否有重叠站点
merged = demand_stations.merge(return_stations, on=['经度', '纬度'], how='inner')
if len(merged) > 0:
    print(f"重叠站点数量: {len(merged)}")
else:
    print("没有重叠站点")
