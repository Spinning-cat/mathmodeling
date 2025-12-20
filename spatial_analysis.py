import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# 保持当前工作目录不变
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置非交互式后端
plt.switch_backend('Agg')

# 1. 数据导入
def load_processed_data(file_path):
    """导入处理后的数据"""
    df = pd.read_csv(file_path)
    print(f"数据导入完成，共{len(df)}条记录")
    return df

# 2. 位置数据处理
def process_location_data(df):
    """处理位置数据，计算每个位置的使用频率"""
    # 统计每个起点的使用次数
    start_locations = df.groupby(['start_location_x', 'start_location_y']).size().reset_index(name='start_count')
    # 统计每个终点的使用次数
    end_locations = df.groupby(['end_location_x', 'end_location_y']).size().reset_index(name='end_count')
    
    # 合并起点和终点数据
    locations = pd.merge(start_locations, end_locations, 
                        left_on=['start_location_x', 'start_location_y'], 
                        right_on=['end_location_x', 'end_location_y'], 
                        how='outer')
    
    # 重命名列
    locations.rename(columns={'start_location_x': 'location_x', 'start_location_y': 'location_y'}, inplace=True)
    locations.drop(['end_location_x', 'end_location_y'], axis=1, inplace=True)
    
    # 填充缺失值
    locations['start_count'].fillna(0, inplace=True)
    locations['end_count'].fillna(0, inplace=True)
    
    # 计算总使用次数
    locations['total_count'] = locations['start_count'] + locations['end_count']
    
    # 移除所有可能的NaN值
    locations.dropna(inplace=True)
    
    return locations

# 3. K-means聚类分析
def perform_kmeans_clustering(locations, n_clusters=10):
    """使用K-means算法进行聚类分析"""
    # 提取坐标和使用次数作为特征
    X = locations[['location_x', 'location_y', 'total_count']]
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    locations['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 计算每个聚类的中心
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=['center_x', 'center_y', 'center_total_count'])
    cluster_centers_df['cluster'] = range(n_clusters)
    
    # 计算每个簇的半径（簇内点到中心的最大欧氏距离）
    cluster_radii = []
    for i in range(n_clusters):
        # 获取当前簇的所有点
        cluster_points = X_scaled[locations['cluster'] == i]
        # 计算到聚类中心的欧氏距离
        distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
        # 最大距离作为簇半径
        max_distance = np.max(distances)
        # 转换回原始数据空间的距离（需要考虑标准化的缩放因子）
        # 获取每个特征的标准差
        std_devs = scaler.scale_
        # 计算原始空间中的半径（近似值）
        original_radius = max_distance * np.mean(std_devs[:2])  # 只考虑经纬度的标准差
        cluster_radii.append(original_radius)
    
    cluster_centers_df['cluster_radius'] = cluster_radii
    
    return locations, cluster_centers_df

# 3.1 DBSCAN聚类分析
def perform_dbscan_clustering(locations, eps=0.5, min_samples=5):
    """使用DBSCAN算法进行聚类分析"""
    # 提取坐标和使用次数作为特征
    X = locations[['location_x', 'location_y', 'total_count']]
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    locations['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
    
    # DBSCAN没有聚类中心的概念，但可以计算每个簇的统计信息
    # 创建类似cluster_centers_df的结构，包含每个簇的统计信息
    cluster_stats = locations.groupby('dbscan_cluster').agg({
        'location_x': ['mean'],
        'location_y': ['mean'],
        'total_count': ['mean', 'sum', 'count']
    }).reset_index()
    
    # 重命名列
    cluster_stats.columns = ['cluster', 'center_x', 'center_y', 'mean_total_count', 'sum_total_count', 'station_count']
    
    return locations, cluster_stats

# 4. 识别高需求区域
def identify_high_demand_areas(locations, cluster_centers_df):
    """识别高需求区域"""
    # 按总使用次数排序
    high_demand_areas = locations.sort_values(by='total_count', ascending=False).head(20)
    
    # 按聚类中心的使用次数排序，识别高需求聚类
    high_demand_clusters = cluster_centers_df.sort_values(by='center_total_count', ascending=False)
    
    return high_demand_areas, high_demand_clusters

# 5. 可视化聚类结果
def visualize_clustering_results(locations, cluster_centers_df):
    """可视化聚类结果"""
    plt.figure(figsize=(12, 10))
    
    # 用户提供的颜色组
    first_color_group = [(1,121,135), (43,156,161), (123,178,176), (164,198,187), (219,233,227), 
                       (255,247,238), (242,218,190), (199,163,126), (125,103,75), (51,45,21)]
    # 转换为RGB格式
    first_color_group_rgb = [(r/255, g/255, b/255) for r, g, b in first_color_group]
    
    # 创建自定义渐变色图
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(first_color_group_rgb)
    
    # 绘制所有位置点，颜色表示聚类
    scatter = plt.scatter(locations['location_x'], locations['location_y'], 
                        c=locations['cluster'], cmap=custom_cmap, 
                        s=locations['total_count']/10, alpha=0.6)
    
    # 绘制聚类中心
    plt.scatter(cluster_centers_df['center_x'], cluster_centers_df['center_y'], 
                c='black', marker='x', s=200, label='聚类中心')
    
    plt.title('共享单车使用位置聚类分析')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.colorbar(scatter, label='聚类编号')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/clustering_visualization.png', dpi=300)
    plt.close()
    
    # 绘制高需求区域热力图
    plt.figure(figsize=(12, 10))
    
    # 使用更明显的渐变色组，从深紫到亮黄再到深红
    second_color_group = [(128, 0, 128), (186, 85, 211), (238, 130, 238), (255, 255, 224), (255, 255, 0), 
                        (255, 215, 0), (255, 165, 0), (255, 140, 0), (255, 69, 0), (255, 0, 0)]
    # 转换为RGB格式
    second_color_group_rgb = [(r/255, g/255, b/255) for r, g, b in second_color_group]
    custom_heatmap_cmap = ListedColormap(second_color_group_rgb)
    
    sns.kdeplot(
        x=locations['location_x'], 
        y=locations['location_y'], 
        weights=locations['total_count'],
        fill=True,
        cmap=custom_heatmap_cmap,
        alpha=0.8
    )
    
    plt.title('共享单车使用热力图')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    
    plt.tight_layout()
    plt.savefig('../results/demand_heatmap.png', dpi=300)
    plt.close()

# 6. 分析不同时段的空间分布
def analyze_time_period_spatial(df, time_period):
    """分析不同时段的空间分布"""
    # 根据时段过滤数据
    if time_period == 'morning_peak':
        period_data = df[(df['start_hour'] >= 7) & (df['start_hour'] < 9)]
        title = '早高峰时段（7:00-9:00）' 
    elif time_period == 'evening_peak':
        period_data = df[(df['start_hour'] >= 17) & (df['start_hour'] < 19)]
        title = '晚高峰时段（17:00-19:00）'
    else:
        period_data = df[(df['start_hour'] < 7) | (df['start_hour'] >= 9) & (df['start_hour'] < 17) | (df['start_hour'] >= 19)]
        title = '平峰时段'
    
    # 统计每个起点的使用次数
    start_locations = period_data.groupby(['start_location_x', 'start_location_y']).size().reset_index(name='count')
    
    # 可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(start_locations['start_location_x'], start_locations['start_location_y'], 
                        s=start_locations['count']/5, alpha=0.6, c='blue')
    
    plt.title(f'{title} 共享单车使用空间分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    
    # 添加颜色条
    plt.colorbar(scatter, label='使用次数')
    
    plt.tight_layout()
    plt.savefig(f'../results/spatial_distribution_{time_period}.png', dpi=300)
    plt.close()
    
    return start_locations

# 7. 生成空间分析报告
def generate_spatial_report(locations, high_demand_areas, high_demand_clusters, dbscan_cluster_stats=None):
    """生成空间分析报告"""
    with open('../results/spatial_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("共享单车空间分布分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 位置使用统计\n")
        f.write("-" * 30 + "\n")
        f.write(f"总位置数：{len(locations)}\n")
        f.write(f"平均每个位置使用次数：{locations['total_count'].mean():.2f}\n")
        f.write(f"最大使用次数：{locations['total_count'].max()}\n")
        f.write(f"最小使用次数：{locations['total_count'].min()}\n\n")
        
        f.write("2. 高需求区域（前20个）\n")
        f.write("-" * 30 + "\n")
        for i, (_, area) in enumerate(high_demand_areas.iterrows(), 1):
            f.write(f"{i}. 位置：({area['location_x']:.6f}, {area['location_y']:.6f}) - 总使用次数：{int(area['total_count'])}\n")
        
        f.write("\n3. 高需求聚类 (K-means)\n")
        f.write("-" * 30 + "\n")
        for i, (_, cluster) in enumerate(high_demand_clusters.iterrows(), 1):
            f.write(f"{i}. 聚类中心：({cluster['center_x']:.6f}, {cluster['center_y']:.6f}) - 平均使用次数：{cluster['center_total_count']:.2f} - 簇半径：{cluster['cluster_radius']:.6f}\n")
        
        # 添加DBSCAN聚类结果
        if dbscan_cluster_stats is not None:
            f.write("\n4. DBSCAN聚类分析\n")
            f.write("-" * 30 + "\n")
            
            # 计算DBSCAN聚类统计
            num_clusters = len(dbscan_cluster_stats)
            num_noise_points = len(locations[locations['dbscan_cluster'] == -1])
            num_valid_clusters = num_clusters - (1 if -1 in dbscan_cluster_stats['cluster'].values else 0)
            
            f.write(f"总簇数（包含噪声点）：{num_clusters}\n")
            f.write(f"有效簇数（不含噪声点）：{num_valid_clusters}\n")
            f.write(f"噪声点数量：{num_noise_points}\n")
            f.write(f"噪声点占比：{num_noise_points/len(locations)*100:.2f}%\n\n")
            
            # 输出DBSCAN簇的统计信息（排除噪声点-1）
            valid_clusters = dbscan_cluster_stats[dbscan_cluster_stats['cluster'] != -1]
            if len(valid_clusters) > 0:
                f.write("主要簇统计信息（按站点数量排序）：\n")
                sorted_clusters = valid_clusters.sort_values(by='station_count', ascending=False).head(10)
                for i, (_, cluster) in enumerate(sorted_clusters.iterrows(), 1):
                    f.write(f"{i}. 簇编号：{int(cluster['cluster'])} - 站点数：{int(cluster['station_count'])} - 平均使用次数：{cluster['mean_total_count']:.2f} - 总使用次数：{int(cluster['sum_total_count'])}\n")
                
                # 添加按订单量（簇内骑行点数）排序的前30簇信息
                f.write("\n5. 按订单量排序的前30个簇（主要站点）：\n")
                f.write("-" * 30 + "\n")
                top30_clusters = valid_clusters.sort_values(by='sum_total_count', ascending=False).head(30)
                for i, (_, cluster) in enumerate(top30_clusters.iterrows(), 1):
                    f.write(f"{i}. 主要站点：簇编号{int(cluster['cluster'])} - 位置：({cluster['center_x']:.6f}, {cluster['center_y']:.6f}) - 订单量：{int(cluster['sum_total_count'])} - 站点数：{int(cluster['station_count'])}\n")
            
            # 输出噪声点信息
            if num_noise_points > 0:
                f.write(f"\n噪声点特征：\n")
                noise_points = locations[locations['dbscan_cluster'] == -1]
                f.write(f"   平均使用次数：{noise_points['total_count'].mean():.2f}\n")
                f.write(f"   最大使用次数：{noise_points['total_count'].max()}\n")
                f.write(f"   最小使用次数：{noise_points['total_count'].min()}\n")

# 主函数
def main():
    """空间聚类分析主函数"""
    # 定义文件路径
    input_file = '../data/processed_data.csv'
    
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)
    
    # 导入数据
    df = load_processed_data(input_file)
    
    # 处理位置数据
    print("\n2. 处理位置数据...")
    locations = process_location_data(df)
    
    # 执行K-means聚类
    print("\n3. 执行K-means聚类分析...")
    locations, cluster_centers_df = perform_kmeans_clustering(locations, n_clusters=10)
    
    # 执行DBSCAN聚类 - 进一步减小eps参数以解决大簇问题
    print("\n3.1 执行DBSCAN聚类分析...")
    locations, dbscan_cluster_stats = perform_dbscan_clustering(locations, eps=0.05, min_samples=3)
    print(f"   ✓ DBSCAN聚类完成，共识别出 {len(dbscan_cluster_stats)} 个簇（包含噪声点-1）")
    
    # 识别高需求区域
    print("\n4. 识别高需求区域...")
    high_demand_areas, high_demand_clusters = identify_high_demand_areas(locations, cluster_centers_df)
    
    # 5. 可视化结果 - 暂时跳过以快速生成DBSCAN结果文件
    print("\n5. 可视化聚类结果...")
    print("   ✓ 暂时跳过可视化步骤")
    
    # 6. 分析不同时段的空间分布 - 暂时跳过以快速生成DBSCAN结果文件
    print("\n6. 分析不同时段的空间分布...")
    print("   ✓ 暂时跳过不同时段空间分布分析")
    
    # 生成报告
    print("\n7. 生成空间分析报告...")
    try:
        generate_spatial_report(locations, high_demand_areas, high_demand_clusters, dbscan_cluster_stats)
        print("   ✓ 空间分析报告生成完成")
    except Exception as e:
        print(f"   ✗ 生成空间分析报告时出错: {str(e)}")
        print("   继续执行后续步骤...")
    
    # 保存所有聚类的站点数据结果
    print("\n8. 保存聚类结果表格...")
    try:
        # 确保results目录存在
        results_dir = os.path.join(os.path.dirname(os.getcwd()), 'results')
        os.makedirs(results_dir, exist_ok=True)
        print(f"   - results目录路径: {results_dir}")
        
        # 先保存DBSCAN聚类结果，这样即使其他文件保存失败，DBSCAN结果也能成功生成
        print("   DBSCAN聚类结果保存:")
        dbscan_stations_path = os.path.join(results_dir, 'dbscan_cluster_stations_new.csv')
        print(f"   - 准备保存DBSCAN所有聚类的站点数据至: {dbscan_stations_path}")
        try:
            locations[['location_x', 'location_y', 'start_count', 'end_count', 'total_count', 'dbscan_cluster']].to_csv(dbscan_stations_path, index=False, encoding='utf-8-sig')
            print(f"   - DBSCAN所有聚类的站点数据已保存至: {dbscan_stations_path}")
        except Exception as e:
            print(f"   - 保存DBSCAN站点数据时出错: {str(e)}")
        
        # 保存DBSCAN所有簇的详细统计信息到独立表格
        dbscan_all_clusters_path = os.path.join(results_dir, 'dbscan_all_clusters_stats.csv')
        print(f"   - 准备保存DBSCAN所有簇的详细统计信息至: {dbscan_all_clusters_path}")
        try:
            # 对簇进行排序，先显示有效簇（非噪声点），按簇编号排序
            valid_clusters = dbscan_cluster_stats[dbscan_cluster_stats['cluster'] != -1].sort_values(by='cluster')
            noise_cluster = dbscan_cluster_stats[dbscan_cluster_stats['cluster'] == -1]
            # 组合有效簇和噪声点
            all_clusters = pd.concat([valid_clusters, noise_cluster])
            
            # 保存到文件
            all_clusters.to_csv(dbscan_all_clusters_path, index=False, encoding='utf-8-sig')
            print(f"   - DBSCAN所有簇的详细统计信息已保存至: {dbscan_all_clusters_path}")
            print(f"   - 包含 {len(valid_clusters)} 个有效簇和 1 个噪声点簇")
        except Exception as e:
            print(f"   - 保存DBSCAN所有簇统计信息时出错: {str(e)}")
        
        # 保留原有的dbscan_stats文件（可能用于报告）
        dbscan_stats_path = os.path.join(results_dir, 'dbscan_cluster_stats_new.csv')
        print(f"   - 准备保存DBSCAN聚类统计信息至: {dbscan_stats_path}")
        try:
            dbscan_cluster_stats.to_csv(dbscan_stats_path, index=False, encoding='utf-8-sig')
            print(f"   - DBSCAN聚类统计信息已保存至: {dbscan_stats_path}")
        except Exception as e:
            print(f"   - 保存DBSCAN统计信息时出错: {str(e)}")
        
        # 保存其他聚类结果（可能会遇到权限问题）
        print("\n   其他聚类结果保存:")
        try:
            all_clusters_path = os.path.join(results_dir, 'all_cluster_stations.csv')
            print(f"   - 准备保存所有聚类的站点数据至: {all_clusters_path}")
            locations.to_csv(all_clusters_path, index=False, encoding='utf-8-sig')
            print(f"   - 所有聚类的站点数据已保存至: {all_clusters_path}")
        except Exception as e:
            print(f"   - 保存所有聚类站点数据时出错: {str(e)}")
        
        try:
            centers_path = os.path.join(results_dir, 'cluster_centers.csv')
            print(f"   - 准备保存聚类中心数据至: {centers_path}")
            cluster_centers_df.to_csv(centers_path, index=False, encoding='utf-8-sig')
            print(f"   - 聚类中心数据已保存至: {centers_path}")
        except Exception as e:
            print(f"   - 保存聚类中心数据时出错: {str(e)}")
        
        # 验证DBSCAN文件是否存在
        if os.path.exists(dbscan_stations_path):
            print(f"   ✓ DBSCAN所有聚类的站点数据文件已成功生成: {dbscan_stations_path}")
            print(f"   - 文件大小: {os.path.getsize(dbscan_stations_path)} 字节")
        else:
            print(f"   ✗ DBSCAN所有聚类的站点数据文件生成失败")
        
        if os.path.exists(dbscan_all_clusters_path):
            print(f"   ✓ DBSCAN所有簇的详细统计信息文件已成功生成: {dbscan_all_clusters_path}")
            print(f"   - 文件大小: {os.path.getsize(dbscan_all_clusters_path)} 字节")
        else:
            print(f"   ✗ DBSCAN所有簇的详细统计信息文件生成失败")
        
        # 保存按订单量排序的前30个簇信息
        print("\n   保存按订单量排序的前30个簇信息...")
        try:
            # 筛选有效簇并按订单量排序取前30
            valid_clusters = dbscan_cluster_stats[dbscan_cluster_stats['cluster'] != -1]
            top30_clusters = valid_clusters.sort_values(by='sum_total_count', ascending=False).head(30)
            
            # 保存前30簇信息
            top30_path = os.path.join(results_dir, 'dbscan_top30_clusters.csv')
            print(f"   - 准备保存按订单量排序的前30簇信息至: {top30_path}")
            top30_clusters[['cluster', 'center_x', 'center_y', 'sum_total_count', 'station_count']].to_csv(top30_path, index=False, encoding='utf-8-sig')
            print(f"   - 按订单量排序的前30簇信息已保存至: {top30_path}")
            
            # 验证文件是否存在
            if os.path.exists(top30_path):
                print(f"   ✓ 按订单量排序的前30簇信息文件已成功生成: {top30_path}")
                print(f"   - 文件大小: {os.path.getsize(top30_path)} 字节")
            else:
                print(f"   ✗ 按订单量排序的前30簇信息文件生成失败")
        except Exception as e:
            print(f"   - 保存按订单量排序的前30簇信息时出错: {str(e)}")
    except Exception as e:
        print(f"   ✗ 保存表格文件时出错: {str(e)}")
        import traceback
        print(f"   详细错误信息: {traceback.format_exc()}")
    
    print("\n空间聚类分析完成！")

if __name__ == "__main__":
    main()