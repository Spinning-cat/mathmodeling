import pandas as pd
import numpy as np
import os

def feature_engineering():
    # 设置文件路径
    input_path = "E:\\QI002\\Documents\\trae_projects\\math model HW\\results\\top30_count.csv"
    output_path = "E:\\QI002\\Documents\\trae_projects\\math model HW\\results\\top30_features.csv"
    
    print("1. 读取时间序列数据...")
    # 读取时间序列数据
    df = pd.read_csv(input_path)
    print(f"   ✓ 读取完成，共 {len(df)} 条记录")
    
    print("2. 添加时间特征...")
    # 时间片转换为小时和分钟
    df['hour'] = (df['time_slot'] - 1) // 2
    df['minute'] = ((df['time_slot'] - 1) % 2) * 30
    
    # 是否早高峰 (7:00-9:00)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
    
    # 是否晚高峰 (17:00-19:00)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] < 19)).astype(int)
    
    # 是否高峰期
    df['is_peak'] = df['is_morning_peak'] | df['is_evening_peak']
    
    # 时间类型：凌晨、上午、下午、晚上
    def get_time_period(hour):
        if 0 <= hour < 6:
            return 0  # 凌晨
        elif 6 <= hour < 12:
            return 1  # 上午
        elif 12 <= hour < 18:
            return 2  # 下午
        else:
            return 3  # 晚上
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    print("3. 添加历史强度特征...")
    # 对每个簇分别处理
    clusters = df['cluster_id'].unique()
    all_features = []
    
    for cluster in clusters:
        cluster_data = df[df['cluster_id'] == cluster].copy()
        cluster_data = cluster_data.sort_values('time_slot')
        
        # 计算过去1、3、6个时间片的平均租赁量和归还量
        for window in [1, 3, 6]:
            cluster_data[f'rent_count_prev_{window}avg'] = cluster_data['rent_count'].rolling(window=window).mean().shift(1)
            cluster_data[f'return_count_prev_{window}avg'] = cluster_data['return_count'].rolling(window=window).mean().shift(1)
        
        # 计算过去1、3、6个时间片的最大租赁量和归还量
        for window in [1, 3, 6]:
            cluster_data[f'rent_count_prev_{window}max'] = cluster_data['rent_count'].rolling(window=window).max().shift(1)
            cluster_data[f'return_count_prev_{window}max'] = cluster_data['return_count'].rolling(window=window).max().shift(1)
        
        # 计算过去1、3、6个时间片的最小租赁量和归还量
        for window in [1, 3, 6]:
            cluster_data[f'rent_count_prev_{window}min'] = cluster_data['rent_count'].rolling(window=window).min().shift(1)
            cluster_data[f'return_count_prev_{window}min'] = cluster_data['return_count'].rolling(window=window).min().shift(1)
        
        # 计算过去1、3、6个时间片的总量
        for window in [1, 3, 6]:
            cluster_data[f'rent_count_prev_{window}sum'] = cluster_data['rent_count'].rolling(window=window).sum().shift(1)
            cluster_data[f'return_count_prev_{window}sum'] = cluster_data['return_count'].rolling(window=window).sum().shift(1)
        
        all_features.append(cluster_data)
    
    # 合并所有簇的数据
    df_features = pd.concat(all_features, ignore_index=True)
    
    print("4. 添加短期变化特征...")
    # 计算相邻时间片的变化率
    all_changes = []
    
    for cluster in clusters:
        cluster_data = df_features[df_features['cluster_id'] == cluster].copy()
        cluster_data = cluster_data.sort_values('time_slot')
        
        # 租赁量变化率
        cluster_data['rent_count_change'] = cluster_data['rent_count'].diff()
        cluster_data['rent_count_change_rate'] = cluster_data['rent_count'].pct_change()
        
        # 归还量变化率
        cluster_data['return_count_change'] = cluster_data['return_count'].diff()
        cluster_data['return_count_change_rate'] = cluster_data['return_count'].pct_change()
        
        # 净变化（租赁量-归还量）
        cluster_data['net_count'] = cluster_data['rent_count'] - cluster_data['return_count']
        cluster_data['net_count_change'] = cluster_data['net_count'].diff()
        
        all_changes.append(cluster_data)
    
    # 合并所有簇的数据
    df_final = pd.concat(all_changes, ignore_index=True)
    
    # 处理NaN值和inf值（由于移位和变化率计算产生的）
    df_final = df_final.fillna(0)
    df_final = df_final.replace([np.inf, -np.inf], 0)
    
    print("5. 保存特征工程结果...")
    # 保存到CSV文件
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ 特征工程结果已保存至 {output_path}")
    print(f"   ✓ 共 {len(df_final)} 条记录")
    
    # 打印前几行数据查看结果
    print("\n结果示例：")
    print(df_final[['cluster_id', 'time_slot', 'rent_count', 'return_count', 'hour', 'minute', 'is_morning_peak', 'is_evening_peak', 'rent_count_prev_1avg', 'return_count_prev_1avg', 'rent_count_change_rate', 'return_count_change_rate', 'net_count']].head(10))

if __name__ == "__main__":
    feature_engineering()
