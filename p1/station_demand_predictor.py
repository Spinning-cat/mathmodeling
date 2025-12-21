import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """
    加载特征工程后的数据
    """
    print("1. 加载特征工程后的数据...")
    features_path = "E:/QI002/Documents/trae_projects/math model HW/results/top30_features.csv"
    df = pd.read_csv(features_path)
    print(f"   ✓ 数据加载完成，共 {len(df)} 条记录")
    return df

def train_model_for_cluster(df, cluster_id, target_col='rent_count'):
    """
    为指定簇训练XGBoost模型
    """
    # 筛选该簇的数据
    cluster_data = df[df['cluster_id'] == cluster_id].copy()
    
    # 定义特征和目标
    features = [
        'hour', 'minute', 'is_morning_peak', 'is_evening_peak', 'is_peak', 'time_period',
        'rent_count_prev_1avg', 'return_count_prev_1avg',
        'rent_count_prev_3avg', 'return_count_prev_3avg',
        'rent_count_prev_6avg', 'return_count_prev_6avg',
        'rent_count_prev_1max', 'return_count_prev_1max',
        'rent_count_prev_3max', 'return_count_prev_3max',
        'rent_count_prev_6max', 'return_count_prev_6max',
        'rent_count_prev_1min', 'return_count_prev_1min',
        'rent_count_prev_3min', 'return_count_prev_3min',
        'rent_count_prev_6min', 'return_count_prev_6min',
        'rent_count_prev_1sum', 'return_count_prev_1sum',
        'rent_count_prev_3sum', 'return_count_prev_3sum',
        'rent_count_prev_6sum', 'return_count_prev_6sum',
        'rent_count_change', 'rent_count_change_rate',
        'return_count_change', 'return_count_change_rate',
        'net_count', 'net_count_change'
    ]
    
    # 移除与目标相关的特征（避免数据泄漏）
    if target_col == 'rent_count':
        features = [f for f in features if 'rent' not in f and f != 'net_count' and f != 'net_count_change']
    else:  # return_count
        features = [f for f in features if 'return' not in f and f != 'net_count' and f != 'net_count_change']
    
    X = cluster_data[features]
    y = cluster_data[target_col]
    
    # 训练模型（由于时间序列数据较少，我们使用全部数据训练）
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y, verbose=False)
    
    # 计算模型评分
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return model, features, {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def predict_morning_peak(df, clusters, rent_models, return_models, rent_features, return_features):
    """
    预测早高峰（7:00-9:00）的租赁量和归还量
    """
    print("3. 预测早高峰时段需求...")
    
    # 定义早高峰时间片（7:00-9:00 对应时间片15-18）
    morning_peak_time_slots = [15, 16, 17, 18]
    
    # 创建空的预测结果列表
    predictions = []
    
    print(f"   调试信息：共有 {len(clusters)} 个簇需要预测")
    print(f"   调试信息：需要预测的时间片为 {morning_peak_time_slots}")
    
    for i, cluster_id in enumerate(clusters):
        print(f"   正在处理簇 {i+1}/{len(clusters)}: {cluster_id}")
        # 获取该簇的历史数据
        cluster_data = df[df['cluster_id'] == cluster_id].copy()
        
        for time_slot in morning_peak_time_slots:
            print(f"      正在预测时间片 {time_slot}")
            # 找到最接近的历史数据作为特征
            # 对于时间片15（7:00），我们使用时间片14（6:30）的数据作为前一个时间片
            prev_time_slot = time_slot - 1
            
            # 检查前一个时间片是否存在
            prev_data_filter = cluster_data[cluster_data['time_slot'] == prev_time_slot]
            if len(prev_data_filter) == 0:
                print(f"      警告：簇 {cluster_id} 的时间片 {prev_time_slot} 不存在")
                continue
                
            prev_data = prev_data_filter.iloc[0]
            
            # 创建特征向量
            rent_features_dict = {}
            return_features_dict = {}
            
            # 首先填充时间相关特征
            hour = (time_slot - 1) // 2
            minute = ((time_slot - 1) % 2) * 30
            is_morning_peak = 1 if 7 <= hour < 9 else 0
            is_evening_peak = 1 if 17 <= hour < 19 else 0
            is_peak = is_morning_peak or is_evening_peak
            time_period = 1  # 6-12点属于上午
            
            # 更新特征字典
            rent_features_dict.update({
                'hour': hour,
                'minute': minute,
                'is_morning_peak': is_morning_peak,
                'is_evening_peak': is_evening_peak,
                'is_peak': is_peak,
                'time_period': time_period
            })
            
            return_features_dict.update({
                'hour': hour,
                'minute': minute,
                'is_morning_peak': is_morning_peak,
                'is_evening_peak': is_evening_peak,
                'is_peak': is_peak,
                'time_period': time_period
            })
            
            # 填充其他特征
            for feature in rent_features:
                if feature in prev_data.index:
                    rent_features_dict[feature] = prev_data[feature]
            
            for feature in return_features:
                if feature in prev_data.index:
                    return_features_dict[feature] = prev_data[feature]
            
            # 转换为DataFrame
            rent_X_pred = pd.DataFrame([rent_features_dict])[rent_features]
            return_X_pred = pd.DataFrame([return_features_dict])[return_features]
            
            # 进行预测
            predicted_rent = max(0, rent_models[cluster_id].predict(rent_X_pred)[0])
            predicted_return = max(0, return_models[cluster_id].predict(return_X_pred)[0])
            
            # 计算净需求缺口（租赁量 - 归还量）
            net_gap = predicted_rent - predicted_return
            
            # 添加到预测结果
            predictions.append({
                'cluster_id': cluster_id,
                'time_slot': time_slot,
                'hour': hour,
                'minute': minute,
                'predicted_rent_count': round(predicted_rent),
                'predicted_return_count': round(predicted_return),
                'net_demand_gap': round(net_gap)
            })
    
    predictions_df = pd.DataFrame(predictions)
    print(f"   ✓ 早高峰预测完成，共 {len(predictions_df)} 条记录")
    return predictions_df

def visualize_results(predictions_df):
    """
    可视化预测结果
    """
    print("4. 可视化预测结果...")
    
    # 创建结果目录
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 1. 早高峰各时间片的平均净需求缺口
    plt.figure(figsize=(10, 6))
    time_slot_avg = predictions_df.groupby('time_slot')['net_demand_gap'].mean()
    time_slot_avg.plot(kind='bar', color='skyblue')
    plt.title('早高峰各时间片平均净需求缺口')
    plt.xlabel('时间片')
    plt.ylabel('净需求缺口（租赁量-归还量）')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.savefig(f"{results_dir}/morning_peak_net_demand.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 各簇早高峰总净需求缺口
    plt.figure(figsize=(12, 8))
    cluster_total = predictions_df.groupby('cluster_id')['net_demand_gap'].sum().sort_values(ascending=False)
    cluster_total.plot(kind='bar', color='lightgreen')
    plt.title('各簇早高峰总净需求缺口')
    plt.xlabel('簇ID')
    plt.ylabel('总净需求缺口（租赁量-归还量）')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(f"{results_dir}/cluster_morning_peak_total.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✓ 可视化完成")

def main():
    """
    主函数
    """
    # 1. 加载数据
    df = load_data()
    
    # 2. 为每个簇训练模型
    print("2. 训练预测模型...")
    clusters = df['cluster_id'].unique()
    
    rent_models = {}
    return_models = {}
    rent_features = {}
    return_features = {}
    model_metrics = {}
    
    for cluster_id in clusters:
        print(f"   - 训练簇 {cluster_id} 的租赁量模型...")
        rent_model, r_features, r_metrics = train_model_for_cluster(df, cluster_id, target_col='rent_count')
        rent_models[cluster_id] = rent_model
        rent_features[cluster_id] = r_features
        
        print(f"   - 训练簇 {cluster_id} 的归还量模型...")
        return_model, ret_features, ret_metrics = train_model_for_cluster(df, cluster_id, target_col='return_count')
        return_models[cluster_id] = return_model
        return_features[cluster_id] = ret_features
        
        model_metrics[cluster_id] = {"rent": r_metrics, "return": ret_metrics}
    
    # 3. 预测早高峰需求
    predictions_df = predict_morning_peak(df, clusters, rent_models, return_models, 
                                          rent_features[clusters[0]], return_features[clusters[0]])
    
    # 4. 可视化结果
    visualize_results(predictions_df)
    
    # 5. 保存预测结果
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "station_demand_gap.csv")
    predictions_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"5. 预测结果已保存至 {results_path}")
    
    # 6. 计算每个簇的总净需求缺口
    cluster_total_gap = predictions_df.groupby('cluster_id')['net_demand_gap'].sum()
    print("\n早高峰各簇总净需求缺口：")
    print(cluster_total_gap)
    
    print("\n任务完成！")

if __name__ == "__main__":
    main()