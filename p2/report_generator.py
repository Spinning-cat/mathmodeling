#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块：生成调度优化的分析报告
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from config import RESULTS_CONFIG, SCHEDULING_CONFIG


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config):
        """
        初始化报告生成器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
    
    def generate_report(self, solution: dict, scheduling_df: pd.DataFrame, 
                       inventory_df: pd.DataFrame, demand_gap_df: pd.DataFrame):
        """
        生成调度优化分析报告
        
        Args:
            solution: 优化结果
            scheduling_df: 调度方案DataFrame
            inventory_df: 站点库存信息
            demand_gap_df: 需求预测DataFrame
        """
        print("\n=== 开始生成分析报告 ===\n")
        
        report_path = RESULTS_CONFIG['SCHEDULING_REPORT']
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_sched_cost = scheduling_df["cost"].sum() if len(scheduling_df) > 0 else 0
        total_cost = solution.get('total_cost', 0)
        # 确保惩罚成本不为负（避免浮点数精度问题）
        total_penalty = max(total_cost - total_sched_cost, 0)
        total_bikes = scheduling_df["bike_count"].sum() if len(scheduling_df) > 0 else 0
        total_paths = len(scheduling_df)
        
        # 单位调度成本（避免除以0，保留2位小数）
        avg_unit_cost = round(total_sched_cost / total_bikes, 2) if total_bikes > 0 else 0.00
        # 修复：平均单次调度车辆数（保留1位小数）
        avg_bikes_per_trip = round(total_bikes / total_paths, 1) if total_paths > 0 else 0.0
        
        # 加权需求满足率（按需求规模加权，更合理）
        # 确保需求满足率不超过100%
        inventory_df["demand_satisfaction_rate"] = (inventory_df["final_inventory"] / inventory_df["total_demand"]).clip(0, 1.0)
        inventory_df["weighted_satisfaction"] = inventory_df["demand_satisfaction_rate"] * inventory_df["total_demand"]
        weighted_satisfaction = inventory_df["weighted_satisfaction"].sum() / inventory_df["total_demand"].sum()
        
        # 按站点类型统计
        type_stats = inventory_df.groupby("station_type").agg({
            "station_id": "count",
            "total_demand": "sum",
            "weighted_satisfaction": "sum"
        }).rename(columns={"station_id": "station_count"})
        type_stats["satisfaction_rate"] = (type_stats["weighted_satisfaction"] / type_stats["total_demand"]).round(4)
        
        # 获取各类站点数量（防止KeyError）
        large_count = type_stats.loc['large', 'station_count'] if 'large' in type_stats.index else 0
        medium_count = type_stats.loc['medium', 'station_count'] if 'medium' in type_stats.index else 0
        small_count = type_stats.loc['small', 'station_count'] if 'small' in type_stats.index else 0
        
        # 获取各类站点满足率（防止KeyError）
        large_satisfaction = round(type_stats.loc['large', 'satisfaction_rate']*100, 1) if 'large' in type_stats.index else 0.0
        medium_satisfaction = round(type_stats.loc['medium', 'satisfaction_rate']*100, 1) if 'medium' in type_stats.index else 0.0
        small_satisfaction = round(type_stats.loc['small', 'satisfaction_rate']*100, 1) if 'small' in type_stats.index else 0.0
        
        # 生成报告内容
        report = f"""# 共享单车调度优化分析报告
## 1. 调度基本信息
- 调度日期：{datetime.now().strftime('%Y-%m-%d')}
- 调度时间窗口：6小时（00:00-06:00）
- 数据来源：p1预处理数据
- 库存计算逻辑：初始库存 = 调度前一天还车数 - 调度前一天启动数 + 最低保障库存
- 调度规则：仅从最近{SCHEDULING_CONFIG['K_NEIGHBORS']}个邻居站点调度（距离≤{SCHEDULING_CONFIG['MAX_DISTANCE']}km）
- 涉及站点数量：{len(inventory_df)}个（大型{large_count}个+中型{medium_count}个+小型{small_count}个）
- 目标服务水平：{SCHEDULING_CONFIG['DEMAND_SATISFACTION_RATIO']:.1%}

## 2. 调度结果统计
### 2.1 成本统计
- 总调度成本：{round(total_cost, 2)}元
- 调度运输成本：{round(total_sched_cost, 2)}元
- 未满足需求惩罚成本：{round(total_penalty, 2)}元

### 2.2 车辆调度统计
- 总调度车辆数：{total_bikes}辆
- 实际调度路径数：{total_paths}条
- 平均单次调度车辆数：{avg_bikes_per_trip}辆
- 平均调度距离：{round(scheduling_df['distance_km'].mean(), 2) if len(scheduling_df) > 0 else 0.00}km（≤2.5km）
- 平均单位调度成本：{avg_unit_cost}元/辆

### 2.3 需求满足情况
- 整体需求满足率（加权）：{round(weighted_satisfaction*100, 1)}%
- 大型站点满足率：{large_satisfaction}%
- 中型站点满足率：{medium_satisfaction}%
- 小型站点满足率：{small_satisfaction}%
- 满足率≥95%的站点数：{len(inventory_df[inventory_df['demand_satisfaction_rate'] >= 0.95])}个

## 3. 关键站点详情
### 3.1 调出车辆最多的站点（Top3）
{inventory_df.nlargest(3, 'out_bikes')[['station_id', 'station_type', 'initial_inventory', 'out_bikes', 'total_demand']].to_string(index=False)}

### 3.2 调入车辆最多的站点（Top3）
{inventory_df.nlargest(3, 'in_bikes')[['station_id', 'station_type', 'initial_inventory', 'in_bikes', 'total_demand']].to_string(index=False)}

### 3.3 需求满足率最高的站点（Top3）
{inventory_df.nlargest(3, 'demand_satisfaction_rate')[['station_id', 'station_type', 'demand_satisfaction_rate', 'final_inventory', 'total_demand']].to_string(index=False)}

"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"   ✓ 分析报告已保存至: {report_path}")
    
    def _write_summary(self, f, solution: dict, scheduling_df: pd.DataFrame,
                      inventory_df: pd.DataFrame):
        """写入执行摘要"""
        status = solution.get('status', 'Unknown')
        total_cost = solution.get('total_cost', 0)
        total_bikes = solution.get('total_bikes_scheduled', 0)
        
        f.write("### 1.1 优化状态\n\n")
        f.write(f"- **求解状态**: {status}\n")
        f.write(f"- **总调度成本**: {total_cost:.2f} 元\n")
        f.write(f"- **总调度单车数**: {int(total_bikes)} 辆\n")
        f.write(f"- **调度路径数**: {len(scheduling_df)} 条\n\n")
        
        if not scheduling_df.empty:
            avg_distance = scheduling_df['distance_km'].mean()
            total_distance = scheduling_df['distance_km'].sum()
            f.write(f"- **平均调度距离**: {avg_distance:.2f} km\n")
            f.write(f"- **总调度距离**: {total_distance:.2f} km\n\n")
        
        f.write("### 1.2 主要发现\n\n")
        
        if status == 'Optimal' and not scheduling_df.empty:
            # 找出调度量最大的站点
            top_suppliers = scheduling_df.groupby('from_cluster_id')['scheduling_quantity'].sum().nlargest(5)
            top_receivers = scheduling_df.groupby('to_cluster_id')['scheduling_quantity'].sum().nlargest(5)
            
            f.write("**主要供应站点（调出量前5）**:\n")
            for station_id, qty in top_suppliers.items():
                f.write(f"- 站点 {int(station_id)}: {int(qty)} 辆\n")
            
            f.write("\n**主要需求站点（接收量前5）**:\n")
            for station_id, qty in top_receivers.items():
                f.write(f"- 站点 {int(station_id)}: {int(qty)} 辆\n")
        else:
            f.write("- 未找到可行解或求解失败\n")
    
    def _write_model_description(self, f):
        """写入模型说明"""
        f.write("### 2.1 问题描述\n\n")
        f.write("本问题是一个单车调度优化问题，目标是在满足主要站点需求的前提下，")
        f.write("最小化调度成本。主要约束包括：\n\n")
        f.write("1. **供给约束**: 每个站点的调度量不能超过其可用库存\n")
        f.write("2. **需求约束**: 主要站点的需求必须得到满足\n")
        f.write("3. **库存下限约束**: 调度后每个站点的库存不能低于最低阈值\n\n")
        
        f.write("### 2.2 优化算法\n\n")
        f.write("**算法类型**: 线性规划（Linear Programming, LP）\n\n")
        f.write("**求解器**: PuLP (Python Linear Programming)\n\n")
        f.write("**模型特点**:\n\n")
        f.write("- 决策变量: 整数型（x[i][j] 表示从站点i到站点j的调度量）\n")
        f.write("- 目标函数: 最小化总调度成本（距离成本 + 固定成本）\n")
        f.write("- 约束条件: 线性约束\n\n")
        
        f.write("### 2.3 成本模型\n\n")
        f.write("调度成本由两部分组成：\n\n")
        f.write(f"- **距离成本**: {self.config['DISTANCE_COST_PER_KM']} 元/公里\n")
        f.write(f"- **固定成本**: {self.config['FIXED_COST_PER_TRIP']} 元/次（按单车数量分摊）\n\n")
        f.write("总成本 = Σ(距离 × 距离成本系数 × 调度量) + Σ(固定成本/车辆容量 × 调度量)\n\n")
        
        f.write("### 2.4 参数设置\n\n")
        f.write(f"- **最低库存阈值比例**: {self.config['MIN_INVENTORY_THRESHOLD_RATIO'] * 100}%\n")
        f.write(f"- **需求满足率**: {self.config['DEMAND_SATISFACTION_RATIO'] * 100}%\n")
        f.write(f"- **车辆容量**: {self.config['VEHICLE_CAPACITY']} 辆/车\n\n")
    
    def _write_data_overview(self, f, inventory_df: pd.DataFrame,
                            demand_gap_df: pd.DataFrame):
        """写入数据概况"""
        f.write("### 3.1 站点概况\n\n")
        f.write(f"- **总站点数**: {len(inventory_df)}\n")
        f.write(f"- **主要站点数**: {inventory_df['is_main_station'].sum()}\n")
        f.write(f"- **总初始库存**: {inventory_df['initial_inventory'].sum()} 辆\n")
        f.write(f"- **平均初始库存**: {inventory_df['initial_inventory'].mean():.1f} 辆\n")
        f.write(f"- **最大初始库存**: {inventory_df['initial_inventory'].max()} 辆\n")
        f.write(f"- **最小初始库存**: {inventory_df['initial_inventory'].min()} 辆\n\n")
        
        f.write("### 3.2 需求概况\n\n")
        total_demand = demand_gap_df['net_demand_gap'].sum()
        positive_demand = demand_gap_df[demand_gap_df['net_demand_gap'] > 0]['net_demand_gap'].sum()
        f.write(f"- **总净需求**: {total_demand:.0f} 辆\n")
        f.write(f"- **正需求（需要补充）**: {positive_demand:.0f} 辆\n")
        f.write(f"- **需求站点数**: {(demand_gap_df['net_demand_gap'] > 0).sum()} 个\n\n")
    
    def _write_optimization_results(self, f, solution: dict,
                                   scheduling_df: pd.DataFrame):
        """写入优化结果"""
        status = solution.get('status', 'Unknown')
        f.write(f"**求解状态**: {status}\n\n")
        
        if status == 'Optimal' and not scheduling_df.empty:
            f.write("### 4.1 调度统计\n\n")
            f.write(f"- **总调度成本**: {solution.get('total_cost', 0):.2f} 元\n")
            f.write(f"- **总调度单车数**: {int(solution.get('total_bikes_scheduled', 0))} 辆\n")
            f.write(f"- **调度路径数**: {len(scheduling_df)} 条\n")
            f.write(f"- **平均每辆单车调度成本**: {solution.get('total_cost', 0) / max(1, solution.get('total_bikes_scheduled', 1)):.2f} 元/辆\n\n")
            
            f.write("### 4.2 调度路径详情（Top 20）\n\n")
            f.write("| 出发站点 | 目标站点 | 调度量(辆) | 距离(km) | 成本(元) |\n")
            f.write("|---------|---------|-----------|---------|---------|\n")
            
            top_routes = scheduling_df.head(20)
            for _, route in top_routes.iterrows():
                f.write(f"| {int(route['from_cluster_id'])} | {int(route['to_cluster_id'])} | "
                       f"{int(route['scheduling_quantity'])} | {route['distance_km']:.2f} | "
                       f"{route['scheduling_cost']:.2f} |\n")
        else:
            f.write("未找到可行解，请检查约束条件或调整参数。\n\n")
    
    def _write_scheduling_analysis(self, f, scheduling_df: pd.DataFrame,
                                  inventory_df: pd.DataFrame):
        """写入调度方案分析"""
        if scheduling_df.empty:
            f.write("无调度方案数据。\n\n")
            return
        
        # 按站点统计
        out_stats = scheduling_df.groupby('from_cluster_id').agg({
            'scheduling_quantity': 'sum',
            'scheduling_cost': 'sum'
        }).sort_values('scheduling_quantity', ascending=False)
        
        in_stats = scheduling_df.groupby('to_cluster_id').agg({
            'scheduling_quantity': 'sum',
            'scheduling_cost': 'sum'
        }).sort_values('scheduling_quantity', ascending=False)
        
        f.write("### 5.1 主要供应站点（调出量Top 10）\n\n")
        f.write("| 站点ID | 调出量(辆) | 调出成本(元) |\n")
        f.write("|-------|-----------|------------|\n")
        
        for station_id, row in out_stats.head(10).iterrows():
            f.write(f"| {int(station_id)} | {int(row['scheduling_quantity'])} | "
                   f"{row['scheduling_cost']:.2f} |\n")
        
        f.write("\n### 5.2 主要需求站点（接收量Top 10）\n\n")
        f.write("| 站点ID | 接收量(辆) | 接收成本(元) |\n")
        f.write("|-------|-----------|------------|\n")
        
        for station_id, row in in_stats.head(10).iterrows():
            f.write(f"| {int(station_id)} | {int(row['scheduling_quantity'])} | "
                   f"{row['scheduling_cost']:.2f} |\n")
        
        # 距离分析
        f.write("\n### 5.3 调度距离分布\n\n")
        distance_ranges = [
            (0, 2, '0-2km'),
            (2, 5, '2-5km'),
            (5, 10, '5-10km'),
            (10, 20, '10-20km'),
            (20, float('inf'), '20km+'),
        ]
        
        for min_dist, max_dist, label in distance_ranges:
            routes = scheduling_df[
                (scheduling_df['distance_km'] >= min_dist) & 
                (scheduling_df['distance_km'] < max_dist)
            ]
            if len(routes) > 0:
                total_qty = routes['scheduling_quantity'].sum()
                total_cost = routes['scheduling_cost'].sum()
                f.write(f"- **{label}**: {len(routes)} 条路径, {int(total_qty)} 辆, {total_cost:.2f} 元\n")
    
    def _write_conclusions(self, f, solution: dict, scheduling_df: pd.DataFrame):
        """写入结论与建议"""
        f.write("### 6.1 主要结论\n\n")
        
        if solution.get('status') == 'Optimal' and not scheduling_df.empty:
            total_cost = solution.get('total_cost', 0)
            total_bikes = solution.get('total_bikes_scheduled', 0)
            avg_cost_per_bike = total_cost / max(1, total_bikes)
            
            f.write(f"1. 优化模型成功求解，找到了最优调度方案。\n")
            f.write(f"2. 总调度成本为 {total_cost:.2f} 元，平均每辆单车调度成本为 {avg_cost_per_bike:.2f} 元。\n")
            f.write(f"3. 共调度 {int(total_bikes)} 辆单车，涉及 {len(scheduling_df)} 条调度路径。\n\n")
            
            # 分析调度效率
            if not scheduling_df.empty:
                avg_distance = scheduling_df['distance_km'].mean()
                f.write(f"4. 平均调度距离为 {avg_distance:.2f} km，调度范围合理。\n\n")
        else:
            f.write("1. 优化模型未能找到可行解，可能需要调整约束条件或参数设置。\n\n")
        
        f.write("### 6.2 优化建议\n\n")
        f.write("1. **调度时间安排**: 建议在夜间（22:00-6:00）进行大规模调度，避免影响用户使用。\n")
        f.write("2. **车辆配置**: 根据调度距离和数量，合理配置调度车辆数量和容量。\n")
        f.write("3. **动态调整**: 根据实际运营数据，定期更新需求预测和调度方案。\n")
        f.write("4. **成本控制**: 优先调度距离较近的站点，降低运输成本。\n")
        f.write("5. **库存管理**: 建立合理的库存预警机制，提前进行调度准备。\n\n")

