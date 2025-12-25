#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块：生成调度方案的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from config import VISUALIZATION_CONFIG, RESULTS_CONFIG


class SchedulingVisualizer:
    """调度可视化器"""
    
    def __init__(self, config):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def visualize_scheduling_plan(self, scheduling_df: pd.DataFrame,
                                  inventory_df: pd.DataFrame,
                                  top30_clusters: pd.DataFrame, total_cost: float):
        """
        可视化调度结果：供需对比+流向图+满足率分布+成本构成
        
        Args:
            scheduling_df: 调度方案DataFrame
            inventory_df: 站点库存信息
            top30_clusters: 30个主要站点信息
            total_cost: 总调度成本
        """
        print("\n=== 开始生成可视化图表 ===\n")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 确保结果目录存在
        results_dir = RESULTS_CONFIG['VISUALIZATION'].parent
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Top10高需求站点供需对比
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        self._plot_top10_demand_supply(ax1, inventory_df)
        plt.tight_layout()
        path1 = results_dir / "top10_demand_supply.png"
        plt.savefig(str(path1), dpi=self.config['DPI'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ 高需求站点供需对比图已保存至: {path1}")
        
        # 2. 站点分布与调度流向图
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        self._plot_station_flow_map(ax2, scheduling_df, inventory_df, top30_clusters)
        plt.tight_layout()
        path2 = results_dir / "station_flow_map.png"
        plt.savefig(str(path2), dpi=self.config['DPI'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ 站点分布与调度流向图已保存至: {path2}")
        
        # 3. 需求满足率分布
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        self._plot_demand_satisfaction_distribution(ax3, inventory_df)
        plt.tight_layout()
        path3 = results_dir / "demand_satisfaction_distribution.png"
        plt.savefig(str(path3), dpi=self.config['DPI'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ 需求满足率分布图已保存至: {path3}")
        
        # 4. 成本构成饼图
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        self._plot_cost_composition(ax4, scheduling_df, total_cost)
        plt.tight_layout()
        path4 = results_dir / "cost_composition.png"
        plt.savefig(str(path4), dpi=self.config['DPI'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ 成本构成饼图已保存至: {path4}")
        
        # 同时生成合并版本
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        self._plot_top10_demand_supply(ax1, inventory_df)
        self._plot_station_flow_map(ax2, scheduling_df, inventory_df, top30_clusters)
        self._plot_demand_satisfaction_distribution(ax3, inventory_df)
        self._plot_cost_composition(ax4, scheduling_df, total_cost)
        plt.tight_layout()
        combined_path = RESULTS_CONFIG['VISUALIZATION']
        plt.savefig(str(combined_path), dpi=self.config['DPI'], bbox_inches='tight')
        plt.close()
        print(f"   ✓ 合并可视化图表已保存至: {combined_path}")
    
    def _plot_top10_demand_supply(self, ax, inventory_df: pd.DataFrame):
        """绘制Top10高需求站点供需对比图"""
        # 排序获取Top10高需求站点
        top10 = inventory_df.sort_values("total_demand", ascending=False).head(10)
        x_pos = np.arange(len(top10))
        width = 0.25
        
        # 绘制柱状图
        ax.bar(x_pos - width, top10["initial_inventory"], width, label="初始库存", color="#FF6B6B")
        ax.bar(x_pos, top10["final_inventory"], width, label="调度后库存", color="#4ECDC4")
        ax.bar(x_pos + width, top10["total_demand"] * 0.9, width, label=f"目标需求(90%)", color="#45B7D1")
        
        # 设置坐标轴和标题
        ax.set_xlabel("站点")
        ax.set_ylabel("车辆数量")
        ax.set_title("Top10高需求站点供需状态")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top10["cluster_id"], rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_station_flow_map(self, ax, scheduling_df: pd.DataFrame, inventory_df: pd.DataFrame, top30_clusters: pd.DataFrame):
        """绘制站点分布与调度流向图"""
        bounds = self.config['MAP_BOUNDS']
        
        # 画站点（按类型区分颜色和大小，放大散点）
        type_color = {"large": "#E53935", "medium": "#FB8C00", "small": "#43A047"}
        
        # 获取参与调度的站点ID（调出和调入站点）
        if len(scheduling_df) > 0:
            involved_stations = set(scheduling_df["from_station"].tolist() + scheduling_df["to_station"].tolist())
        else:
            involved_stations = set()
        
        for station_type, color in type_color.items():
            mask = inventory_df["station_type"] == station_type
            
            # 对于小站点，只显示参与调度的站点
            if station_type == "small" and len(scheduling_df) > 0:
                mask &= inventory_df["station_id"].isin(involved_stations)
            
            if not mask.any():
                continue
            ax.scatter(
                inventory_df[mask]["center_x"], inventory_df[mask]["center_y"],
                s=inventory_df[mask]["sum_total_count"] * 1.5, alpha=0.7, c=color, label=f"{station_type}站点"
            )
        
        # 画调度箭头（按调度量区分颜色，进一步缩小组距）
        if len(scheduling_df) > 0:
            # 进一步缩小组距，每个区间2-3辆，提高区分度
            color_ranges = [(1, 10, "#2196F3", "1-10辆"), 
                           (11, 20, "#1976D2", "11-20辆"), 
                           (21, 30, "#3F51B5", "21-30辆"), 
                           (31, 40, "#303F9F", "31-40辆"), 
                           (41, 100, "#283593", "41辆以上")]
            
            for min_b, max_b, color, label in color_ranges:
                mask = (scheduling_df["bike_count"] >= min_b) & (scheduling_df["bike_count"] <= max_b)
                filtered_routes = scheduling_df[mask]
                
                for _, row in filtered_routes.iterrows():
                    from_sta = inventory_df[inventory_df["station_id"] == row["from_station"]].iloc[0]
                    to_sta = inventory_df[inventory_df["station_id"] == row["to_station"]].iloc[0]
                    
                    # 箭头粗细适配调度量
                    line_width = np.interp(row["bike_count"], [1, 50], [0.8, 3.5])
                    
                    # 缩短箭头避免重叠
                    dx = (to_sta["center_x"] - from_sta["center_x"]) * 0.7
                    dy = (to_sta["center_y"] - from_sta["center_y"]) * 0.7
                    
                    ax.arrow(
                        from_sta["center_x"], from_sta["center_y"],
                        dx, dy,
                        head_width=0.0005, head_length=0.0005,
                        fc=color, ec=color, alpha=0.6, linewidth=line_width,
                        length_includes_head=True
                    )
                
                if not filtered_routes.empty:
                    ax.plot([], [], color=color, linewidth=2, marker=">", markersize=8, label=label)
        
        # 设置坐标轴和标题
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        ax.set_title("站点分布与调度流向图")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        ax.grid(alpha=0.3)
        ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
        ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
    
    def _plot_demand_satisfaction_distribution(self, ax, inventory_df: pd.DataFrame):
        """绘制需求满足率分布图"""
        # 计算需求满足率
        inventory_df["demand_satisfaction_rate"] = (inventory_df["final_inventory"] / inventory_df["total_demand"]).clip(0, 1.0)
        
        # 设置直方图参数
        bins = np.linspace(0.0, 1.2, 13)
        
        # 绘制直方图
        ax.hist(inventory_df["demand_satisfaction_rate"], bins=bins, color="#96CEB4", edgecolor="white", alpha=0.8)
        
        # 添加目标服务水平参考线
        ax.axvline(0.9, color="#FF6B6B", linestyle="--", linewidth=2, label=f"目标90%")
        
        # 设置坐标轴和标题
        ax.set_xlabel("需求满足率")
        ax.set_ylabel("站点数量")
        ax.set_title("站点需求满足率分布")
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_cost_composition(self, ax, scheduling_df: pd.DataFrame, total_cost: float):
        """绘制成本构成饼图"""
        # 计算调度成本和惩罚成本
        total_sched_cost = scheduling_df["cost"].sum() if len(scheduling_df) > 0 else 0
        total_penalty = max(total_cost - total_sched_cost, 0)  # 确保惩罚成本是非负的
        
        # 设置标签和颜色
        labels = ["调度成本", "惩罚成本"]
        sizes = [total_sched_cost, total_penalty]
        colors = ["#FFEAA7", "#FF6B6B"]
        
        # 绘制饼图
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={"fontsize": 9})
            ax.set_title(f"总成本构成（总计：{sum(sizes):.2f}元）")

