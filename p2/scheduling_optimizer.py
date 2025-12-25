#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单车调度优化模型
使用线性规划（Linear Programming）求解最优调度方案
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from data_loader import calculate_distance
import warnings
warnings.filterwarnings('ignore')


class SchedulingOptimizer:
    """单车调度优化器"""
    
    def __init__(self, config):
        """
        初始化调度优化器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.problem = None
        self.solution = None
    
    def build_optimization_model(self, inventory_df: pd.DataFrame,
                                 demand_gap_df: pd.DataFrame,
                                 top30_clusters: pd.DataFrame) -> dict:
        """
        构建调度优化模型

        模型说明：
        =========
        整数规划模型

        决策变量：
        - x[i][j] = 从站点i调度到站点j的单车数量
        - y[i][j] = 路径激活标识（0=未使用，1=使用）
        - unmet[i] = 站点i的未满足需求

        目标函数：最小化总调度成本 + 惩罚成本
        min Σ(i,j) (距离成本 * x[i][j] + 固定成本 * y[i][j]) + Σ(i) (惩罚系数 * unmet[i])

        约束条件：
        1. 需求满足约束：final_inv + unmet >= target_demand
        2. 容量约束：final_inv <= station_capacity
        3. 调出限制：总调出量 <= 最大调出量 && <= 初始库存的80%
        4. 路径激活与调度量绑定：x <= M * y && x >= y
        5. 时间窗口约束：总运输时间 <= 调度窗口

        Args:
            inventory_df: 站点初始库存信息
            demand_gap_df: 主要站点需求缺口
            top30_clusters: 30个主要站点信息

        Returns:
            包含优化结果的字典
        """
        print("\n=== 开始构建调度优化模型 ===\n")

        # 1. 准备数据
        print("1. 准备优化数据...")

        n = len(inventory_df)
        station_ids = inventory_df["station_id"].tolist()
        all_stations = inventory_df.copy()

        print(f"   总站点数: {n}")

        # 2. 生成合法候选边（基于最近邻+最大距离限制）
        print("2. 生成合法候选边（最近邻+最大距离限制）...")

        K = self.config['K_NEIGHBORS']
        MAX_DISTANCE = self.config['MAX_DISTANCE']
        neighbors = {i: [] for i in range(n)}  # 每个站点的合法邻居列表

        # 计算距离矩阵
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    station_i = all_stations.iloc[i]
                    station_j = all_stations.iloc[j]
                    dist = calculate_distance(
                        station_i['center_x'], station_i['center_y'],
                        station_j['center_x'], station_j['center_y']
                    )
                    distance_matrix[i, j] = dist

        # 为每个站点找到合法邻居
        for i in range(n):
            # 计算站点i到所有其他站点的距离（排除自身）
            distances = [(j, distance_matrix[i][j]) for j in range(n) if i != j]
            # 筛选：距离≤MAX_DISTANCE 的站点，再取最近的K个
            valid_neighbors = sorted([(j, d) for j, d in distances if d <= MAX_DISTANCE], key=lambda x: x[1])[:K]
            # 仅保留邻居索引
            neighbors[i] = [j for j, d in valid_neighbors]
            # 打印前5个站点的邻居信息
            if i < 5:
                print(f"   - 站点{station_ids[i]}：合法邻居{len(neighbors[i])}个，最大距离{max([d for j, d in valid_neighbors]) if valid_neighbors else 0:.2f}km")

        # 生成边集（仅合法候选边）
        edges = [(i, j) for i in range(n) for j in neighbors[i] if len(neighbors[i]) > 0]
        # 生成入边字典
        incoming = {i: [] for i in range(n)}
        for i, j in edges:
            incoming[j].append(i)

        print(f"   生成合法候选边 {len(edges)} 条")

        # 3. 创建问题和决策变量
        print("3. 创建决策变量...")

        prob = LpProblem("Bike_Scheduling_Optimization", LpMinimize)

        # 决策变量：x[(i,j)] 表示从站点i到站点j的调度量
        x = {}
        for (i, j) in edges:
            var_name = f"x_{station_ids[i]}_{station_ids[j]}"
            x[(i, j)] = LpVariable(var_name, lowBound=0, upBound=self.config['MAX_BIKES_PER_TRIP'], cat='Integer')

        # 决策变量：y[(i,j)] 表示路径激活标识
        y = {}
        for (i, j) in edges:
            var_name = f"y_{station_ids[i]}_{station_ids[j]}"
            y[(i, j)] = LpVariable(var_name, lowBound=0, upBound=1, cat='Integer')

        # 固定零变量：避免pulp类型错误
        ZERO_CONST = LpVariable(name="ZERO_CONST", lowBound=0, upBound=0, cat='Integer')

        print(f"   决策变量数: {len(x)} 个x变量 + {len(y)} 个y变量")

        # 4. 目标函数
        print("4. 添加目标函数...")

        # 调度成本：距离成本 + 固定成本
        scheduling_cost = lpSum([
            (distance_matrix[i, j] * self.config['DISTANCE_COST_PER_KM'] * x[(i, j)] + 
             self.config['FIXED_COST_PER_TRIP'] * y[(i, j)])
            for (i, j) in edges
        ])

        # 惩罚成本
        unmet = {}
        penalty_cost = 0
        for i in range(n):
            # 计算调出和调入
            out_sum = lpSum([x[(i, j)] for j in neighbors[i]]) if len(neighbors[i]) > 0 else lpSum([])
            in_sum = lpSum([x[(j, i)] for j in incoming[i]]) if len(incoming[i]) > 0 else lpSum([])
            final_inv = all_stations["initial_inventory"].iloc[i] - out_sum + in_sum + 0 * ZERO_CONST
            
            unmet[i] = LpVariable(name=f"unmet_{station_ids[i]}", lowBound=0, cat='Integer')
            
            # 目标需求
            target_demand = all_stations["peak_demand"].iloc[i] * self.config['DEMAND_SATISFACTION_RATIO']
            prob += (final_inv + unmet[i] >= target_demand), f"unmet_demand_{station_ids[i]}"
            
            penalty_cost += self.config['PENALTY_COEFFICIENT'] * unmet[i]

        # 总目标函数
        prob += scheduling_cost + penalty_cost

        # 5. 添加约束条件
        print("5. 添加约束条件...")

        for i in range(n):
            out_sum = lpSum([x[(i, j)] for j in neighbors[i]]) if len(neighbors[i]) > 0 else lpSum([])
            in_sum = lpSum([x[(j, i)] for j in incoming[i]]) if len(incoming[i]) > 0 else lpSum([])
            final_inv = all_stations["initial_inventory"].iloc[i] - out_sum + in_sum + 0 * ZERO_CONST
            station_inv = all_stations["initial_inventory"].iloc[i]

            # 约束1：总需求满足率约束
            target_total_demand = all_stations["total_demand"].iloc[i] * self.config['DEMAND_SATISFACTION_RATIO']
            prob += (final_inv + unmet[i] >= target_total_demand), f"total_demand_{station_ids[i]}"
            
            # 约束2：调度后库存不超过站点容量
            prob += (final_inv <= all_stations["capacity"].iloc[i]), f"capacity_{station_ids[i]}"
            

            
            # 约束3：单点调出量≤物理上限（40辆）+≤库存80%（避免库存耗尽）
            prob += (out_sum <= self.config['MAX_OUTFLOW_PER_STATION']), f"outflow_maxcap_{station_ids[i]}"
            prob += (out_sum <= station_inv * 0.8), f"outflow_inventory_limit_{station_ids[i]}"

        # 约束4：路径激活与调度量绑定
        M = self.config['MAX_BIKES_PER_TRIP']
        for (i, j) in edges:
            prob += (x[(i, j)] <= M * y[(i, j)]), f"x_le_yM_{i}_{j}"  # x>0 → y=1
            prob += (x[(i, j)] >= y[(i, j)]), f"x_ge_y_{i}_{j}"      # y=1 → x≥1

        # 约束5：时间窗口限制（暂时注释，先验证其他约束）
        # print("6. 添加时间窗口约束...")
        # avg_speed = 30  # km/h
        # for i in range(n):
        #     if len(neighbors[i]) == 0:
        #         continue
        #     transport_time = lpSum([
        #         (distance_matrix[i, j] * 2 / avg_speed) * y[(i, j)]  # 往返时间
        #         for j in neighbors[i]
        #     ])
        #     prob += (transport_time <= self.config['SCHEDULE_WINDOW']), f"time_window_{station_ids[i]}"

        self.problem = prob

        print(f"\n   模型构建完成！")
        print(f"   约束数量: {len(prob.constraints)}")

        # 6. 求解模型
        print("\n7. 求解优化模型...")
        print("   这可能需要几分钟时间，请耐心等待...")

        try:
            solver = PULP_CBC_CMD(
                msg=1,
                timeLimit=self.config['SOLVER_TIME_LIMIT'],
                options=["strongBranching 5", "maxSeconds 300", "gapRel 0.05"]
            )
            prob.solve(solver)
            status = LpStatus[prob.status]

            print(f"\n   求解状态: {status}")

            if status == 'Optimal':
                print("   ✓ 找到最优解！")

                # 提取解
                solution_dict = {}
                total_cost = prob.objective.value()
                total_bikes_scheduled = 0

                for (i, j) in edges:
                    y_val = int(y[(i, j)].varValue or 0)
                    if y_val == 0:
                        continue  # 跳过未激活路径
                    x_val = int(x[(i, j)].varValue or 0)
                    if x_val >= 1:
                        # 计算真实成本
                        transport_cost = distance_matrix[i, j] * self.config['DISTANCE_COST_PER_KM'] * x_val
                        fixed_cost = self.config['FIXED_COST_PER_TRIP'] * y_val
                        path_cost = transport_cost + fixed_cost
                        
                        solution_dict[(i, j)] = {
                            'from_station': station_ids[i],
                            'to_station': station_ids[j],
                            'quantity': x_val,
                            'distance': distance_matrix[i, j],
                            'cost': path_cost,
                        }
                        total_bikes_scheduled += x_val

                print(f"\n   调度统计:")
                print(f"   调度路径数: {len(solution_dict)}")
                print(f"   总调度单车数: {total_bikes_scheduled}")
                print(f"   总调度成本: {total_cost:.2f} 元")

                self.solution = {
                    'status': status,
                    'solution_dict': solution_dict,
                    'total_cost': total_cost,
                    'total_bikes_scheduled': total_bikes_scheduled,
                    'problem': prob,
                }

                return self.solution
            else:
                print(f"   ✗ 求解失败，状态: {status}")
                return {'status': status, 'solution_dict': {}, 'total_cost': 0, 'total_bikes_scheduled': 0}

        except Exception as e:
            print(f"   ✗ 求解过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'Error', 'error': str(e)}
    
    def generate_scheduling_plan(self, solution: dict, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成调度方案DataFrame

        Args:
            solution: 优化结果
            inventory_df: 站点库存信息

        Returns:
            调度方案DataFrame
        """
        if solution['status'] != 'Optimal' or not solution.get('solution_dict'):
            return pd.DataFrame()

        scheduling_list = []
        
        # 计算每个站点的调入和调出量
        outflow = {i: 0 for i in range(len(inventory_df))}
        inflow = {i: 0 for i in range(len(inventory_df))}

        for (i, j), info in solution['solution_dict'].items():
            # 使用站点索引而不是cluster_id
            from_station = inventory_df.iloc[i]
            to_station = inventory_df.iloc[j]
            quantity = info['quantity']
            
            # 记录调出和调入量
            outflow[i] += quantity
            inflow[j] += quantity

            scheduling_list.append({
                'from_station': from_station['station_id'],
                'to_station': to_station['station_id'],
                'bike_count': quantity,
                'distance_km': round(info['distance'], 2),
                'cost': round(info['cost'], 2),
            })

        scheduling_df = pd.DataFrame(scheduling_list)

        # 按调度量排序
        scheduling_df = scheduling_df.sort_values('bike_count', ascending=False)
        
        # 更新每个站点的调度后库存和调入调出量
        final_inventory_list = []
        for i in range(len(inventory_df)):
            initial_inv = inventory_df.iloc[i]['initial_inventory']
            final_inv = initial_inv - outflow[i] + inflow[i]
            final_inventory_list.append(final_inv)
        
        inventory_df['final_inventory'] = final_inventory_list
        inventory_df['out_bikes'] = list(outflow.values())
        inventory_df['in_bikes'] = list(inflow.values())
        
        return scheduling_df

