#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：共享单车调度优化主程序
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SCHEDULING_CONFIG, RESULTS_CONFIG, VISUALIZATION_CONFIG
from data_loader import DataLoader
from inventory_calculator import InventoryCalculator
from scheduling_optimizer import SchedulingOptimizer
from visualizer import SchedulingVisualizer
from report_generator import ReportGenerator


def main():
    """主函数"""
    print("=" * 60)
    print("共享单车调度优化系统")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        print("\n【步骤1】加载数据")
        print("-" * 60)
        loader = DataLoader()
        data = loader.load_all()
        
        # 2. 计算初始库存
        print("\n【步骤2】计算初始库存")
        print("-" * 60)
        inventory_calc = InventoryCalculator(SCHEDULING_CONFIG)
        inventory_df = inventory_calc.calculate_initial_inventory(
        data['all_clusters'],  # 使用所有1810个聚类站点
        data['top30_count'],
        data['processed_data'],
        data['top30_clusters']
    )
        
        # 保存初始库存文件
        inventory_output = RESULTS_CONFIG['STATION_INVENTORY']
        inventory_output.parent.mkdir(parents=True, exist_ok=True)
        inventory_df.to_csv(str(inventory_output), index=False, encoding='utf-8-sig')
        print(f"\n✓ 初始库存已保存至: {inventory_output}")
        
        # 3. 构建并求解优化模型
        print("\n【步骤3】构建并求解调度优化模型")
        print("-" * 60)
        optimizer = SchedulingOptimizer(SCHEDULING_CONFIG)
        solution = optimizer.build_optimization_model(
            inventory_df,
            data['station_demand_gap'],
            data['top30_clusters']
        )
        
        if solution.get('status') != 'Optimal':
            print("\n✗ 优化失败，无法生成调度方案")
            print(f"求解状态: {solution.get('status')}")
            return
        
        # 4. 生成调度方案
        print("\n【步骤4】生成调度方案")
        print("-" * 60)
        scheduling_df = optimizer.generate_scheduling_plan(solution, inventory_df)
        
        if scheduling_df.empty:
            print("✗ 调度方案为空")
            return
        
        # 保存调度方案
        scheduling_output = RESULTS_CONFIG['SCHEDULING_PLAN']
        scheduling_output.parent.mkdir(parents=True, exist_ok=True)
        scheduling_df.to_csv(str(scheduling_output), index=False, encoding='utf-8-sig')
        print(f"✓ 调度方案已保存至: {scheduling_output}")
        
        # 5. 生成可视化
        print("\n【步骤5】生成可视化图表")
        print("-" * 60)
        visualizer = SchedulingVisualizer(VISUALIZATION_CONFIG)
        visualizer.visualize_scheduling_plan(
            scheduling_df,
            inventory_df,
            data['top30_clusters'],
            solution['total_cost']
        )
        
        # 6. 生成分析报告
        print("\n【步骤6】生成分析报告")
        print("-" * 60)
        report_gen = ReportGenerator(SCHEDULING_CONFIG)
        report_gen.generate_report(
            solution,
            scheduling_df,
            inventory_df,
            data['station_demand_gap']
        )
        
        # 7. 输出总结
        print("\n" + "=" * 60)
        print("调度优化完成！")
        print("=" * 60)
        print(f"\n输出文件:")
        print(f"  1. 初始库存: {RESULTS_CONFIG['STATION_INVENTORY']}")
        print(f"  2. 调度方案: {RESULTS_CONFIG['SCHEDULING_PLAN']}")
        print(f"  3. 可视化图表: {RESULTS_CONFIG['VISUALIZATION']}")
        print(f"  4. 分析报告: {RESULTS_CONFIG['SCHEDULING_REPORT']}")
        print(f"\n调度统计:")
        print(f"  - 总调度成本: {solution.get('total_cost', 0):.2f} 元")
        print(f"  - 总调度单车数: {int(solution.get('total_bikes_scheduled', 0))} 辆")
        print(f"  - 调度路径数: {len(scheduling_df)} 条")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n✗ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

