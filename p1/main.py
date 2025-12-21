import os
import sys
import argparse
from data_preprocessing import DataPreprocessor
from time_analysis import TimeAnalyzer
import spatial_analysis
from predictive_model import DemandPredictor
from config import DATA_CONFIG, RESULTS_CONFIG, TIME_ANALYSIS_CONFIG, SPATIAL_ANALYSIS_CONFIG, PREDICTION_CONFIG
from utils import log_info, ensure_dir_exists


def main():
    """
    主程序入口，整合所有分析和预测模块
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='共享单车需求分析与预测系统')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                      help='1: 标准分析流程, 2: 作业要求流程')
    args = parser.parse_args()
    
    log_info("=== 共享单车需求分析与预测系统 ===")
    
    # 创建结果目录
    ensure_dir_exists(RESULTS_CONFIG['RESULTS_DIR'])
    
    try:
        # 1. 数据预处理
        log_info("\n1. 开始数据预处理...")
        preprocessor = DataPreprocessor()
        preprocessor.preprocess()
        log_info("数据预处理完成！")
        
        if args.mode == 1:
            # 2. 时间模式分析
            log_info("\n2. 开始时间模式分析...")
            time_analyzer = TimeAnalyzer()
            time_analyzer.run_all_analyses()
            log_info("时间模式分析完成！")
            
            # 3. 空间分析
            log_info("\n3. 开始空间分析...")
            # 由于spatial_analysis.py使用函数式设计，直接调用其main函数
            spatial_analysis.main()
            log_info("空间分析完成！")
            
            # 4. 需求预测
            log_info("\n4. 开始需求预测...")
            predictor = DemandPredictor()
            predictor.run()
            log_info("需求预测完成！")
            
        elif args.mode == 2:
            # 执行作业要求的完整流程
            log_info("\n2. 开始执行作业要求流程...")
            predictor = DemandPredictor()
            results = predictor.run_assignment(n=20, future_hours=24)
            log_info("作业要求流程完成！")
            
            # 输出作业结果摘要
            log_info("\n=== 作业结果摘要 ===")
            log_info(f"早高峰主要需求站点已导出: {RESULTS_CONFIG['RESULTS_DIR']}/morning_peak_demand_stations.csv")
            log_info(f"站点需求预测已导出: {RESULTS_CONFIG['RESULTS_DIR']}/station_demand_predictions.csv")
            log_info(f"站点归还预测已导出: {RESULTS_CONFIG['RESULTS_DIR']}/station_return_predictions.csv")
        
        log_info("\n=== 所有分析任务已完成！ ===")
        log_info(f"分析结果已保存至: {RESULTS_CONFIG['RESULTS_DIR']}")
        
    except Exception as e:
        log_info(f"\n程序执行错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()