#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
负责共享单车数据的清洗、转换和特征提取
"""

import pandas as pd
from typing import Tuple

from config import (DATA_CONFIG, RESULTS_CONFIG, PREPROCESSING_CONFIG)
from utils import (load_csv_file, save_csv_file, ensure_dir_exists, log_info)


class DataPreprocessor:
    """数据预处理类，封装所有数据预处理功能"""
    
    def __init__(self):
        """初始化数据预处理器"""
        self.raw_data_path = DATA_CONFIG['RAW_DATA_PATH']
        self.processed_data_path = DATA_CONFIG['PROCESSED_DATA_PATH']
        
        # 预处理配置
        self.longitude_range = PREPROCESSING_CONFIG['LONGITUDE_RANGE']
        self.latitude_range = PREPROCESSING_CONFIG['LATITUDE_RANGE']
        self.duration_min = PREPROCESSING_CONFIG['DURATION_MIN']
        self.duration_max = PREPROCESSING_CONFIG['DURATION_MAX']
        
    def convert_time_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """将时间字符串转换为datetime格式
        
        Args:
            df: 原始数据框
        
        Returns:
            转换时间格式后的数据框
        """
        log_info("转换时间格式...")
        df['start_time'] = pd.to_datetime(df['start_time'], format='%Y/%m/%d %H:%M')
        df['end_time'] = pd.to_datetime(df['end_time'], format='%Y/%m/%d %H:%M')
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间相关特征
        
        Args:
            df: 数据框
        
        Returns:
            添加时间特征后的数据框
        """
        log_info("提取时间特征...")
        
        # 提取小时
        df['start_hour'] = df['start_time'].dt.hour
        df['end_hour'] = df['end_time'].dt.hour
        
        # 提取星期几（0=周一，6=周日）
        df['start_weekday'] = df['start_time'].dt.weekday
        df['end_weekday'] = df['end_time'].dt.weekday
        
        # 提取日期
        df['start_date'] = df['start_time'].dt.date
        df['end_date'] = df['end_time'].dt.date
        
        # 判断是否为工作日（周一至周五为工作日）
        df['is_workday_start'] = (df['start_weekday'] < 5).astype(int)
        df['is_workday_end'] = (df['end_weekday'] < 5).astype(int)
        
        # 判断时间段（早高峰、晚高峰、平峰）
        morning_start, morning_end = PREPROCESSING_CONFIG['TIME_PERIODS']['MORNING_PEAK']
        evening_start, evening_end = PREPROCESSING_CONFIG['TIME_PERIODS']['EVENING_PEAK']
        
        def get_time_period(hour: int) -> int:
            if morning_start <= hour < morning_end:
                return 1  # 早高峰
            elif evening_start <= hour < evening_end:
                return 2  # 晚高峰
            else:
                return 0  # 平峰
        
        df['time_period_start'] = df['start_hour'].apply(get_time_period)
        df['time_period_end'] = df['end_hour'].apply(get_time_period)
        
        return df
    
    def calculate_ride_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算骑行时长并过滤异常值
        
        Args:
            df: 数据框
        
        Returns:
            过滤后的数据框
        """
        log_info("计算骑行时长并过滤异常值...")
        
        df['ride_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        
        # 过滤异常值
        initial_count = len(df)
        df = df[(df['ride_duration'] >= self.duration_min) & 
                (df['ride_duration'] <= self.duration_max)]
        
        filtered_count = len(df)
        removed_count = initial_count - filtered_count
        log_info(f"骑行时长过滤: 移除 {removed_count} 条记录 ({initial_count} → {filtered_count})")
        
        return df
    
    def filter_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤位置数据，只保留上海地区范围内的记录
        
        Args:
            df: 数据框
        
        Returns:
            过滤后的数据框
        """
        log_info("过滤位置数据...")
        
        min_lon, max_lon = self.longitude_range
        min_lat, max_lat = self.latitude_range
        
        # 过滤起点坐标
        initial_count = len(df)
        
        df = df[(df['start_location_x'] >= min_lon) & (df['start_location_x'] <= max_lon) &
                (df['start_location_y'] >= min_lat) & (df['start_location_y'] <= max_lat) &
                (df['end_location_x'] >= min_lon) & (df['end_location_x'] <= max_lon) &
                (df['end_location_y'] >= min_lat) & (df['end_location_y'] <= max_lat)]
        
        filtered_count = len(df)
        removed_count = initial_count - filtered_count
        log_info(f"位置过滤: 移除 {removed_count} 条记录 ({initial_count} → {filtered_count})")
        
        return df
    
    def preprocess(self) -> pd.DataFrame:
        """执行完整的数据预处理流程
        
        Returns:
            处理后的数据框
        """
        log_info(f"开始数据预处理，原始数据路径: {self.raw_data_path}")
        
        # 1. 加载原始数据
        df = load_csv_file(self.raw_data_path)
        
        # 2. 转换时间格式
        df = self.convert_time_format(df)
        
        # 3. 提取时间特征
        df = self.extract_time_features(df)
        
        # 4. 计算骑行时长并过滤异常值
        df = self.calculate_ride_duration(df)
        
        # 5. 过滤位置数据
        df = self.filter_location_data(df)
        
        # 6. 保存处理后的数据
        save_csv_file(df, self.processed_data_path, index=False)
        
        # 7. 打印数据基本信息
        log_info("\n数据预处理完成！")
        log_info(f"数据量：{len(df)}条记录")
        log_info(f"时间范围：{df['start_time'].min()} 至 {df['end_time'].max()}")
        log_info(f"平均骑行时长：{df['ride_duration'].mean():.2f}分钟")
        
        return df


def main():
    """数据预处理主函数"""
    # 确保结果目录存在
    ensure_dir_exists(RESULTS_CONFIG['RESULTS_DIR'])
    
    # 创建并运行数据预处理器
    preprocessor = DataPreprocessor()
    preprocessor.preprocess()


if __name__ == "__main__":
    main()