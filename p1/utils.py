#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块：包含项目中使用的公共函数
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Tuple, Any

from config import (PREPROCESSING_CONFIG, VISUALIZATION_CONFIG)


def setup_matplotlib():
    """
    设置Matplotlib环境，包括中文字体和非交互式后端
    """
    plt.rcParams['font.sans-serif'] = VISUALIZATION_CONFIG['FONT_FAMILY']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = VISUALIZATION_CONFIG['FONT_SIZE']
    plt.switch_backend('Agg')  # 设置非交互式后端，避免显示窗口


def ensure_dir_exists(directory: Union[str, Path]) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def load_csv_file(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    加载CSV文件
    
    Args:
        file_path: 文件路径
        **kwargs: pandas.read_csv的其他参数
    
    Returns:
        加载的数据框
    
    Raises:
        FileNotFoundError: 如果文件不存在
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        print(f"成功加载文件: {file_path} (共{len(df)}条记录)")
        return df
    except Exception as e:
        raise RuntimeError(f"加载文件失败: {file_path}，错误信息: {str(e)}")


def save_csv_file(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    保存数据框到CSV文件
    
    Args:
        df: 要保存的数据框
        file_path: 文件路径
        **kwargs: pandas.to_csv的其他参数
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # 确保父目录存在
    ensure_dir_exists(file_path.parent)
    
    try:
        df.to_csv(file_path, **kwargs)
        print(f"数据已保存到: {file_path}")
    except Exception as e:
        raise RuntimeError(f"保存文件失败: {file_path}，错误信息: {str(e)}")


def get_time_period(hour: int) -> int:
    """
    根据小时获取时段类型
    
    Args:
        hour: 小时（0-23）
    
    Returns:
        时段类型：0-平峰，1-早高峰，2-晚高峰
    """
    morning_start, morning_end = PREPROCESSING_CONFIG['TIME_PERIODS']['MORNING_PEAK']
    evening_start, evening_end = PREPROCESSING_CONFIG['TIME_PERIODS']['EVENING_PEAK']
    
    if morning_start <= hour < morning_end:
        return 1  # 早高峰
    elif evening_start <= hour < evening_end:
        return 2  # 晚高峰
    else:
        return 0  # 平峰


def get_time_period_label(hour: int) -> str:
    """
    根据小时获取时段标签
    
    Args:
        hour: 小时（0-23）
    
    Returns:
        时段标签：morning_peak, evening_peak, normal, night
    """
    morning_start, morning_end = PREPROCESSING_CONFIG['TIME_PERIODS']['MORNING_PEAK']
    evening_start, evening_end = PREPROCESSING_CONFIG['TIME_PERIODS']['EVENING_PEAK']
    
    if morning_start <= hour < morning_end:
        return 'morning_peak'
    elif evening_start <= hour < evening_end:
        return 'evening_peak'
    elif 9 <= hour < 17:
        return 'normal'
    else:
        return 'night'


def is_workday(weekday: int) -> int:
    """
    判断是否为工作日
    
    Args:
        weekday: 星期几（0=周一，6=周日）
    
    Returns:
        1-工作日，0-非工作日
    """
    return 1 if weekday < 5 else 0


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点之间的欧几里得距离
    
    Args:
        lat1, lon1: 第一个点的经纬度
        lat2, lon2: 第二个点的经纬度
    
    Returns:
        两点之间的距离
    """
    return ((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2) ** 0.5


def save_figure(fig: plt.Figure, file_path: Union[str, Path], dpi: int = 300, **kwargs) -> None:
    """
    保存Matplotlib图形
    
    Args:
        fig: 要保存的图形对象
        file_path: 保存路径
        dpi: 图像分辨率
        **kwargs: plt.savefig的其他参数
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # 确保父目录存在
    ensure_dir_exists(file_path.parent)
    
    try:
        fig.savefig(file_path, dpi=dpi, **kwargs)
        plt.close(fig)  # 关闭图形，释放内存
        print(f"图形已保存到: {file_path}")
    except Exception as e:
        plt.close(fig)  # 即使保存失败也关闭图形
        raise RuntimeError(f"保存图形失败: {file_path}，错误信息: {str(e)}")


def create_figure(figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    创建Matplotlib图形对象
    
    Args:
        figsize: 图形尺寸
    
    Returns:
        图形对象
    """
    return plt.figure(figsize=figsize)


def log_info(message: str) -> None:
    """
    记录信息日志
    
    Args:
        message: 日志信息
    """
    print(f"[INFO] {message}")


def log_error(message: str) -> None:
    """
    记录错误日志
    
    Args:
        message: 日志信息
    """
    print(f"[ERROR] {message}")
