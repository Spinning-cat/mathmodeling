#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：共享单车调度优化模块
核心功能：基于历史骑行数据计算初始库存，构建调度优化模型，生成次日早高峰调度方案
"""

__version__ = "1.0.1"
__author__ = "调度优化模块"
__description__ = "最小化调度成本，满足主要站点供需平衡（基于某日数据优化次日早高峰）"

# 暴露核心类和函数
from .inventory_calculator import InventoryCalculator
from .scheduling_optimizer import SchedulingOptimizer
from .main import run_scheduling_optimization