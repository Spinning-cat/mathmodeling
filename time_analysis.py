import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATA_CONFIG, RESULTS_CONFIG, TIME_ANALYSIS_CONFIG
from utils import log_info, ensure_dir_exists, load_csv_file, setup_matplotlib


class TimeAnalyzer:
    """
    时段分析类，用于分析共享单车的使用时间模式
    """
    
    def __init__(self):
        """
        初始化TimeAnalyzer实例
        """
        self.config = {
            'DATA': DATA_CONFIG,
            'RESULTS': RESULTS_CONFIG,
            'ANALYSIS': TIME_ANALYSIS_CONFIG
        }
        self.data = None
        
        # 用户提供的颜色组
        self.first_color_group = [(1,121,135), (43,156,161), (123,178,176), (164,198,187), (219,233,227), 
                                (255,247,238), (242,218,190), (199,163,126), (125,103,75), (51,45,21)]
        self.second_color_group = [(58,13,96), (101,68,150), (151,141,190), (198,196,223), (235,236,242), 
                                 (252,230,201), (253,193,117), (228,137,30), (181,90,6), (127,59,8)]
        # 转换为RGB格式
        self.first_color_group_rgb = [(r/255, g/255, b/255) for r, g, b in self.first_color_group]
        self.second_color_group_rgb = [(r/255, g/255, b/255) for r, g, b in self.second_color_group]
        
        # 设置中文字体
        setup_matplotlib()
        # 设置非交互式后端
        plt.switch_backend('Agg')
    
    def load_data(self) -> pd.DataFrame:
        """
        加载处理后的数据
        :return: DataFrame
        """
        log_info("正在加载处理后的数据...")
        data_path = self.config['DATA']['PROCESSED_DATA_PATH']
        self.data = load_csv_file(data_path)
        
        # 转换时间格式
        self.data['start_time'] = pd.to_datetime(self.data['start_time'])
        self.data['end_time'] = pd.to_datetime(self.data['end_time'])
        
        log_info(f"数据导入完成，共{len(self.data)}条记录")
        return self.data
    
    def analyze_hourly_pattern(self) -> tuple:
        """
        分析小时级别的使用频率分布
        :return: (出发小时统计, 到达小时统计)
        """
        log_info("开始分析小时级别的使用频率分布...")
        
        # 按小时统计出发和到达的使用频率
        hourly_start = self.data.groupby('start_hour').size()
        hourly_end = self.data.groupby('end_hour').size()
        
        # 可视化小时分布
        plt.figure(figsize=(12, 6))
        
        # 绘制出发频率
        plt.subplot(1, 2, 1)
        # 使用用户提供的新颜色创建渐变
        from matplotlib.colors import LinearSegmentedColormap
        
        # 用户提供的新颜色（RGB值）
        new_colors = [(122,1,1), (190,20,32), (251,240,217), (1,47,72), (102,154,186)]
        # 转换为RGB格式
        new_colors_rgb = [(r/255, g/255, b/255) for r, g, b in new_colors]
        
        # 创建渐变颜色映射
        hourly_cmap = LinearSegmentedColormap.from_list('hourly_cmap', new_colors_rgb)
        
        # 为每个小时创建一个渐变颜色
        colors = [hourly_cmap(i/23) for i in range(24)]
        
        sns.barplot(x=hourly_start.index, y=hourly_start.values, palette=colors)
        plt.title('出发时间分布（小时）')
        plt.xlabel('小时')
        plt.ylabel('使用次数')
        plt.xticks(range(0, 24))
        
        # 绘制到达频率
        plt.subplot(1, 2, 2)
        sns.barplot(x=hourly_end.index, y=hourly_end.values, palette=colors)
        plt.title('到达时间分布（小时）')
        plt.xlabel('小时')
        plt.ylabel('使用次数')
        plt.xticks(range(0, 24))
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.config['RESULTS']['RESULTS_DIR'], 'hourly_pattern.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        log_info(f"小时使用模式图已保存至: {output_path}")
        
        log_info(f"小时级别使用频率分析完成，结果已保存到 {output_path}")
        return hourly_start, hourly_end
    
    def analyze_weekday_weekend(self) -> tuple:
        """
        分析工作日和非工作日的使用差异
        :return: (工作日出发统计, 周末出发统计)
        """
        log_info("开始分析工作日和非工作日的使用差异...")
        
        # 工作日和非工作日的出发频率
        weekday_start = self.data[self.data['is_workday_start'] == 1].groupby('start_hour').size()
        weekend_start = self.data[self.data['is_workday_start'] == 0].groupby('start_hour').size()
        
        # 可视化工作日和非工作日的差异
        plt.figure(figsize=(12, 6))
        
        # 绘制出发频率
        plt.subplot(1, 2, 1)
        plt.plot(weekday_start.index, weekday_start.values, label='工作日', marker='o', linewidth=2)
        plt.plot(weekend_start.index, weekend_start.values, label='周末', marker='s', linewidth=2)
        plt.title('工作日vs周末 出发时间分布')
        plt.xlabel('小时')
        plt.ylabel('使用次数')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算各时段的使用量
        time_periods = ['平峰', '早高峰', '晚高峰']
        weekday_counts = []
        weekend_counts = []
        
        for period in time_periods:
            if period == '平峰':
                weekday_count = self.data[(self.data['is_workday_start'] == 1) & 
                                         ((self.data['start_hour'] < 7) | ((self.data['start_hour'] >= 9) & (self.data['start_hour'] < 17)) | (self.data['start_hour'] >= 19))].shape[0]
                weekend_count = self.data[(self.data['is_workday_start'] == 0) & 
                                         ((self.data['start_hour'] < 7) | ((self.data['start_hour'] >= 9) & (self.data['start_hour'] < 17)) | (self.data['start_hour'] >= 19))].shape[0]
            elif period == '早高峰':
                weekday_count = self.data[(self.data['is_workday_start'] == 1) & 
                                         (self.data['start_hour'] >= 7) & (self.data['start_hour'] < 9)].shape[0]
                weekend_count = self.data[(self.data['is_workday_start'] == 0) & 
                                         (self.data['start_hour'] >= 7) & (self.data['start_hour'] < 9)].shape[0]
            elif period == '晚高峰':
                weekday_count = self.data[(self.data['is_workday_start'] == 1) & 
                                         (self.data['start_hour'] >= 17) & (self.data['start_hour'] < 19)].shape[0]
                weekend_count = self.data[(self.data['is_workday_start'] == 0) & 
                                         (self.data['start_hour'] >= 17) & (self.data['start_hour'] < 19)].shape[0]
            
            weekday_counts.append(weekday_count)
            weekend_counts.append(weekend_count)
        
        # 绘制时段分布
        plt.subplot(1, 2, 2)
        x = np.arange(len(time_periods))
        width = 0.35
        
        # 使用渐变颜色组中的颜色
        plt.bar(x - width/2, weekday_counts, width, label='工作日', color=self.first_color_group_rgb[1])
        plt.bar(x + width/2, weekend_counts, width, label='周末', color=self.first_color_group_rgb[7])
        
        plt.title('工作日vs周末 时段使用分布')
        plt.xlabel('时段')
        plt.ylabel('使用次数')
        plt.xticks(x, time_periods)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(weekday_counts):
            plt.text(i - width/2, v + 500, str(v), ha='center', va='bottom')
        for i, v in enumerate(weekend_counts):
            plt.text(i + width/2, v + 500, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.config['RESULTS']['RESULTS_DIR'], 'weekday_weekend_pattern.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        log_info(f"工作日周末对比图已保存至: {output_path}")
        
        log_info(f"工作日和非工作日使用差异分析完成，结果已保存到 {output_path}")
        return weekday_start, weekend_start
    
    def analyze_daily_trend(self) -> pd.Series:
        """
        分析每日使用趋势
        :return: 每日使用量统计
        """
        log_info("开始分析每日使用趋势...")
        
        # 按日期统计使用量
        daily_counts = self.data.groupby('start_date').size()
        
        # 可视化每日趋势
        plt.figure(figsize=(12, 6))
        # 使用渐变颜色组中的颜色
        plt.plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2, color=self.first_color_group_rgb[0])
        plt.title('每日使用量趋势')
        plt.xlabel('日期')
        plt.ylabel('使用次数')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.config['RESULTS']['RESULTS_DIR'], 'daily_trend.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        log_info(f"每日趋势图已保存至: {output_path}")
        
        log_info(f"每日使用趋势分析完成，结果已保存到 {output_path}")
        return daily_counts
    
    def analyze_ride_duration(self) -> pd.Series:
        """
        分析骑行时长分布
        :return: 骑行时长统计
        """
        log_info("开始分析骑行时长分布...")
        
        # 骑行时长统计
        duration_stats = self.data['ride_duration'].describe()
        log_info("骑行时长统计：")
        log_info(duration_stats.to_string())
        
        # 可视化骑行时长分布
        plt.figure(figsize=(12, 6))
        
        from matplotlib.colors import LinearSegmentedColormap
        
        # 用户提供的第一组渐变颜色（用于直方图）
        hist_colors = [(143,71,109), (196,123,145), (216,146,156), (227,179,187), (239,213,219)]
        # 转换为RGB格式
        hist_colors_rgb = [(r/255, g/255, b/255) for r, g, b in hist_colors]
        
        # 用户提供的第二组渐变颜色（用于箱线图）
        box_colors = [(1,79,153), (60,121,176), (120,164,201), (179,225,228), (238,245,251)]
        # 转换为RGB格式
        box_colors_rgb = [(r/255, g/255, b/255) for r, g, b in box_colors]
        
        # 创建渐变颜色映射
        hist_cmap = LinearSegmentedColormap.from_list('hist_cmap', hist_colors_rgb)
        box_cmap = LinearSegmentedColormap.from_list('box_cmap', box_colors_rgb)
        
        # 直方图
        plt.subplot(1, 2, 1)
        
        # 绘制直方图并获取数据
        ax = sns.histplot(self.data['ride_duration'], bins=50, kde=True)
        
        # 获取直方图的柱子
        bars = ax.patches
        # 创建渐变颜色列表
        gradient_colors = [hist_cmap(i/len(bars)) for i in range(len(bars))]
        
        # 设置每个柱子的颜色
        for i, bar in enumerate(bars):
            bar.set_facecolor(gradient_colors[i])
            bar.set_edgecolor('none')  # 移除边框
        
        # 设置KDE曲线颜色为深蓝色
        kde_line = ax.lines[0]  # 获取KDE曲线
        kde_line.set_color(box_colors_rgb[0])  # 使用深蓝色
        kde_line.set_linewidth(2)
        
        plt.title('骑行时长分布')
        plt.xlabel('骑行时长（分钟）')
        plt.ylabel('频数')
        
        # 箱线图
        plt.subplot(1, 2, 2)
        
        # 绘制箱线图
        ax = sns.boxplot(y=self.data['ride_duration'])
        
        # 设置箱线图的不同部分使用渐变色
        box_parts = ['boxes', 'whiskers', 'caps', 'medians', 'fliers']
        for i, part in enumerate(box_parts):
            if part == 'boxes':
                # 箱体使用渐变色的中间色
                plt.setp(ax.artists, facecolor=box_colors_rgb[2])
                plt.setp(ax.lines, color=box_colors_rgb[0])  # 边框使用深色
            elif part == 'medians':
                # 中位数线使用深色
                plt.setp(ax.lines[i*2:i*2+2], color=box_colors_rgb[0], linewidth=2)
            elif part == 'fliers':
                # 异常值使用深色
                plt.setp(ax.lines[i*2:i*2+2], color=box_colors_rgb[0], markerfacecolor=box_colors_rgb[2])
            else:
                # 其他部分使用渐变色
                for j, line in enumerate(ax.lines[i*2:i*2+2]):
                    line.set_color(box_colors_rgb[1])
        
        plt.title('骑行时长箱线图')
        plt.ylabel('骑行时长（分钟）')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.config['RESULTS']['RESULTS_DIR'], 'duration_distribution.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        log_info(f"骑行时长分布图已保存至: {output_path}")
        
        log_info(f"骑行时长分布分析完成，结果已保存到 {output_path}")
        return duration_stats
    
    def generate_report(self, hourly_start: pd.Series, hourly_end: pd.Series, 
                       duration_stats: pd.Series) -> None:
        """
        生成时段使用频率分析报告
        :param hourly_start: 出发小时统计
        :param hourly_end: 到达小时统计
        :param duration_stats: 骑行时长统计
        """
        log_info("正在生成时段使用频率分析报告...")
        
        report_path = os.path.join(self.config['RESULTS']['RESULTS_DIR'], 'time_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("时段使用频率分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 小时级使用频率分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"出发高峰时段：{hourly_start.idxmax()}时，使用量：{hourly_start.max()}\n")
            f.write(f"到达高峰时段：{hourly_end.idxmax()}时，使用量：{hourly_end.max()}\n")
            f.write(f"使用最少时段：{hourly_start.idxmin()}时，使用量：{hourly_start.min()}\n\n")
            
            f.write("2. 工作日/周末差异分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"工作日平均日使用量：{self.data[self.data['is_workday_start'] == 1].groupby('start_date').size().mean():.0f}\n")
            f.write(f"周末平均日使用量：{self.data[self.data['is_workday_start'] == 0].groupby('start_date').size().mean():.0f}\n\n")
            
            f.write("3. 骑行时长分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"平均骑行时长：{duration_stats['mean']:.2f}分钟\n")
            f.write(f"中位数骑行时长：{duration_stats['50%']:.2f}分钟\n")
            f.write(f"最短骑行时长：{duration_stats['min']:.2f}分钟\n")
            f.write(f"最长骑行时长：{duration_stats['max']:.2f}分钟\n")
        
        log_info(f"分析报告已生成到 {report_path}")
    
    def run_all_analyses(self) -> None:
        """
        运行所有时段分析
        """
        log_info("开始进行时段使用频率分析...")
        
        # 加载数据
        self.load_data()
        
        # 执行各项分析
        hourly_start, hourly_end = self.analyze_hourly_pattern()
        self.analyze_weekday_weekend()
        self.analyze_daily_trend()
        duration_stats = self.analyze_ride_duration()
        
        # 生成分析报告
        self.generate_report(hourly_start, hourly_end, duration_stats)
        
        log_info("时段使用频率分析全部完成！")


def main():
    """
    主函数
    """
    # 创建结果目录
    ensure_dir_exists(RESULTS_CONFIG['RESULTS_DIR'])
    
    # 初始化分析器
    analyzer = TimeAnalyzer()
    
    # 运行所有分析
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()