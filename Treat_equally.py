import pandas as pd
import time
import datetime
import numpy as np
import warnings
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.tseries.offsets import DateOffset
from scipy.stats import zscore
from scipy.stats import boxcox
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# -*- “正态激增时刻”和“正态骤降时刻” -*-
def process_stock_volume(day):
    # 剔除f_volume为0的时刻
    day = day[day['f_volume'] > 0].copy()
    day = day[~day['f_time'].isin([929, 930, 1500])]

    # 重新排序
    day = day.sort_values(by=['stock', 'f_date', 'f_time']).reset_index(drop=True)

    # 定义Box-Cox变换和计算函数
    def boxcox_transform(group):
        group = group.sort_values(by=['f_date', 'f_time']).reset_index(drop=True)
        if group['f_volume'].nunique() == 1:
            group['f_volume_boxcox'] = group['f_volume']
        else:
            group['f_volume_boxcox'], lam = boxcox(group['f_volume'])
        group['volume_change'] = group['f_volume_boxcox'].diff()
        mean_change = group['volume_change'].mean()
        std_change = group['volume_change'].std()
        group['positive_spike'] = group['volume_change'] > (mean_change + std_change)
        group['negative_drop'] = group['volume_change'] < (mean_change - std_change)
        return group

    # 使用tqdm显示进度条
    tqdm.pandas(desc="Processing stocks")

    # 对每个股票分别进行 Box-Cox 变换和计算
    day = day.groupby('stock').progress_apply(boxcox_transform).reset_index(drop=True)

    return day

# -*- 波动公平因子 -*-
def calculate_volatility_and_return(day, time_df):
    # 只保留time_df中的stock, f_time, positive_spike, negative_drop列
    time_df = time_df[['stock', 'f_time', 'positive_spike', 'negative_drop']]

    # 将day和time_df合并，day为左表，以stock和f_time为键
    day = pd.merge(day, time_df, on=['stock', 'f_time'], how='left')

    # 处理NaN值，将NaN值填充为False
    day['positive_spike'].fillna(False, inplace=True)
    day['negative_drop'].fillna(False, inplace=True)

    # 计算每一个“正态耀眼五分钟”和“正态黯淡五分钟”的区间波动率
    bright_volatilities = []
    dark_volatilities = []

    # 获取每个is_bright和is_dark的起始和结束索引
    bright_intervals = day[day['positive_spike'] == True].index
    dark_intervals = day[day['negative_drop'] == True].index

    for idx in tqdm(bright_intervals, desc="Calculating bright volatilities"):
        end_idx = idx + 4
        if end_idx in day.index and day.loc[idx:end_idx, 'stock'].nunique() == 1:
            interval = day.loc[idx:end_idx, 'f_pct']
            bright_volatilities.append({
                'stock': day.at[idx, 'stock'],
                'f_date': day.at[idx, 'f_date'],
                'bright_volatility': interval.std()
            })

    for idx in tqdm(dark_intervals, desc="Calculating dark volatilities"):
        end_idx = idx + 4
        if end_idx in day.index and day.loc[idx:end_idx, 'stock'].nunique() == 1:
            interval = day.loc[idx:end_idx, 'f_pct']
            dark_volatilities.append({
                'stock': day.at[idx, 'stock'],
                'f_date': day.at[idx, 'f_date'],
                'dark_volatility': interval.std()
            })

    bright_volatility_df = pd.DataFrame(bright_volatilities)
    dark_volatility_df = pd.DataFrame(dark_volatilities)

    # 计算每个股票的“正态耀眼波动率”和“正态黯淡波动率”的均值
    mean_bright_volatility = bright_volatility_df.groupby('stock')['bright_volatility'].mean().reset_index(name='mean_bright_volatility')
    mean_dark_volatility = dark_volatility_df.groupby('stock')['dark_volatility'].mean().reset_index(name='mean_dark_volatility')

    # 将均值合并回原数据框
    day = day.merge(mean_bright_volatility, on='stock', how='left')
    day = day.merge(mean_dark_volatility, on='stock', how='left')

    # 计算“波动公平度”
    day['volatility_fairness'] = np.abs(day['mean_bright_volatility'] - day['mean_dark_volatility'])

    # 计算“波动公平收益率”
    day['volatility_fair_return'] = day['daily_return'] * day['volatility_fairness']

    # 只保留每个stock和f_date的最后一行
    result = day.groupby(['stock', 'f_date']).last().reset_index()
    result = result[['stock', 'f_date', 'volatility_fair_return']]

    return result

# -*- 收益公平因子 -*-
def calculate_return_fair_profit(day,time_df):
    time_df = time_df[['stock', 'f_time', 'positive_spike', 'negative_drop']]

    # 将day和time_df合并，day为左表，以stock和f_time为键
    day = pd.merge(day, time_df, on=['stock', 'f_time'], how='left')
        # 处理NaN值，将NaN值填充为False
    day['positive_spike'].fillna(False, inplace=True)
    day['negative_drop'].fillna(False, inplace=True)
    
    # 1. 创建两个新列，一个为spike_return，一个为drop_return
    day['spike_return'] = np.where(day['positive_spike'], day['f_pct'], 0)
    day['drop_return'] = np.where(day['negative_drop'], day['f_pct'], 0)

    # 2. 对每一个stock分别计算当天所有非零spike_return和非零drop_return的均值，并计算“收益公平度”
    mean_spike_return = day[day['spike_return'] != 0].groupby(['stock', 'f_date'])['spike_return'].mean().reset_index(name='mean_spike_return')
    mean_drop_return = day[day['drop_return'] != 0].groupby(['stock', 'f_date'])['drop_return'].mean().reset_index(name='mean_drop_return')

    # 合并均值数据
    day = day.merge(mean_spike_return, on=['stock', 'f_date'], how='left')
    day = day.merge(mean_drop_return, on=['stock', 'f_date'], how='left')

    # 计算“收益公平度”
    day['return_fairness'] = np.abs(day['mean_spike_return'] - day['mean_drop_return'])

    # 3. 将daily_return与当日的“收益公平度”相乘，得到“收益公平收益率”
    day['return_fair_profit'] = day['daily_return'] * day['return_fairness']

    # 只保留每个stock和f_date的最后一行
    result = day.groupby(['stock', 'f_date']).last().reset_index()
    result = result[['stock', 'f_date', 'return_fair_profit']]

    return result

# -*- 降频为月频 -*-
def calculate_rolling_mean(final_correlation_df, columns, periods=20):
    final_correlation_df = final_correlation_df.copy()
    # 确保f_date列为字符串格式
    final_correlation_df['f_date'] = final_correlation_df['f_date'].astype(str)
    
    # 将f_date列转换为日期格式
    final_correlation_df['f_date'] = pd.to_datetime(final_correlation_df['f_date'], format='%Y%m%d')

    # 按stock和f_date排序
    final_correlation_df = final_correlation_df.sort_values(by=['stock', 'f_date'])
    
    # 计算滚动均值
    rolling_means = final_correlation_df.groupby('stock').apply(lambda x: x[columns].rolling(window=periods, min_periods=1).mean())
    
    # 将滚动均值合并回原数据框
    rolling_means = rolling_means.reset_index(level=0, drop=True)
    final_correlation_df[columns] = rolling_means

    # 获取每个月的最后一个交易日
    final_correlation_df['month'] = final_correlation_df['f_date'].dt.to_period('M')
    last_dates = final_correlation_df.groupby('month')['f_date'].max().reset_index()['f_date']

    # 筛选出每个月的最后一个交易日的滚动均值
    result = final_correlation_df[final_correlation_df['f_date'].isin(last_dates)]

    return result

# -*- 合成一视同仁因子 -*-
def calculate_treat_equal(vol, ret, volatility_col, return_col):
    # 合并 vol 和 ret 数据框，依据 'stock' 和 'f_date' 列进行内连接
    grouped = pd.merge(vol, ret, on=['stock', 'f_date'], how='inner')

    # 只保留需要的列
    grouped = grouped[['stock', 'f_date', volatility_col, return_col]]

    # 计算 moderate_risk 列
    grouped['treat_equal'] = (grouped[volatility_col] + grouped[return_col]) / 2

    return grouped