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
from synergy import *
warnings.filterwarnings('ignore')

# -*- 每分钟计算过去5分钟内的20个数据的均值和标准差，判断位置 -*-
def calculate_rolling_features_optimized(df):
    window = 5  # 窗口大小为5分钟
    
    # 计算滚动均值和标准差
    df['combined_mean'] = df[['f_open', 'f_high', 'f_low', 'f_close']].mean(axis=1).rolling(window=window).mean()
    df['combined_std'] = df[['f_open', 'f_high', 'f_low', 'f_close']].mean(axis=1).rolling(window=window).std()

    # 计算上轨和下轨
    df['upper_band'] = df['combined_mean'] + df['combined_std']
    df['lower_band'] = df['combined_mean'] - df['combined_std']

    # 标记 f_close 相对于上下轨的位置
    df['position'] = 0
    df.loc[df['f_close'] > df['upper_band'], 'position'] = 1
    df.loc[df['f_close'] < df['lower_band'], 'position'] = -1

    return df

# -*- 计算每个股票每分钟成交量占比和每个股票对应的协同股票成交量占比 -*-
def calculate_volume_percentage(df):
    # 1. 删除指定的 f_time
    df = df[~df['f_time'].isin([929, 930, 931, 932])]
    
    # 2. 计算每个 f_time 的 f_volume 总和
    df['f_time_total_volume'] = df.groupby('f_time')['f_volume'].transform('sum')
    
    # 3. 计算每个 stock 在该 f_time 的 f_volume 占比
    df['volume_percentage'] = df['f_volume'] / df['f_time_total_volume']
    
    # 4. 计算协同成交量
    # 首先，计算每个 f_time 每个 position 的 volume_percentage 之和
    position_volume_sum = df.groupby(['f_time', 'position'])['volume_percentage'].transform('sum')
    
    # 然后，将这个结果赋给每一行作为协同成交量
    df['synergistic_volume'] = position_volume_sum
    
    return df

# -*- 计算两个成交量占比列的相关系数 -*-
def calculate_correlation(df):
    """
    计算个股日内 f_time 成交量占比序列与“协同成交量”占比序列之间的相关系数，并包含 daily_return 列。
    """
    correlation_df = df.groupby(['f_date', 'stock']).apply(lambda x: pd.Series({
        'correlation': x['volume_percentage'].corr(x['synergistic_volume']),
        'daily_return': x['daily_return'].iloc[0]
    })).reset_index()

    return correlation_df

# -*- 降频为月频 -*-
def calculate_rolling_mean_std(final_correlation_df, column, periods=20):
    # 确保f_date列为字符串格式
    final_correlation_df['f_date'] = final_correlation_df['f_date'].astype(str)
    
    # 将f_date列转换为日期格式
    final_correlation_df['f_date'] = pd.to_datetime(final_correlation_df['f_date'], format='%Y%m%d')

    # 按stock和f_date排序
    final_correlation_df = final_correlation_df.sort_values(by=['stock', 'f_date'])
    
    # 计算滚动均值和滚动标准差
    rolling_means = final_correlation_df.groupby('stock')[column].rolling(window=periods, min_periods=1).mean()
    rolling_stds = final_correlation_df.groupby('stock')[column].rolling(window=periods, min_periods=1).std()
    
    # 将滚动均值和滚动标准差合并回原数据框
    final_correlation_df[column + '_mean'] = rolling_means.reset_index(level=0, drop=True)
    final_correlation_df[column + '_std'] = rolling_stds.reset_index(level=0, drop=True)
    
    # 计算新的因子列
    final_correlation_df[column + '_factor'] = (final_correlation_df[column + '_mean'] + final_correlation_df[column + '_std']) / 2

    # 获取每个月的最后一个交易日
    final_correlation_df['month'] = final_correlation_df['f_date'].dt.to_period('M')
    last_dates = final_correlation_df.groupby('month')['f_date'].max().reset_index()['f_date']

    # 筛选出每个月的最后一个交易日的滚动均值和滚动标准差
    result = final_correlation_df[final_correlation_df['f_date'].isin(last_dates)]

    return result[['stock', 'f_date', column + '_mean', column + '_std', column + '_factor']]