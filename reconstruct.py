"""
title: 勇攀高峰
author:Yuan Shi
"""
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

# -*- 预期数据准备 -*-
def preprocess_day(df):
    """
    合并 f_code 和 f_market 列，删除原有列，将 stock 列放到最前面，
    计算 f_pct并按 stock 和 f_time 排序。

    参数:
    df:
    - f_code: 股票代码,类型为object。
    - f_market: 市场代码,类型为object。
    - f_date: 交易日期,表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - f_time: 交易分钟,类型为int64。
    - f_preclose: 前收盘价,类型为float64。
    - f_open: 开盘价,类型为float64。
    - f_high: 最高价,类型为float64。
    - f_low: 最低价,类型为float64。
    - f_close: 收盘价,类型为float64。
    - f_volume: 交易量,类型为float64。
    - f_amount: 交易金额,类型为float64。

    """
    # 合并 f_code 和 f_market 列 
    df['stock'] = df['f_code'] + '.' + df['f_market']
    
    # 删除 f_code 和 f_market 列
    df.drop(['f_code', 'f_market'], axis=1, inplace=True)
    
    # 将 stock 列放到最前面
    columns = ['stock'] + [col for col in df if col != 'stock']
    df = df[columns]
    
    # 计算 f_pct
    df['f_pct'] = df['f_close'] / df['f_preclose'] - 1
    
    # 按 stock 和 f_time 排序
    df.sort_values(by=['stock', 'f_time'], inplace=True)
    
    return df

# -*- 灾后重建因子 -*-
def calculate_reconstruction(day):
    """
    参数:
    df:
    - stock: 股票代码,类型为object。
    - f_date: 交易日期,表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - f_time: 交易分钟,类型为int64。
    - f_preclose: 前收盘价,类型为float64。
    - f_open: 开盘价,类型为float64。
    - f_high: 最高价,类型为float64。
    - f_low: 最低价,类型为float64。
    - f_close: 收盘价,类型为float64。
    - f_volume: 交易量,类型为float64。
    - f_amount: 交易金额,类型为float64。
    - f_pct: 每分钟收益率,类型为float64。

    """    
    # 1. 剔除 f_time=929, 930, 1500 的数据
    day = day[~day['f_time'].isin([929, 930, 1500])].copy()

    # 2. 计算每分钟的“更优波动率”
    def better_volatility_pandas(data):
        std = np.std(data)
        mean = np.mean(data)
        return (std / mean) ** 2

    better_volatility_using_pandas = (
        day.groupby('stock')[['f_open', 'f_high', 'f_low', 'f_close']]
        .rolling(5, method="table")
        .apply(better_volatility_pandas, raw=True, engine="numba")
    )
    better_volatility_using_pandas = better_volatility_using_pandas.reset_index(level=0, drop=True).iloc[:, 0].values
    day['better_volatility'] = better_volatility_using_pandas

    # 3. 计算收益波动比，注意 NaN 值
    day['return_volatility_ratio'] = np.where(day['better_volatility'] != 0, day['f_pct'] / day['better_volatility'], np.nan)

    # 4. 计算协方差，并将列名改为 "reconstruction"
    def calculate_reconstruction(group):
        valid_data = group.dropna(subset=['return_volatility_ratio', 'better_volatility'])
        if len(valid_data) < 2:
            return np.nan
        return valid_data['return_volatility_ratio'].cov(valid_data['better_volatility'])

    reconstructions = day.groupby(['stock', 'f_date']).apply(calculate_reconstruction).reset_index()
    reconstructions.columns = ['stock', 'f_date', 'reconstruction']

    return reconstructions

# -*- 勇攀高峰因子 -*-
def calculate_peak_factor(day):
    """
    参数:
    df:
    - stock: 股票代码,类型为object。
    - f_date: 交易日期,表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - f_time: 交易分钟,类型为int64。
    - f_preclose: 前收盘价,类型为float64。
    - f_open: 开盘价,类型为float64。
    - f_high: 最高价,类型为float64。
    - f_low: 最低价,类型为float64。
    - f_close: 收盘价,类型为float64。
    - f_volume: 交易量,类型为float64。
    - f_amount: 交易金额,类型为float64。
    - f_pct: 每分钟收益率,类型为float64。

    """    
    # 1. 剔除 f_time=929, 930, 1500 的数据
    day = day[~day['f_time'].isin([929, 930, 1500])].copy()

    # 2. 计算每分钟的“更优波动率”
    def better_volatility_pandas(data):
        std = np.std(data)
        mean = np.mean(data)
        return (std / mean) ** 2

    better_volatility_using_pandas = (
        day.groupby('stock')[['f_open', 'f_high', 'f_low', 'f_close']]
        .rolling(5, method="table")
        .apply(better_volatility_pandas, raw=True, engine="numba")
    )
    better_volatility_using_pandas = better_volatility_using_pandas.reset_index(level=0, drop=True).iloc[:, 0].values
    day['better_volatility'] = better_volatility_using_pandas

    # 3. 计算收益波动比，注意 NaN 值
    day['return_volatility_ratio'] = np.where(day['better_volatility'] != 0, day['f_pct'] / day['better_volatility'], np.nan)

    # 4. 计算每个股票当日“更优波动率”的均值 mean 和标准差 std
    daily_stats = day.groupby(['stock', 'f_date'])['better_volatility'].agg(['mean', 'std']).reset_index()
    daily_stats.columns = ['stock', 'f_date', 'mean_volatility', 'std_volatility']

    # 合并均值和标准差到 day
    day = day.merge(daily_stats, on=['stock', 'f_date'], how='left')

    # 找到当日所有“更优波动率”大于等于mean+std 的部分
    day['high_volatility'] = day['better_volatility'] >= (day['mean_volatility'] + day['std_volatility'])

    # 计算高波动时段的收益波动比与“更优波动率”的协方差
    def calculate_high_volatility_cov(group):
        high_volatility_data = group[group['high_volatility']]
        if len(high_volatility_data) < 2:
            return np.nan
        return high_volatility_data['return_volatility_ratio'].cov(high_volatility_data['better_volatility'])

    reconstructions = day.groupby(['stock', 'f_date']).apply(calculate_high_volatility_cov).reset_index()
    reconstructions.columns = ['stock', 'f_date', 'peak']

    return reconstructions

# -*- 降低为月频(均值和标准差) -*-
def calculate_rolling_mean_std(final_correlation_df, column, periods=20):
    """
    参数:
    final_correlation_df:(每日各个股票所对应因子值):
    - stock: 股票代码: 类型为object。
    - f_date: 交易日期,表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - column列:因子名  
    """   
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
    
    
    # 获取每个月的最后一个交易日
    final_correlation_df['month'] = final_correlation_df['f_date'].dt.to_period('M')
    last_dates = final_correlation_df.groupby('month')['f_date'].max().reset_index()['f_date']

    # 筛选出每个月的最后一个交易日的滚动均值和滚动标准差
    result = final_correlation_df[final_correlation_df['f_date'].isin(last_dates)]

    return result[['stock', 'f_date', column + '_mean', column + '_std']]

