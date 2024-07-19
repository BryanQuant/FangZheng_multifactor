"""
title: 飞蛾扑火
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
from scipy.stats import boxcox
import statsmodels.api as sm
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

# -*- 日跳跃因子 -*-
def calculate_daily_jump_factor(day):
    """
    参数:
    day:
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
    
    # 剔除开盘和收盘部分的数据
    day = day[~day['f_time'].isin([929, 930, 1500])].copy()
    
    # 计算每分钟的“连续复利收益率”
    day['log_return'] = np.log(day['f_close'] / day['f_preclose'])
    
    # 计算每分钟的“单利收益率”与“连续复利收益率”的差值
    day['simple_return'] = day['f_pct'] 
    day['return_diff'] = day['simple_return'] - day['log_return']
    
    # 计算“泰勒残项”
    day['taylor_residual'] = 2 * day['return_diff'] - day['log_return'] ** 2
    
    # 计算“日跳跃度”因子
    day_jump_factor = day.groupby('stock')['taylor_residual'].mean().reset_index()
    day_jump_factor.columns = ['stock', 'daily_jump_factor']
    
    # 删除多余行
    day_jump_factor['f_date'] = day['f_date'].iloc[0]
    
    return day_jump_factor

# -*- 修正振幅因子1 -*-
def reversed_amplitude_1(stock_df, final_return_equal_df):
    """
    参数:
    stock_df: 清洗后的券池dataframe
    final_return_equal_df: 每个股票的每日“日跳跃度”因子的日数据表
    """   
    # 确保 f_date 列为日期格式
    stock_df['f_date'] = pd.to_datetime(stock_df['f_date'], format='%Y%m%d')
    final_return_equal_df['f_date'] = pd.to_datetime(final_return_equal_df['f_date'], format='%Y%m%d')
    
    # 1. 将两表依据 stock 和 f_date 进行 inner 合并
    merged_df = pd.merge(stock_df, final_return_equal_df, on=['stock', 'f_date'], how='inner')
    
    # 2. 计算振幅
    merged_df['amplitude'] = (merged_df['S_DQ_ADJHIGH'] - merged_df['S_DQ_ADJLOW']) / merged_df['S_DQ_ADJCLOSE'].shift(1)
    
    # 3. 计算日跳跃度因子的截面均值
    daily_jump_factor_mean = merged_df.groupby('f_date')['daily_jump_factor'].transform('mean')
    
    # 4. 计算翻转振幅因子
    merged_df['reversed_amplitude'] = np.where(
        merged_df['daily_jump_factor'] < daily_jump_factor_mean,
        -merged_df['amplitude'],
        merged_df['amplitude']
    )
    
    return merged_df

# -*- 修正振幅因子2 -*-
def reversed_amplitude_2(stock_df):
    """
    参数:
    stock_df: 清洗后的券池dataframe
    final_return_equal_df: 每个股票的每日“日跳跃度”因子的日数据表
    """       
    # 计算单利收益率和连续复利收益率
    stock_df['simple_return'] = (stock_df['S_DQ_ADJHIGH'] - stock_df['S_DQ_ADJLOW'].shift(1)) / stock_df['S_DQ_ADJLOW'].shift(1)
    stock_df['log_return'] = np.log(stock_df['S_DQ_ADJHIGH'] / stock_df['S_DQ_ADJLOW'].shift(1))
    
    # 计算单复利差
    stock_df['return_diff'] = stock_df['simple_return'] - stock_df['log_return']
    
    # 计算泰勒残项
    stock_df['taylor_residual'] = 2 * stock_df['return_diff'] - stock_df['log_return'] ** 2
    
    # 计算每日振幅
    stock_df['amplitude'] = (stock_df['S_DQ_ADJHIGH'] - stock_df['S_DQ_ADJLOW']) / stock_df['S_DQ_ADJCLOSE'].shift(1)
    
    # 计算泰勒残项的截面均值
    taylor_residual_mean = stock_df.groupby('f_date')['taylor_residual'].transform('mean')
    
    # 计算翻转振幅2
    stock_df['reversed_amplitude2'] = np.where(stock_df['taylor_residual'] < taylor_residual_mean,
                                               -stock_df['amplitude'],
                                               stock_df['amplitude'])
    
    return stock_df

# 最终修正振幅因子是由修正振幅因子1和修正振幅因子2取均值

# -*- 降低为月频 (均值和标准差)-*-
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

# -*- 降频为月频 (只有均值) -*-
def calculate_rolling_mean(final_correlation_df, columns, periods=20):
    """
    参数:
    final_correlation_df:(每日各个股票所对应因子值):
    - stock: 股票代码: 类型为object。
    - f_date: 交易日期,表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - column列:因子名  
    """   
    final_correlation_df = final_correlation_df.copy()

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