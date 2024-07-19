"""
title: 草木皆兵
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

# -*- 计算惊恐度 -*-
def calculate_fear_index(combined_df, CSI_985):
    """
    参数:
    combined_df: 全A日行情数据(复权后)
    CSI_985: 中证全指指数日行情数据
    
    """
    # 计算股票每日收益率
    combined_df['pct'] = combined_df['S_DQ_ADJCLOSE'] / combined_df['S_DQ_ADJPRECLOSE'] - 1

    # 读取中证全指数据并计算市场收益率
    CSI_985['csi_pct'] = CSI_985['S_DQ_CLOSE'] / CSI_985['S_DQ_PRECLOSE'] - 1

    # 只保留f_date和csi_pct列
    CSI_985 = CSI_985[['TRADE_DT', 'csi_pct']]

    # 合并数据
    merged_combined_df = pd.merge(combined_df, CSI_985, on='TRADE_DT', how='inner')

    # 1. 将列名重命名
    merged_combined_df = merged_combined_df.rename(columns={'S_INFO_WINDCODE': 'stock', 'TRADE_DT': 'f_date'})

    # 2. 计算“偏离项”和“基准项”，并计算“惊恐度”
    merged_combined_df['deviation'] = (merged_combined_df['pct'] - merged_combined_df['csi_pct']).abs()
    merged_combined_df['benchmark'] = merged_combined_df['pct'].abs() + merged_combined_df['csi_pct'].abs() + 0.1
    merged_combined_df['fear_index'] = merged_combined_df['deviation'] / merged_combined_df['benchmark']

    # 3. 计算“衰减后的惊恐度”
    merged_combined_df['prev_1_fear'] = merged_combined_df.groupby('stock')['fear_index'].shift(1)
    merged_combined_df['prev_2_fear'] = merged_combined_df.groupby('stock')['fear_index'].shift(2)
    merged_combined_df['avg_prev_fear'] = (merged_combined_df['prev_1_fear'] + merged_combined_df['prev_2_fear']) / 2
    merged_combined_df['fear_decay'] = merged_combined_df['fear_index'] - merged_combined_df['avg_prev_fear']

    # 将负值替换为NaN--衰减后的惊恐度
    merged_combined_df.loc[merged_combined_df['fear_decay'] < 0, 'fear_decay'] = np.nan

    # 只保留所需列
    merged_combined_df = merged_combined_df[['stock', 'f_date', 'pct', 'fear_index', 'fear_decay']]

    # 将f_date列转换为numpy.int64格式
    merged_combined_df['f_date'] = merged_combined_df['f_date'].astype(np.int64)

    return merged_combined_df

# filter_df = result_df[result_df['f_date'] == int(date)]
# day_test = pd.merge(day_test, filter_df, on=['f_date', 'stock'], how='inner')
# 要利用上面代码将分钟数据与filter_df合并之后有了‘惊恐度’才能进行下面function的调用
# -*- 所有因子计算 -*-
def calculate_weighted_decisions(merged_day):
    """
    参数:
    merged_day:
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
    - fear_index: 惊恐度,类型为float64。
    - fear_decay: 衰减后的惊恐度,类型为float64。
    
    """
    # 1. 计算“原始惊恐”因子
    merged_day['raw_fear'] = merged_day['fear_index'] * merged_day['pct']
    
    # 2. 计算每日波动率
    daily_volatility = merged_day.groupby(['stock', 'f_date'])['f_pct'].std().reset_index()
    daily_volatility = daily_volatility.rename(columns={'f_pct': 'daily_volatility'})
    merged_day = pd.merge(merged_day, daily_volatility, on=['stock', 'f_date'], how='left')
    
    # 3. 计算“波动率加剧”因子
    merged_day['volatility_amplified'] = merged_day['daily_volatility'] * merged_day['fear_index'] * merged_day['pct']
    
    # 4. 计算“注意力衰退”因子
    merged_day['attention_decay'] = merged_day['fear_decay'] * merged_day['pct']
    
    # 5. 计算“草木皆兵”因子
    merged_day['panic_everywhere'] = merged_day['fear_decay'] * merged_day['daily_volatility'] * merged_day['pct']
    
    # 只保留指定的四列
    merged_day = merged_day[['stock', 'f_date', 'raw_fear', 'volatility_amplified', 'attention_decay', 'panic_everywhere']]
    
    # 每个stock只保留一行
    merged_day = merged_day.drop_duplicates(subset=['stock', 'f_date'], keep='first')
    
    return merged_day

# -*- 降低为月频 (均值和标准差) -*-
# 降低为月频之后等权合成为所对应因子（mean+std）/2
def calculate_rolling_mean_std(final_correlation_df, column, periods=20):
    """

    参数:
    final_correlation_df: 全A日行情数据(复权后)
    - f_date: 交易日期.表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - stock: 股票代码,类型为object。
    - columns: 因子列名
    - period: 时间窗口

    """   
    
    # 确保f_date列为字符串格式
    final_correlation_df['f_date'] = final_correlation_df['f_date'].astype(str)
    
    # 将f_date列转换为日期格式
    final_correlation_df['f_date'] = pd.to_datetime(final_correlation_df['f_date'], format='%Y%m%d')

    # 按stock和f_date排序
    final_correlation_df = final_correlation_df.sort_values(by=['stock', 'f_date'])
    
    # 计算滚动均值和滚动标准差
    rolling_means = final_correlation_df.groupby('stock')[column].rolling(window=periods, min_periods=5).mean()
    rolling_stds = final_correlation_df.groupby('stock')[column].rolling(window=periods, min_periods=5).std()
    
    # 将滚动均值和滚动标准差合并回原数据框
    final_correlation_df[column + '_mean'] = rolling_means.reset_index(level=0, drop=True)
    final_correlation_df[column + '_std'] = rolling_stds.reset_index(level=0, drop=True)
    
    # 获取每个月的最后一个交易日
    final_correlation_df['month'] = final_correlation_df['f_date'].dt.to_period('M')
    last_dates = final_correlation_df.groupby('month')['f_date'].max().reset_index()['f_date']

    # 筛选出每个月的最后一个交易日的滚动均值和滚动标准差
    result = final_correlation_df[final_correlation_df['f_date'].isin(last_dates)]

    return result[['stock', 'f_date', column + '_mean', column + '_std']]
