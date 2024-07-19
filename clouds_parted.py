"""
title: 云开雾散
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

# -*- 模糊关联度因子 -*-
def calculate_blur_correlation(day):
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
    # 1. 剔除 f_time=929, 930, 1500 的数据
    day = day[~day['f_time'].isin([929, 930, 1500])].copy()

    # 2. 计算每分钟的波动率（f_pct的标准差）
    def calculate_volatility(data):
        return data.rolling(window=5).std()

    day['volatility'] = day.groupby('stock')['f_pct'].apply(calculate_volatility)

    # 3. 计算每分钟的模糊性（波动率的标准差）
    def calculate_ambiguity(data):
        return data.rolling(window=5).std()

    day['ambiguity'] = day.groupby('stock')['volatility'].apply(calculate_ambiguity)

    # 4. 计算每只股票模糊性序列与分钟成交金额序列的相关系数
    def calculate_correlation(group):
        valid_data = group.dropna(subset=['ambiguity', 'f_amount'])
        if len(valid_data) < 2:
            return np.nan
        return valid_data['ambiguity'].corr(valid_data['f_amount'])

    # 日模糊关联度因子
    day_blur_correlation = day.groupby(['stock', 'f_date']).apply(calculate_correlation).reset_index()
    day_blur_correlation.columns = ['stock', 'f_date', 'blur_correlation']

    # 将日模糊关联度因子加回到 day DataFrame 中
    day = day.merge(day_blur_correlation, on=['stock', 'f_date'], how='left')

    return day

# -*- 模糊金额比因子，模糊数量比因子，模糊价差因子 -*-
def calculate_blur_money_ratio(day):
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
    day = calculate_blur_correlation(day)

    # 1. 计算每只股票当日模糊性的均值
    daily_ambiguity_mean = day.groupby(['stock', 'f_date'])['ambiguity'].transform('mean')

    # 2. 认定模糊性大于均值的部分为模糊性较大的时间段，记为“起雾时刻”
    day['fog_time'] = day['ambiguity'] > daily_ambiguity_mean

    # 3. 计算“起雾时刻”的分钟成交金额的均值，记为“雾中金额”
    fog_money_mean = day[day['fog_time']].groupby(['stock', 'f_date'])['f_amount'].transform('mean')

    # 4. 计算日内所有时间的分钟成交金额的均值，记为“总体金额”
    overall_money_mean = day.groupby(['stock', 'f_date'])['f_amount'].transform('mean')

    # 5. 使用“雾中金额”除以“总体金额”，记为“日模糊金额比”
    day['blur_money_ratio'] = fog_money_mean / overall_money_mean

    # 6. 计算“起雾时刻”的分钟成交量的均值，记为“雾中数量”
    fog_volume_mean = day[day['fog_time']].groupby(['stock', 'f_date'])['f_volume'].transform('mean')

    # 7. 计算日内所有时间的分钟成交量的均值，记为“总体数量”
    overall_volume_mean = day.groupby(['stock', 'f_date'])['f_volume'].transform('mean')

    # 8. 计算“雾中数量”除以“总体数量”，记为“日模糊数量比”
    day['blur_volume_ratio'] = fog_volume_mean / overall_volume_mean

    # 9. 计算“日模糊价差”因子
    day['blur_price_spread'] = day['blur_money_ratio'] - day['blur_volume_ratio']
    
    # 删除所有包含 NaN 的行
    day = day.dropna(subset=['blur_correlation', 'blur_money_ratio', 'blur_volume_ratio', 'blur_price_spread'])
   
    # 保留需要的列并只保留每个股票的一行
    result = day[['stock', 'f_date', 'blur_correlation', 'blur_money_ratio', 'blur_volume_ratio', 'blur_price_spread']].drop_duplicates(subset=['stock', 'f_date'])

    
    return result

# -*- 降低为月频 (均值和标准差) -*-
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

# -*- 修正日模糊价差因子 -*-
def calculate_adjusted_blur_spread(final_return_equal_df):
    """
    参数:
    final_correlation_df:(每日各个股票所对应因子值):
    - stock: 股票代码: 类型为object。
    - f_date: 交易日期,表示数据对应的日期。类型为int64,为YYYYMMDD格式。
    - 剩余多个因子列,这里是对blur_price_spread因子进行修正
    """
    final_return_equal_df = final_return_equal_df.copy()
    
    # 确保 f_date 列为日期格式
    final_return_equal_df['f_date'] = pd.to_datetime(final_return_equal_df['f_date'].astype(str), format='%Y%m%d')
    
    # 按股票和日期排序
    final_return_equal_df = final_return_equal_df.sort_values(by=['stock', 'f_date'])
    
    # 计算过去10天“blur_price_spread”的标准差
    final_return_equal_df['blur_price_spread_std'] = final_return_equal_df.groupby('stock')['blur_price_spread'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    
    # 计算当天“日模糊价差”为负的部分的和s1
    final_return_equal_df['negative_blur_price_spread'] = final_return_equal_df['blur_price_spread'].apply(lambda x: x if x < 0 else 0)
    final_return_equal_df['s1'] = final_return_equal_df.groupby('f_date')['negative_blur_price_spread'].transform('sum')
    
    # 计算“修正日模糊价差”
    def adjust_blur_price_spread(row):
        if row['blur_price_spread'] < 0:
            return row['blur_price_spread'] / row['blur_price_spread_std'] if row['blur_price_spread_std'] != 0 else row['blur_price_spread']
        else:
            return row['blur_price_spread']
    
    final_return_equal_df['adjusted_blur_price_spread'] = final_return_equal_df.apply(adjust_blur_price_spread, axis=1)
    
    # 计算当天“修正日模糊价差”为负的部分的和s2
    final_return_equal_df['negative_adjusted_blur_price_spread'] = final_return_equal_df['adjusted_blur_price_spread'].apply(lambda x: x if x < 0 else 0)
    final_return_equal_df['s2'] = final_return_equal_df.groupby('f_date')['negative_adjusted_blur_price_spread'].transform('sum')
    
    # 调整“修正日模糊价差”为负的部分
    def final_adjustment(row):
        if row['adjusted_blur_price_spread'] < 0:
            return row['adjusted_blur_price_spread'] / row['s2'] * row['s1'] if row['s2'] != 0 else row['adjusted_blur_price_spread']
        else:
            return row['adjusted_blur_price_spread']
    
    final_return_equal_df['final_adjusted_blur_price_spread'] = final_return_equal_df.apply(final_adjustment, axis=1)
    
    # 返回结果
    return final_return_equal_df[['stock', 'f_date', 'final_adjusted_blur_price_spread']]

# 然后需要计算计算过去 20 个交易日的“修正日模糊价差”的均值，记为“均修正模糊价差”因子。
# grouped = calculate_rolling_mean(adjusted_blur_spread_df,'final_adjusted_blur_price_spread', periods=20)
# grouped = grouped.dropna(subset=['final_adjusted_blur_price_spread'])
# grouped['final_adjusted_blur_price_spread'] = grouped.groupby('f_date')['final_adjusted_blur_price_spread'].transform(lambda x: mad_(x.values))
# 再将“均修正模糊价差”因子与“稳模糊价差”因子等权合成，得到“修正模糊价差”因子。
# grouped['adjusted_blur_price_spread'] = (grouped['final_adjusted_blur_price_spread']+grouped['blur_price_spread_std'])/2