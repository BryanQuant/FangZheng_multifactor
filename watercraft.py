"""
title: 水中行舟
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
from scipy.stats import spearmanr
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

# -*- 计算合理收益率 -*-
# combined_df：全A日行情数据（复权后）
def calculate_reasonable_return(combined_df):
    """
    参数:
    combined_df: 全A日行情数据(复权后)
    
    """
    
    # 计算日内收益率
    combined_df['intraday_return'] = combined_df['S_DQ_ADJCLOSE'] / combined_df['S_DQ_ADJOPEN'] - 1

    # 计算 t 日及过去 19 个交易日的合理收益率
    combined_df['reasonable_return'] = combined_df.groupby('S_INFO_WINDCODE')['intraday_return'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True)

    # 只保留所需的三列
    combined_df = combined_df[['S_INFO_WINDCODE', 'TRADE_DT', 'reasonable_return']]

    # 重命名列名
    combined_df = combined_df.rename(columns={'S_INFO_WINDCODE': 'stock', 'TRADE_DT': 'f_date'})
    
    # 将 f_date 列转换为字符串格式，然后再转换为 numpy.int64 格式
    combined_df['f_date'] = combined_df['f_date'].astype(np.int64)

    return combined_df

# -*- 计算日度高低额差 -*-
def calculate_high_low_volume_difference(day_test):
    """
    参数:
    day_test:
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
    - reasonable_return: 合理收益率, 类型为float64。
    - S_DQ_MV: 个股流通市值, 类型为float64。
    """   
    # 1. 剔除f_time=929的行，并获取当日开盘价
    day_test = day_test[day_test['f_time'] != 929].copy()
    day_test['open_price'] = day_test.groupby('stock')['f_open'].transform(lambda x: x.iloc[0])

    # 2. 计算每分钟的“相对开盘收益率”
    day_test['relative_open_return'] = (day_test['f_close'] / day_test['open_price']) - 1

    # 3. 标记价格处于相对高位和低位的时刻
    day_test['high_price_flag'] = day_test['relative_open_return'] > day_test['reasonable_return']
    day_test['low_price_flag'] = day_test['relative_open_return'] < day_test['reasonable_return']

    # 4. 分别计算“高位成交额”和“低位成交额”
    day_test['high_volume'] = np.where(day_test['high_price_flag'], day_test['f_amount'], 0)
    day_test['low_volume'] = np.where(day_test['low_price_flag'], day_test['f_amount'], 0)
    high_volume_sum = day_test.groupby('stock')['high_volume'].sum().reset_index(name='high_volume_sum')
    low_volume_sum = day_test.groupby('stock')['low_volume'].sum().reset_index(name='low_volume_sum')

    # 合并回原数据框
    day_test = day_test.merge(high_volume_sum, on='stock')
    day_test = day_test.merge(low_volume_sum, on='stock')

    # 5. 计算“高低额差”
    day_test['high_low_volume_diff'] = (day_test['high_volume_sum'] - day_test['low_volume_sum']) / day_test['S_DQ_MV']

    # 保留所需列，并去重
    result = day_test[['stock', 'f_date', 'high_low_volume_diff']].drop_duplicates()

    return result

# -*- 计算“随波逐流”因子 -*-
def calculate_follow_the_trend(final_return_equal_df):
    """
    参数:
    final_return_equal_df: 日度高低额差因子表
    
    """
    # 将f_date列转换为日期格式
    final_return_equal_df['f_date'] = pd.to_datetime(final_return_equal_df['f_date'], format='%Y%m%d')

    # 按股票和日期排序
    final_return_equal_df = final_return_equal_df.sort_values(by=['stock', 'f_date'])

    # 获取每个月的最后一个交易日
    final_return_equal_df['month'] = final_return_equal_df['f_date'].dt.to_period('M')
    last_dates = final_return_equal_df.groupby('month')['f_date'].max().reset_index()['f_date']

    # 创建前19个交易日的列
    for i in range(1, 20):
        final_return_equal_df[f'lag_{i}'] = final_return_equal_df.groupby('stock')['high_low_volume_diff'].shift(i)

    # 删除包含NaN值的行
    final_return_equal_df = final_return_equal_df.dropna(subset=[f'lag_{i}' for i in range(1, 20)])

    # 只保留每个月最后一个交易日的行
    last_day_df = final_return_equal_df[final_return_equal_df['f_date'].isin(last_dates)]

    # 去掉high_low_volume_diff和所有lag_i列都为0的行
    cols_to_check = ['high_low_volume_diff'] + [f'lag_{i}' for i in range(1, 20)]
    last_day_df = last_day_df[(last_day_df[cols_to_check] != 0).any(axis=1)]

    # 保留需要的列
    cols_to_keep = ['stock', 'f_date', 'high_low_volume_diff'] + [f'lag_{i}' for i in range(1, 20)]
    last_day_df = last_day_df[cols_to_keep]

    # 初始化存储结果的列表
    results = []

    # 按每个f_date计算
    for date in tqdm(last_day_df['f_date'].unique(), desc="Processing each date"):
        date_df = last_day_df[last_day_df['f_date'] == date]
        data_matrix = date_df.drop(columns=['stock', 'f_date']).T

        # 计算spearman相关系数矩阵
        correlation_matrix = data_matrix.corr(method='spearman').abs()

        # 将对角线元素设为 NaN
        np.fill_diagonal(correlation_matrix.values, np.nan)

        # 计算“随波逐流”因子
        follow_the_trend = correlation_matrix.mean(axis=1)
        
        # 创建结果DataFrame并添加到结果列表中
        result_df = pd.DataFrame({
            'stock': date_df['stock'].values,
            'f_date': date,
            'follow_the_trend': follow_the_trend.values
        })
        results.append(result_df)

    # 合并所有结果
    final_result_df = pd.concat(results, ignore_index=True)

    return final_result_df

# -*- 计算“孤雁出群”因子 -*-
def calculate_market_dispersion_and_lonely_bird(day):
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
    # 1. 剔除开盘和收盘部分的信息
    day = day[~day['f_time'].isin([929])].copy()

    # 2. 计算每一分钟所有股票的分钟收益率的标准差，作为“分钟市场分化度”
    day['market_dispersion'] = day.groupby('f_time')['f_pct'].transform('std')

    # 3. 计算这一天所有“分钟市场分化度”的均值，找到“不分化时刻”
    mean_dispersion = day['market_dispersion'].mean()
    day['non_dispersion_time'] = day['market_dispersion'] < mean_dispersion

    # 4. 取市场上所有股票在当日“不分化时刻”的成交额序列
    non_dispersion_data = day[day['non_dispersion_time']].pivot(index='f_time', columns='stock', values='f_amount')

    # 计算相关系数矩阵
    correlation_matrix = non_dispersion_data.corr(method='pearson').abs()

    # 将对角线元素设为 NaN
    np.fill_diagonal(correlation_matrix.values, np.nan)

    # 计算每只股票与其余股票的相关系数绝对值的均值，记为“日孤雁出群”因子
    lonely_bird_factors = correlation_matrix.mean(axis=1)

    # 将结果转换为数据框并返回
    result = lonely_bird_factors.reset_index()
    result.columns = ['stock', 'lonely_bird']
    result['f_date'] = day['f_date'].iloc[0]

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

# -*- 合成水中行舟因子 -*-
def calculate_water_craft(bird, trend, lone_bird, follow_trend):
    """
    参数:
    bird (DataFrame): 包含孤雁出群因子数据的 DataFrame
    trend (DataFrame): 包含随波逐流因子数据的 DataFrame
    lone_bird (str): 因子列的名称
    follow_trend (str): 因子列的名称
    """
    # 合并 vol 和 ret 数据框，依据 'stock' 和 'f_date' 列进行内连接
    grouped = pd.merge(bird, trend, on=['stock', 'f_date'], how='inner')

    # 只保留需要的列
    grouped = grouped[['stock', 'f_date', lone_bird, follow_trend]]

    # 计算 moderate_risk 列
    grouped['water_craft'] = (grouped[lone_bird] + grouped[follow_trend]) / 2

    return grouped