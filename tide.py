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

# -*- 全潮汐因子 -*-
def calculate_tidal_events(day):
    # 剔除 f_time=929, 930, 1500 的数据
    day = day[~day['f_time'].isin([929, 930, 1500])].copy()

    # 重新标记 f_time
    day['f_time'] = day.groupby('stock').cumcount() + 1

    # 计算每分钟及其前后 4 分钟的成交量总和，作为“邻域成交量”
    day['neighborhood_volume'] = day.groupby('stock')['f_volume'].apply(
        lambda x: x.rolling(window=9, min_periods=9, center=True).sum()
    )

    # 找到“顶峰时刻”
    peak_time = day.groupby('stock')['neighborhood_volume'].idxmax()
    day['peak_time'] = False
    day.loc[peak_time, 'peak_time'] = True

    results = []

    for stock, stock_data in day.groupby('stock'):
        stock_data = stock_data.sort_values('f_time')

        # 找到“顶峰时刻”
        peak_time = stock_data.loc[stock_data['peak_time'], 'f_time'].values[0]

        # 找到“涨潮时刻”
        pre_peak_data = stock_data[(stock_data['f_time'] < peak_time) & (stock_data['f_time'] >= 5)]
        if not pre_peak_data.empty:
            low_pre_peak = pre_peak_data.loc[pre_peak_data['neighborhood_volume'].idxmin()]
            surge_time = low_pre_peak['f_time']
            surge_volume = low_pre_peak['neighborhood_volume']
            surge_close = low_pre_peak['f_close']
        else:
            surge_time = np.nan
            surge_volume = np.nan
            surge_close = np.nan

        # 找到“退潮时刻”
        post_peak_data = stock_data[(stock_data['f_time'] > peak_time) & (stock_data['f_time'] <= 233)]
        if not post_peak_data.empty:
            low_post_peak = post_peak_data.loc[post_peak_data['neighborhood_volume'].idxmin()]
            ebb_time = low_post_peak['f_time']
            ebb_volume = low_post_peak['neighborhood_volume']
            ebb_close = low_post_peak['f_close']
        else:
            ebb_time = np.nan
            ebb_volume = np.nan
            ebb_close = np.nan

        # 计算价格变动速率
        if not np.isnan(surge_time) and not np.isnan(ebb_time):
            price_change_rate = (ebb_close - surge_close) / surge_close / (ebb_time - surge_time)
        else:
            price_change_rate = np.nan

        # 保存结果
        results.append({
            'stock': stock,
            'f_date': stock_data['f_date'].iloc[0],
            'peak_time': peak_time,
            'surge_time': surge_time,
            'surge_volume': surge_volume,
            'surge_close': surge_close,
            'ebb_time': ebb_time,
            'ebb_volume': ebb_volume,
            'ebb_close': ebb_close,
            'price_change_rate': price_change_rate
        })

    result_df = pd.DataFrame(results)
    result_df = result_df[['stock','f_date','price_change_rate']]
    return result_df

# -*- “强势半潮汐”因子 和 “弱势半潮汐”因子 -*-
def calculate_tidal_half_events(day):
    # 剔除 f_time=929, 930, 1500 的数据
    day = day[~day['f_time'].isin([929, 930, 1500])].copy()

    # 重新标记 f_time
    day['f_time'] = day.groupby('stock').cumcount() + 1

    # 计算每分钟及其前后 4 分钟的成交量总和，作为“邻域成交量”
    day['neighborhood_volume'] = day.groupby('stock')['f_volume'].apply(
        lambda x: x.rolling(window=9, min_periods=9, center=True).sum()
    )

    # 找到“顶峰时刻”
    peak_time = day.groupby('stock')['neighborhood_volume'].idxmax()
    day['peak_time'] = False
    day.loc[peak_time, 'peak_time'] = True

    results = []

    for stock, stock_data in day.groupby('stock'):
        stock_data = stock_data.sort_values('f_time')

        # 找到“顶峰时刻”
        peak_time = stock_data.loc[stock_data['peak_time'], 'f_time'].values[0]

        # 找到“涨潮时刻”
        pre_peak_data = stock_data[(stock_data['f_time'] < peak_time) & (stock_data['f_time'] >= 5)]
        if not pre_peak_data.empty:
            low_pre_peak = pre_peak_data.loc[pre_peak_data['neighborhood_volume'].idxmin()]
            surge_time = low_pre_peak['f_time']
            surge_volume = low_pre_peak['neighborhood_volume']
            surge_close = low_pre_peak['f_close']
        else:
            surge_time = np.nan
            surge_volume = np.nan
            surge_close = np.nan

        # 找到“退潮时刻”
        post_peak_data = stock_data[(stock_data['f_time'] > peak_time) & (stock_data['f_time'] <= 233)]
        if not post_peak_data.empty:
            low_post_peak = post_peak_data.loc[post_peak_data['neighborhood_volume'].idxmin()]
            ebb_time = low_post_peak['f_time']
            ebb_volume = low_post_peak['neighborhood_volume']
            ebb_close = low_post_peak['f_close']
        else:
            ebb_time = np.nan
            ebb_volume = np.nan
            ebb_close = np.nan

        if not np.isnan(surge_time) and not np.isnan(ebb_time):
            if surge_volume < ebb_volume:
                # 涨潮是强势半潮汐
                strong_start_time = surge_time
                strong_end_time = peak_time
                strong_start_close = surge_close
                strong_end_close = stock_data.loc[stock_data['f_time'] == peak_time, 'f_close'].values[0]

                weak_start_time = peak_time
                weak_end_time = ebb_time
                weak_start_close = stock_data.loc[stock_data['f_time'] == peak_time, 'f_close'].values[0]
                weak_end_close = ebb_close
            else:
                # 退潮是强势半潮汐
                strong_start_time = peak_time
                strong_end_time = ebb_time
                strong_start_close = stock_data.loc[stock_data['f_time'] == peak_time, 'f_close'].values[0]
                strong_end_close = ebb_close

                weak_start_time = surge_time
                weak_end_time = peak_time
                weak_start_close = surge_close
                weak_end_close = stock_data.loc[stock_data['f_time'] == peak_time, 'f_close'].values[0]

            # 计算强势半潮汐价格变化速率
            strong_price_change_rate = (strong_end_close - strong_start_close) / strong_start_close / (strong_end_time - strong_start_time)

            # 计算弱势半潮汐价格变化速率
            weak_price_change_rate = (weak_end_close - weak_start_close) / weak_start_close / (weak_end_time - weak_start_time)
        else:
            strong_price_change_rate = np.nan
            weak_price_change_rate = np.nan

        # 保存结果
        results.append({
            'stock': stock,
            'f_date': stock_data['f_date'].iloc[0],
            'peak_time': peak_time,
            'surge_time': surge_time,
            'surge_volume': surge_volume,
            'surge_close': surge_close,
            'ebb_time': ebb_time,
            'ebb_volume': ebb_volume,
            'ebb_close': ebb_close,
            'strong_price_change_rate': strong_price_change_rate,
            'weak_price_change_rate': weak_price_change_rate
        })

    result_df = pd.DataFrame(results)
    result_df = result_df[['stock','f_date','strong_price_change_rate','weak_price_change_rate']]
    return result_df

# -*- 降频为月频 (只有均值) -*-
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

# -*- 降频为月频 （均值和标准差）-*-
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
    
    
    # 获取每个月的最后一个交易日
    final_correlation_df['month'] = final_correlation_df['f_date'].dt.to_period('M')
    last_dates = final_correlation_df.groupby('month')['f_date'].max().reset_index()['f_date']

    # 筛选出每个月的最后一个交易日的滚动均值和滚动标准差
    result = final_correlation_df[final_correlation_df['f_date'].isin(last_dates)]

    return result[['stock', 'f_date', column + '_mean', column + '_std']]


