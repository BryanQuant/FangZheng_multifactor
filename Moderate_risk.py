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

# -*- 激增时刻 -*-
def process_spike(day):
    # 剔除开盘和收盘数据，也就是剔除f_time为929的数据
    day = day[day['f_time'] != 929]

    # 计算个股每分钟的成交量相对于上一分钟的成交量的差值，作为该分钟成交量的增加量
    day['volume_diff'] = day.groupby('stock')['f_volume'].diff().fillna(0)

    # 计算每天每只stock分钟频成交量的增加量的均值 mean 和标准差 std
    volume_stats = day.groupby('stock')['volume_diff'].agg(['mean', 'std']).reset_index()

    # 合并统计数据到原始数据框
    day = day.merge(volume_stats, on='stock', how='left', suffixes=('', '_stats'))

    # 定义成交量激增的时刻
    day['is_spike'] = day['volume_diff'] > (day['mean'] + day['std'])

    # 输出结果，包含每个分钟的成交量激增标记
    result = day[['stock', 'f_date', 'f_time', 'f_volume', 'volume_diff', 'f_pct','is_spike']]

    return result

# -*- 耀眼波动率因子 -*-
def calculate_shiny_metrics(day):
    
    # 定义“耀眼5分钟”
    day['shiny_5_min'] = np.nan
    day.loc[day['is_spike'], 'shiny_5_min'] = day['f_time']

    # 向下填充“耀眼5分钟”列，以覆盖随后的4分钟
    day['shiny_5_min'] = day.groupby('stock')['shiny_5_min'].fillna(method='ffill', limit=4)

    # 过滤出有效的“耀眼5分钟”
    day['is_valid_shiny'] = day['f_time'] - day['shiny_5_min'] <= 4
    shiny_intervals = day[day['is_valid_shiny']]

   # 计算每个“耀眼5分钟”内收益率的标准差，作为“耀眼波动率”
    shiny_volatilities = shiny_intervals.groupby(['stock', 'f_date', 'shiny_5_min'])['f_pct'].std().reset_index()
    shiny_volatilities = shiny_volatilities.rename(columns={'f_pct': 'shiny_volatility'})

    # 计算每个股票的所有“耀眼波动率”的均值，作为“日耀眼波动率”
    daily_shiny_volatility = shiny_volatilities.groupby(['stock', 'f_date'])['shiny_volatility'].mean().reset_index()
    daily_shiny_volatility = daily_shiny_volatility.rename(columns={'shiny_volatility': 'daily_shiny_volatility'})

    # 计算“日耀眼波动率”的均值
    market_mean_volatility = daily_shiny_volatility['daily_shiny_volatility'].mean()

    # 计算个股“适度日耀眼波动率”
    daily_shiny_volatility['moderate_shiny_volatility'] = np.abs(
        daily_shiny_volatility['daily_shiny_volatility'] - market_mean_volatility
    )
    daily_shiny_volatility = daily_shiny_volatility[['stock','f_date','moderate_shiny_volatility']]
    
    return daily_shiny_volatility

# -*- 耀眼收益率因子 -*-
def calculate_shiny_return(day_test):
    # 1. 找到“激增时刻”对应的每个stock的每f_time的f_pct，称为“耀眼收益率”
    day_test['shiny_return'] = np.where(day_test['is_spike'], day_test['f_pct'], np.nan)
    
    # 2. 对每个stock在t日内所有的“耀眼收益率”求均值，作为“日耀眼收益率”
    daily_shiny_return = day_test.groupby(['stock', 'f_date'])['shiny_return'].mean().reset_index()
    daily_shiny_return = daily_shiny_return.rename(columns={'shiny_return': 'daily_shiny_return'})
    
    # 3. 计算所有stock的“日耀眼收益率”的均值，作为市场平均水平
    market_mean_shiny_return = daily_shiny_return['daily_shiny_return'].mean()
    
    # 计算“适度日耀眼收益率”
    daily_shiny_return['moderate_shiny_return'] = np.abs(
        daily_shiny_return['daily_shiny_return'] - market_mean_shiny_return
    )
    daily_shiny_return = daily_shiny_return[['stock','f_date','moderate_shiny_return']]
    
    return daily_shiny_return

# -*- 降频为月频 (均值和标准差) -*-
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

# -*- 合成适度冒险因子 -*-
def calculate_moderate_risk(vol, ret, volatility_col, return_col):
    """
    计算 moderate risk 并返回结果 DataFrame

    参数:
    vol (DataFrame): 包含股票波动率数据的 DataFrame
    ret (DataFrame): 包含股票收益率数据的 DataFrame
    volatility_col (str): moderate_shiny_volatility_mean 列的名称
    return_col (str): moderate_shiny_return_mean 列的名称

    返回:
    DataFrame: 包含股票、日期、moderate_risk 的 DataFrame
    """
    # 合并 vol 和 ret 数据框，依据 'stock' 和 'f_date' 列进行内连接
    grouped = pd.merge(vol, ret, on=['stock', 'f_date'], how='inner')

    # 只保留需要的列
    grouped = grouped[['stock', 'f_date', volatility_col, return_col]]

    # 计算 moderate_risk 列
    grouped['moderate_risk'] = (grouped[volatility_col] + grouped[return_col]) / 2

    return grouped