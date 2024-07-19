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

# -*- 获得当日各个stock的累积收益 -*-
def calculate_daily_return(df):
    """
    计算每个 stock 当日的涨跌幅。
    """
    # 计算每个 stock 的累积收益
    df['cumulative_return'] = df.groupby(['f_date', 'stock'])['f_pct'].apply(lambda x: (1 + x).cumprod() - 1)
    
    # 提取每日最后一个时间点的累积收益作为当日涨跌幅
    daily_return = df.groupby(['f_date', 'stock']).apply(lambda x: x['cumulative_return'].iloc[-1]).reset_index()
    daily_return.rename(columns={0: 'daily_return'}, inplace=True)
    
    # 将 daily_return 合并回原 DataFrame
    df = df.merge(daily_return, on=['f_date', 'stock'], how='left')
    
    return df

# -*- 异常值处理和标准化函数 -*-
def mad_(points):
    k = 1.4826
    tmp = points.copy()
    median = np.median(points, axis=0)
    diff = np.abs(points - median)
    mad = np.median(diff)
    std = k * mad
    # 参考统计学中正态分布的95%置信区间，采用正负1.96个标准差
    maxval, minval = median + 3.0 * std, median - 3.0 * std
    
    tmp[tmp < minval] = np.linspace(median - 3.5 * std, median - 3.0 * std, num=len(tmp[tmp < minval]))
    tmp[tmp > maxval] = np.linspace(median + 3.0 * std, median + 3.5 * std, num=len(tmp[tmp > maxval]))
    tmp = (tmp - np.mean(tmp)) / np.std(tmp)
    return tmp

# -*- 股票券池处理 -*-
def stock_filter(combined_df, back_file, not_old_file):
    # 排序
    combined_df = combined_df.sort_values(by=['TRADE_DT'])

    # 涨停股删除------------------------------
    combined_df['S_DQ_LIMIT'].fillna(0, inplace=True)

    # 当S_DQ_OPEN等于S_DQ_LIMIT时，标记为1，否则为0
    combined_df['up'] = (combined_df['S_DQ_OPEN'] == combined_df['S_DQ_LIMIT']).astype(int)

    # 停牌股删除------------------------------
    # 当S_DQ_TRADESTATUSCODE等于0时，标记为1，否则为0
    combined_df['stop'] = (combined_df['S_DQ_TRADESTATUSCODE'] == 0).astype(int)

    # 读取back和not_old数据
    back = pd.read_pickle(back_file)
    not_old = pd.read_pickle(not_old_file)
    
    # 新上市股票删除---------------------------
    not_old['new'] = 1

    # 只保留S_INFO_WINDCODE，S_INFO_LISTDATE，new三列
    not_old = not_old[['S_INFO_WINDCODE', 'S_INFO_LISTDATE', 'new']]

    # 合并数据框，combined_df为左表，TRADE_DT与S_INFO_LISTDATE对齐
    combined_df = combined_df.merge(not_old, left_on=['S_INFO_WINDCODE', 'TRADE_DT'], right_on=['S_INFO_WINDCODE', 'S_INFO_LISTDATE'], how='left')

    # 对new列进行向下填充
    combined_df['new'] = combined_df.groupby('S_INFO_WINDCODE')['new'].ffill().fillna(0).astype(int)

    # 退市股票删除---------------------------
    back['back'] = 1

    # 只保留S_INFO_WINDCODE，S_INFO_DELISTDATE，back三列
    back = back[['S_INFO_WINDCODE', 'S_INFO_DELISTDATE', 'back']]

    # 合并数据框，combined_df为左表，TRADE_DT与S_INFO_DELISTDATE对齐
    combined_df = combined_df.merge(back, left_on=['S_INFO_WINDCODE', 'TRADE_DT'], right_on=['S_INFO_WINDCODE', 'S_INFO_DELISTDATE'], how='left')

    # 对back列进行向下填充
    combined_df['back'] = combined_df.groupby('S_INFO_WINDCODE')['back'].ffill().fillna(0).astype(int)

    # 对 'up', 'new', 'back' 列求和，生成 'sum' 列
    combined_df['sum'] = combined_df[['up', 'new', 'back','stop']].sum(axis=1)

    # 只保留 'sum' 列等于 0 的行
    result_df = combined_df[combined_df['sum'] == 0]

    return result_df

# -*- 计算月度收益率 -*-
def calculate_monthly_returns(in_df):
    test_df = in_df.copy()
    # 将TRADE_DT转换为日期格式
    test_df['TRADE_DT'] = pd.to_datetime(test_df['TRADE_DT'], format='%Y%m%d')

    # 创建一个列表示年月
    test_df['YEAR_MONTH'] = test_df['TRADE_DT'].dt.to_period('M')

    # 计算累计收益
    test_df['CUMULATIVE_RETURN'] = (1 + test_df['S_DQ_PCTCHANGE']).groupby(test_df['S_INFO_WINDCODE']).cumprod()

    # 提取每个月的第一个和最后一个交易日
    first_day_df = test_df.groupby(['S_INFO_WINDCODE', 'YEAR_MONTH']).first().reset_index()
    last_day_df = test_df.groupby(['S_INFO_WINDCODE', 'YEAR_MONTH']).last().reset_index()

    # 合并第一个和最后一个交易日的数据
    merged_df = pd.merge(first_day_df, last_day_df, on=['S_INFO_WINDCODE', 'YEAR_MONTH'], suffixes=('_FIRST', '_LAST'))

    # 计算月涨跌幅
    merged_df['MONTHLY_RETURN'] = (merged_df['CUMULATIVE_RETURN_LAST'] / merged_df['CUMULATIVE_RETURN_FIRST']) - 1

    # 提取需要的列
    result_df = merged_df[['S_INFO_WINDCODE', 'TRADE_DT_LAST', 'MONTHLY_RETURN']]

    # 重命名列
    result_df = result_df.rename(columns={'TRADE_DT_LAST': 'TRADE_DT'})

    return result_df

# -*- 行业暴露度 -*-
def process_industry_stock(industry, test):
    # 将f_date列转换为字符串格式
    test['f_date'] = test['f_date'].astype(str)
    
    # 将Trade_Day列转换为字符串格式
    industry['Trade_Day'] = industry['Trade_Day'].astype(str)
    
    # 合并两个数据框，依据stock等于S_CON_WINDCODE，f_date等于Trade_Day
    merged_df = pd.merge(test, industry, left_on=['stock', 'f_date'], right_on=['S_CON_WINDCODE', 'Trade_Day'], how='inner')
    
    # 对S_INFO_WINDCODE进行get_dummies处理
    dummies = pd.get_dummies(merged_df['S_INFO_WINDCODE'], prefix='Industry')
    
    # 将dummies与merged_df合并
    merged_df = pd.concat([merged_df, dummies], axis=1)
    
    # 删除不需要的列
    merged_df = merged_df.drop(['S_CON_WINDCODE', 'Trade_Day', 'S_INFO_WINDCODE'], axis=1)
    
    # 返回处理后的数据框
    return merged_df

# -*- 行业/市值暴露度 -*-
def process_and_merge(industry, test, mv):
    # 使用前面的函数处理industry和test
    industry_stock = process_industry_stock(industry, test)
    
    # 删除S_DQ_MV为NaN的行
    mv = mv.dropna(subset=['S_DQ_MV'])

    # 对S_DQ_MV进行log变换
    mv['S_DQ_MV'] = np.log(mv['S_DQ_MV'])
    
    # 将f_date列从datetime格式转换为字符串格式'YYYYMMDD'
    industry_stock['f_date'] = pd.to_datetime(industry_stock['f_date']).dt.strftime('%Y%m%d')
    
    # 将TRADE_DT列转换为字符串格式
    mv['TRADE_DT'] = mv['TRADE_DT'].astype(str)
    
    # 只保留log后的S_DQ_MV列
    mv = mv[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_MV']]
    
    # 合并industry_stock和mv，依据stock等于S_INFO_WINDCODE，f_date等于TRADE_DT
    final_df = pd.merge(industry_stock, mv, left_on=['stock', 'f_date'], right_on=['S_INFO_WINDCODE', 'TRADE_DT'], how='inner')
    
    # 删除不需要的列
    final_df = final_df.drop(['S_INFO_WINDCODE', 'TRADE_DT'], axis=1)
    
    return final_df

# -*- 中性化处理 -*-
def calculate_residuals(final_df, target_column):
    residuals = []
    
    grouped = final_df.groupby('f_date')
    for name, group in tqdm(grouped, desc='Calculating residuals'):
        y = group[target_column].astype(float)
        x = group.iloc[:, 3:].astype(float)  # 从第三列开始为x值
        
        # 使用OLS回归并计算残差
        model = sm.OLS(y, x, hasconst=False, missing='drop').fit()
        resid = model.resid
        
        # 将残差加入数据框
        temp_df = group.copy()
        temp_df['factor'] = resid
        temp_df = temp_df[['stock', 'f_date', 'factor']]
        residuals.append(temp_df)
    
    # 合并所有残差结果
    residuals_df = pd.concat(residuals, ignore_index=True)
    
    return residuals_df

# -*- 读取文件夹文件并合并 -*-
def read_and_concat_files(file_paths, description):
    """
    读取文件路径列表中的所有文件，并合并为一个DataFrame。
    使用tqdm显示进度条。
    """
    final_df = pd.DataFrame()
    with tqdm(total=len(file_paths), desc=f'Reading {description}') as pbar:
        for file_path in file_paths:
            df = pd.read_pickle(file_path)
            final_df = pd.concat([final_df, df], ignore_index=True)
            pbar.update(1)
    return final_df

# -*- 最大回撤 -*-
def calculate_max_drawdown(cumulative_returns):
    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown

# -*- 因子检验 -*-
def factor_testing(check_factor_df, group_column):
    # 将f_date列从字符串格式转换为datetime格式
    check_factor_df['f_date'] = pd.to_datetime(check_factor_df['f_date'], format='%Y%m%d')

    # 确保数据按时间排序
    check_factor_df = check_factor_df.sort_values(by=['f_date'])

    # 获取所有的日期
    all_dates = check_factor_df['f_date'].unique()

    # 分成10组，从小到大分组
    check_factor_df['group'] = check_factor_df.groupby('f_date')[group_column].transform(
        lambda x: pd.qcut(-x.rank(method='first', ascending=False), 10, labels=False, duplicates='drop') + 1)
    
    # 创建一个DataFrame存储每组的月收益
    group_monthly_returns = pd.DataFrame(index=all_dates, columns=[f'Group_{i}' for i in range(1, 11)])

    # 计算每个月的分组收益
    for current_date in tqdm(all_dates[:-1], desc="Calculating group returns"):
        next_date = all_dates[all_dates > current_date][0]  # 获取下一个日期
        for i in range(1, 11):
            group_stocks = check_factor_df[(check_factor_df['f_date'] == current_date) & (check_factor_df['group'] == i)]['stock']
            next_month_returns = check_factor_df[(check_factor_df['f_date'] == next_date) & (check_factor_df['stock'].isin(group_stocks))]['MONTHLY_RETURN']
            group_monthly_returns.loc[next_date, f'Group_{i}'] = next_month_returns.mean()

    # 计算累积收益
    cumulative_returns = (1 + group_monthly_returns.fillna(0)).cumprod()
    cumulative_returns.iloc[0] = 1  # 将第一个月的值设为1

    # 计算多空组合收益（多头组-空头组）
    long_short_returns = group_monthly_returns['Group_1'] - group_monthly_returns['Group_10']
    cumulative_long_short_returns = (1 + long_short_returns.fillna(0)).cumprod()
    cumulative_long_short_returns.iloc[0] = 1  # 将第一个月的值设为1

    # 计算long-average组合收益（多头组-剩余组平均）
    average_returns = group_monthly_returns.drop(columns=['Group_1']).mean(axis=1)
    long_average_returns = group_monthly_returns['Group_1'] - average_returns
    cumulative_long_average_returns = (1 + long_average_returns.fillna(0)).cumprod()
    cumulative_long_average_returns.iloc[0] = 1  # 将第一个月的值设为1

    # 计算IC和Rank IC与下个月收益的相关性
    check_factor_df['next_month_return'] = check_factor_df.groupby('stock')['MONTHLY_RETURN'].shift(-1)
    ic_series = check_factor_df.groupby('f_date').apply(
        lambda x: x[group_column].corr(x['next_month_return']))
    rank_ic_series = check_factor_df.groupby('f_date').apply(
        lambda x: x[group_column].rank().corr(x['next_month_return'].rank()))

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std * np.sqrt(12)

    rank_ic_mean = rank_ic_series.mean()
    rank_ic_std = rank_ic_series.std()
    rank_ir = rank_ic_mean / rank_ic_std

    # 计算年化收益
    annualized_return = cumulative_long_short_returns.iloc[-1] ** (12 / len(cumulative_long_short_returns)) - 1
    group1_annualized_return = cumulative_returns['Group_1'].iloc[-1] ** (12 / len(cumulative_returns)) - 1
    long_average_annualized_return = cumulative_long_average_returns.iloc[-1] ** (12 / len(cumulative_long_average_returns)) - 1
    
    # 计算年化波动率
    annualized_volatility = long_short_returns.std() 

    # 计算最大回撤
    max_drawdown = calculate_max_drawdown(cumulative_long_short_returns)

    # 统计月度胜率
    monthly_win_rate = (long_short_returns > 0).mean()

    # 将结果整合到一个DataFrame里
    results = pd.DataFrame({
        'Metric': ['IC Mean', 'IC Std', 'Information Ratio', 'Rank IC Mean', 'Rank IC Std', 'Rank Information Ratio', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Monthly Win Rate', 'Group 1 Annualized Return', 'Long-Average Annualized Return'],
        'Value': [ic_mean, ic_std, ir, rank_ic_mean, rank_ic_std, rank_ir, annualized_return, annualized_volatility, max_drawdown, monthly_win_rate, group1_annualized_return, long_average_annualized_return]
    })
    results['Value'] = results['Value'].apply(lambda x: f'{x:.2%}' if isinstance(x, (float, np.floating)) else x)
    
    print(results)

    # 画出10组分组表现
    plt.figure(figsize=(14, 8))
    for i in range(1, 11):
        plt.plot(cumulative_returns.index, cumulative_returns[f'Group_{i}'], label=f'Group {i}')
    plt.legend()
    plt.title('Cumulative Returns by Group')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    
    # 设置横坐标显示年份，不重复显示
    date_range = pd.date_range(start=cumulative_returns.index.min(), end=cumulative_returns.index.max(), freq='6M')
    plt.xticks(date_range, date_range.strftime('%Y-%m'), rotation=45)
    
    plt.show()

    # 画出多空组合累积收益图
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_long_short_returns.index, cumulative_long_short_returns, label='Long-Short Portfolio')
    plt.legend()
    plt.title('Cumulative Returns of Long-Short Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    
    # 设置横坐标显示年份，不重复显示
    plt.xticks(date_range, date_range.strftime('%Y-%m'), rotation=45)
    
    plt.show()

    # 计算并画出10组的年化收益柱状图
    annualized_returns = (group_monthly_returns + 1).prod() ** (12 / group_monthly_returns.count()) - 1

    plt.figure(figsize=(14, 8))
    annualized_returns.plot(kind='bar')
    plt.title('Annualized Returns by Group')
    plt.xlabel('Group')
    plt.ylabel('Annualized Return')
    plt.xticks(rotation=0)
    plt.show()

    # 画出long-average组合累积收益图
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_long_average_returns.index, cumulative_long_average_returns, label='Long-Average Portfolio')
    plt.legend()
    plt.title('Cumulative Returns of Long-Average Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    
    # 设置横坐标显示年份，不重复显示
    plt.xticks(date_range, date_range.strftime('%Y-%m'), rotation=45)
    
    plt.show()
