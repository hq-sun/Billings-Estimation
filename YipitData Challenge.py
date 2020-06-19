#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:28:04 2020

@author: Heqing Sun
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get current working directory
os.getcwd()

# Read Excel data
df = pd.read_excel (r'./data/raw/Q4_2013_Groupon_North_America_Data_XLSX (1).xlsx', sheet_name='Q4 2013 Raw Data')
## 138534 obs, 7 vars
df.head()

# =============================================================================
# Define some functions for EDA
# =============================================================================
# For all varibales
def dataframe_description(df, col):
    print('Column Name:', col)
    print('Number of Rows:', len(df.index))
    print('Number of Missing Values:', df[col].isnull().sum())
    print('Percent Missing:', df[col].isnull().sum()/len(df.index)*100, '%')
    print('Number of Unique Values:', len(df[col].unique()))
    print('\n')

# For continuous variables    
def descriptive_stats_continuous(df, col):
    print('Column Name:', col)
    print('Mean:', np.mean(df[col]))
    print('Median:', np.nanmedian(df[col]))
    print('Standard Deviation:', np.std(df[col]))
    print('Minimum:', np.min(df[col]))
    print('Maximum:', np.max(df[col]))
    print('\n')

# Plotting distribution plots for continuous variables
def plot_distribution(df, col):
    sns.set(style='darkgrid')
    ax = sns.distplot(df[col].dropna())
    plt.xticks(rotation=90)
    plt.title('Distribution Plot for ' + col)
    plt.show()
    
# Plotting countplots for categorical vairables 
def plot_counts(df, col):
    sns.set(style='darkgrid')
    ax = sns.countplot(x=col, data=df)
    plt.xticks(rotation=90)
    plt.title('Count Plot')
    plt.show()
    
# =============================================================================
# EDA
# =============================================================================
df.columns.values
continuous_vars = ['Units Sold', 'Billings']
categorical_vars = ['Start Date', 'Segment', 'Inventory Type']

for col in list(df.columns.values):
    dataframe_description(df, col)
## No mising value in all columns

for col in list(continuous_vars):
    descriptive_stats_continuous(df, col)
## Units Sold varibale: minimum is -9100.0, which is weird
## Billings variabls: minimu is -218062.90099999993, which is weird

for col in list(continuous_vars):
    plot_distribution(df, col)

for col in list(categorical_vars):
    plot_counts(df, col)

# Split the dataset by segments
local = df[(df.Segment=='Local')]
goods = df[(df.Segment=='Goods')]
travel = df[(df.Segment=='Travel')]


local_ts = local[['Start Date','Billings']]
local_ts['Billings'].value_counts()
local_ts['Billings'].isnull().sum()
local_ts['Start Date'] = local_ts['Start Date'].astype(str)

goods_ts = goods[['Start Date','Billings']]
goods_ts['Billings'].value_counts()
goods_ts['Billings'].isnull().sum()
goods_ts['Start Date'] = goods_ts['Start Date'].astype(str)

travel_ts = travel[['Start Date','Billings']]
travel_ts['Billings'].value_counts()
travel_ts['Billings'].isnull().sum()
travel_ts['Start Date'] = travel_ts['Start Date'].astype(str)

# =============================================================================
# Aggreagation to day level using sum
# =============================================================================
local_ts_agg = local_ts.groupby(['Start Date'], as_index=True)['Billings'].sum()
goods_ts_agg = goods_ts.groupby(['Start Date'], as_index=True)['Billings'].sum()
travel_ts_agg = travel_ts.groupby(['Start Date'], as_index=True)['Billings'].sum()


# =============================================================================
# Some plots
# =============================================================================
from matplotlib import pyplot
local_ts_agg.plot()
pyplot.show()

goods_ts_agg.plot()
pyplot.show()

travel_ts_agg.plot()
pyplot.show()

# =============================================================================
# Imputation
# =============================================================================


