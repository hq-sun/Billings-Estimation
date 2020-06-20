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
df_backup = df.copy()
## 138534 obs, 7 vars
df.head()

# =============================================================================
# EDA
# =============================================================================
# Define some functions
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

df.columns.values
## ['Deal ID', 'Units Sold', 'Billings', 'Start Date', 'Deal URL', 'Segment', 'Inventory Type']
continuous_vars = ['Units Sold', 'Billings']
categorical_vars = ['Start Date', 'Segment', 'Inventory Type']

for col in list(df.columns.values):
    dataframe_description(df, col)
## Each row is a unique Deal ID, no duplicates
## No mising value in all columns

for col in list(continuous_vars):
    descriptive_stats_continuous(df, col)
## Units Sold varibale: minimum is -9100.0, which is weird
## Billings variabls: minimu is -218062.90099999993, which is weird

for col in list(continuous_vars):
    plot_distribution(df, col)

for col in list(categorical_vars):
    plot_counts(df, col)

# Check if Units Sold is same symbol with Billings
df_check = df.copy()
df_check['Units Sold x Billings'] = df_check['Units Sold'] * df_check['Billings']
df_check[df_check['Units Sold x Billings'] < 0]
## All rows Units Sold and Billings variables have the same sign

# Create indicator variable for 'First - Party' inventory
# df = df_backup.copy()
df['Inventory Type'].unique()
df['Inventory_FP'] = np.where(df['Inventory Type']=='First - Party', 1, 0)
df['Inventory_TP'] = np.where(df['Inventory Type']=='Third - Party', 1, 0)
df.drop(['Deal ID', 'Deal URL', 'Inventory Type'], axis = 1, inplace = True) 
df.columns

# Split the dataset by segments
local = df[(df.Segment=='Local')]
goods = df[(df.Segment=='Goods')]
travel = df[(df.Segment=='Travel')]

# Aggregation
local['Start Date'] = local['Start Date'].astype(str)
local['Start Date'] = pd.to_datetime(local['Start Date'])
local_agg = local.groupby(['Start Date'], as_index=True).agg({'Units Sold':'sum','Billings':'sum','Inventory_FP':'sum','Inventory_TP':'sum'})


goods['Start Date'] = goods['Start Date'].astype(str)
goods['Start Date'] = pd.to_datetime(goods['Start Date'])
goods_agg = goods.groupby(['Start Date'], as_index=True).agg({'Units Sold':'sum','Billings':'sum','Inventory_FP':'sum','Inventory_TP':'sum'})

travel['Start Date'] = travel['Start Date'].astype(str)
travel['Start Date'] = pd.to_datetime(travel['Start Date'])
travel_agg = travel.groupby(['Start Date'], as_index=True).agg({'Units Sold':'sum','Billings':'sum','Inventory_FP':'sum','Inventory_TP':'sum'})


# Sort local aggregated dataframe
local_agg_sorted = local_agg.sort_index()

# Split local dataframe to two - before 2013 and Year 2013
local_2013 = local_agg_sorted.loc['2013-01-01':'2013-12-31'] ## 354 obs (missing 11 days)
local_before_2013 = local_agg_sorted.loc[:'2012-12-31'] ## 150 obs

# Add 11 rows represent missing 11 days
local_2013_full = local_2013.asfreq(freq='1D')

# Join two dataframes vertically
local_agg_full_sorted = local_before_2013.append(local_2013_full) ## 515 obs

# Write dataframe to csv and pickle file
local_agg_full_sorted.to_csv('./data/clean/local_agg_full_sorted.csv')
local_agg_full_sorted.to_pickle('./data/clean/local_agg_full_sorted.pkl')

local_agg_full_sorted = pd.read_pickle(r'./data/clean/local_agg_full_sorted.pkl')

# =============================================================================
# Imputation
# =============================================================================

# Plot with missing data
local_agg_full_sorted.plot()

# Method 1 - Simple Imputer -- not so good b/c all missing days are imputed as same
from sklearn.impute import SimpleImputer
local_si = SimpleImputer().fit_transform(local_agg_full_sorted)
## 418

# Method 2 - Direct Interpolation
local_ip = local_agg_full_sorted.interpolate()
local_ip.plot()
## 442

# Method 3 - pchip Interpolation
local_pc = local_agg_full_sorted.interpolate(method='pchip') ## cumulative distribution 
local_pc.plot()
## 439

# Method 4 - akima Interpolation - USE THIS ONE FOR NOW
local_ak = local_agg_full_sorted.interpolate(method='akima') ##  smooth plotting
local_ak.plot()
## 436

# Method 5 - time Interpolation
# Works on daily and higher resolution data to interpolate given length of interval
local_time = local_agg_full_sorted.interpolate(method='time')
local_time.plot()


# Method 6 - MICE OKAY
local_mice = pd.read_csv(r'./data/clean/local_mice.csv')
local_mice.set_index('Start.Date', inplace=True)
local_mice.plot()
## 417

# Method 7 - kNN -- not so good b/c all missing days are imputed as same
from sklearn.impute import KNNImputer
local_knn = KNNImputer(n_neighbors=5).fit_transform(local_agg_full_sorted)
local_knn = pd.DataFrame(local_knn)
local_knn.index = list(local_agg_full_sorted.index) 
local_knn.plot()
## 418

# =============================================================================
# Get sum of each segment
# =============================================================================
local_ak.sum(axis = 0, skipna = True)/1000000
# Units Sold       14.432131
# Billings        436.451398

local_time.sum(axis = 0, skipna = True)/1000000
# Units Sold       14.992623
# Billings        442.018623

goods_agg.sum(axis = 0, skipna = True)/1000000
# Units Sold       10.419746
# Billings        282.245671

travel_agg.sum(axis = 0, skipna = True)/1000000
# Units Sold       0.378910
# Billings        70.552062


# =============================================================================
# LSTM
# =============================================================================
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Split local dataframe to two - Yeear 2012 and Year 2013 (omit one obs in 2011)
local_2013_full = local_agg_full_sorted.loc['2013-01-01':'2013-12-31'] ## 365 obs
local_2012 = local_agg_full_sorted.loc['2012-01-01':'2012-12-31'] ## 149 obs (after Sep 1, no missing)
local_2013_full.plot()
local_2012.plot()

local_2013_9_12 = local_agg_full_sorted.loc['2013-09-01':'2013-12-31'] 
local_2012_9_12 = local_agg_full_sorted.loc['2012-09-01':'2012-12-31'] 
local_2013_9_12.plot()
local_2012_9_12.plot()

local_2013_9_12_ts = local_2013_9_12[['Billings']]
local_2012_9_12_ts = local_2012_9_12[['Billings']]


# fix random seed for reproducibility
np.random.seed(7)

dataset = local_2013_9_12_ts.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# =============================================================================
# Try Some TIme Series
# =============================================================================
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

# Aggreagation to day level using sum
local_ts_agg = local_ts.groupby(['Start Date'], as_index=True)['Billings'].sum()
goods_ts_agg = goods_ts.groupby(['Start Date'], as_index=True)['Billings'].sum()
travel_ts_agg = travel_ts.groupby(['Start Date'], as_index=True)['Billings'].sum()

# Some plots
from matplotlib import pyplot
local_ts_agg.plot()
pyplot.show()

goods_ts_agg.plot()
pyplot.show()

travel_ts_agg.plot()
pyplot.show()



