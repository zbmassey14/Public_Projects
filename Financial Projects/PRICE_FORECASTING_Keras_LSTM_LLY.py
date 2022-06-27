#!/usr/bin/env python
# coding: utf-8

# In[113]:


#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import fmpsdk
import pyfmpcloud
import numpy as np
import ta as ta
from pyfmpcloud import stock_time_series as sts
from pyfmpcloud import settings
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.momentum import RSIIndicator
from ta.utils import dropna
from ta.volatility import BollingerBands
from functools import reduce
import seaborn as sns
import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import random

#Adding technical indicators
from ta.momentum import RSIIndicator
from ta.momentum import AwesomeOscillatorIndicator
from ta.momentum import ROCIndicator
from ta.momentum import pvo
from ta.trend import ADXIndicator
from ta.others import DailyLogReturnIndicator
from ta.trend import CCIIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import KAMAIndicator
from ta.trend import MassIndex
from ta.trend import DPOIndicator


# In[114]:


#Project Goal: Predict Stock Price Movement
#First we retrive stock prive data on LLY
#Then we caluate technical inidators which help identify movement
#Then we preprocess the data for our models
#First we make a Single-Step LSTM
#Then we move to a Single-Shot Multi-Step LSTM
#We then evaluate the performance & deploy the model for the next months predictions


# In[115]:


#Set the API key
settings.set_apikey('39c5facda41abb1e5c6336fb376f40ed')

#Pull stock price data "lly"
stock = sts.historical_stock_data('LLY', dailytype = 'change', start = '2015-04-15', end = '2022-06-20')
#Need to reverse the order
stock = stock[::-1] 
stock.close


# In[116]:


#Look at a plot
plt.plot(stock.close) #or plt.plot(lly["close"])


# In[117]:


#Adding technical indicators  to be used as feature inputs
#RSI
rsi = RSIIndicator(close = stock.close, window = 21)
rsi = rsi.rsi()
rsi = rsi.to_frame()

#Awesome
awesome = AwesomeOscillatorIndicator(high = stock.high,
                                     low = stock.low,
                                     window1 = 5,
                                     window2 = 34)
awesome = awesome.awesome_oscillator()
awesome = awesome.to_frame()

#Kaufman’s Adaptive Moving Average (KAMA)
kama = KAMAIndicator(close = stock.close)
kama = kama.kama()
kama = kama.to_frame()

#Rate of Change (ROC)
roc = ROCIndicator(close = stock.close,
                   window = 12)
roc = roc.roc()
roc = roc.to_frame()

#Percentage Volume Oscillator
pvo = pvo(volume = stock.volume,
          window_slow = 26,
          window_fast = 12,
          window_sign = 9)
pvo = pvo
pvo = pvo.to_frame()


#Directional Index
adx = ADXIndicator(high = stock.high,
                   low = stock.low,
                   close = stock.close,
                   window = 14)
adx_neg = adx.adx_neg()
adx_pos = adx.adx_pos()
adx = adx.adx()
adx_neg = adx_neg.to_frame()
adx_pos = adx_pos.to_frame()
adx = adx.to_frame()

#Daily Log Return
logret = DailyLogReturnIndicator(close = stock.close)
logret = logret.daily_log_return()

#Commodidty Channel Index
cci = CCIIndicator(high = stock.high,
                   low = stock.low,
                   close = stock.close)
cci = cci.cci()
cci = cci.to_frame()

#True Average Range
tar = AverageTrueRange(high = stock.high,
                       low = stock.low,
                       close = stock.close)
tar = tar.average_true_range()
tar = tar.to_frame()

#Schaff Trend Cycle (STC)
mass = MassIndex(high = stock.high,
                 low = stock.low)
mass = mass.mass_index()
mass = mass.to_frame()

#Detrended Price Oscillator (DPO)
dpo = DPOIndicator(close = stock.close)
dpo = dpo.dpo()
dpo = dpo.to_frame()


# In[118]:


#Merge the indicators & stock price data
stock2 = pd.concat([stock, rsi, awesome, kama, roc,
                    pvo, adx, adx_pos, adx_neg, logret,
                    cci, tar, mass, dpo], axis=1)

#Cleaning the data
pd.set_option('display.max_columns', None)
stock3 = stock2.dropna()
stock3 = stock3.drop(["label", "symbol"], axis = 1)
stock3

#Now the data is cleaned and ready to start modeling
#Since we are going to mainly be focused on single-shot Multistep predictions
#All of the features would be avaible for the inputs


# In[119]:


#Pre-processing:


#Split the data
#No random shuffling
#Data has to be in consecuitive samples 

#Marking the column index
column_indices = {name: i for i, name in enumerate(stock3.columns)}

#Doing a 70/20/10 split
#Train on 70, validation on 20, then test on the final 10
n = len(stock3)
train_df = stock3[0:int(n*0.7)]
val_df = stock3[int(n*0.7):int(n*0.9)]
test_df = stock3[int(n*0.9):]

#Define the number of features for the model (24)
num_features = stock3.shape[1]


# In[120]:


#Normalizing the data for the NN
#Basing all normalizations off the training since
#we will not know the testing std/mean

#Preping the data for the tensorflow windowing functions

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# In[121]:


#Now the data is prepped for the tensorflow functions
#https://www.tensorflow.org/tutorials/structured_data/time_series
get_ipython().run_line_magic('run', 'tf_windowing.ipynb')


# In[122]:


#Tensorflow Functions:

#Referenced from https://www.tensorflow.org

#These functions were developed by TensorFlow
#The functoins are:
#1. Windowing the data into TF datasets
#2. Spliting the windows
#3. Making TF dataset for the models
#4. Plotting the outputs
#5. Feedback loop for AR model
#6. A compile & fit function for each model
    #For this we changed the source code to allow for different optimizers, etc


# In[123]:


#Define the window using the Window Generator Function
#The models will make a set of predictions based on a window of consecuitive samples

#Width = number of timesteps of the input & label windows
#offset = the time offset between them

#For example:
#1. 
#input_width = 30
#label_width = 1
#offset = 30
#Predicts a single day's price 30 days in the future based on previous 30 days

#2.
#input_width = 6
#label_width = 1
#offset = 1
#Predicts a single day's price 1 day in the future based on previous 6 days


#3.
#input_width = 90
#label_width = 30
#offset = 30
#Predicts a 30 day sequence starting 1 day in the future based on previous 90 days

#For our models, it would make the most sense to spread
#things out since we are dealing with stock prices
#and the overall movement is more important than the
#short term fluxuations


# In[124]:


#SINGLE STEP Models:
#Predicts ONE time step in the future based ONLY ON CURRENT CONDITIONS
#Single step models DO NOT take a full windodw of data into account

#For our purposes of predicting future pricing movement, these models
#are pretty useless.


# In[125]:


#LSTM Model1
#This will be a basic model that is trained on 30 days & returns ONE prediction in the future
#Return_sequences=True allows the model to be trained on the 30 days at a time

#Since this model only predicts ONE DAY in the future, it is not that useful
#But this helps build a foundation of understanding

#So based on the window set below:
#We will predict ONE future day at a time, for a 30 day sequence, based on the previous 30 days.

wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    label_columns=['close'])

wide_window
wide_window.plot()


# In[126]:


##LSTM Model1 Part 2: Defining the Model
lstm_model1 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

MAX_EPOCHS = 50

history = compile_and_fit_adam(lstm_model1, wide_window)

#To Record the predictions
val_performance= {}
performance = {}
IPython.display.clear_output()
val_performance['LSTM1'] = lstm_model1.evaluate(wide_window.val)
performance['LSTM1'] = lstm_model1.evaluate(wide_window.test, verbose=0)
wide_window.plot(lstm_model1)


# In[127]:


#We can see the predictions are very close to the labels. 
#But,
#This is for one time step at a time
#Meaning only ONE FUTURE day was predicted at a time

#So given the nature of the goal in mind, 
#This is not very useful in gaining insight on the future movement of stock prices

#Because of this, we will now move on to Muti-Step models, 
#Which have much more value with respect to our goal of predicting future price movement


# In[ ]:


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################


# In[128]:


#Multi step models

#Multistep Models
    #Learns & predicts a RANGE/SEQUENCE of future values
    #Unlike a single-step model, which only predicts one step at a time
    
    #For MUTLI-STEP MODELS
#There are two approaches
    #Single Shot - Entire Future sequence is predicted at once
    #AutoRegressive - Models makes single step predictions which are fed back as input to the model    


# In[129]:


#Model inputs
#These are bascially the same as defined above

#Except in this case, the OUT_STEPS
#Is applied to both the label_width & shift inputs
#This is what make the model a single shot multistep model

#input_width = The number of days to base the prediction off of
#label_width/shift = Range/Sequence of days to predict in the future


# In[132]:


#Single Shot Multiple Step Predictions

#SINGLE SHOT PREDICTIONS
#THE 90 DAYS ARE USED ALL AT ONCE TO PREDICT THE NEXT 30
#In a "single shot",  the model will predict a future sequence

MAX_EPOCHS = 50
OUT_STEPS = 30
multi_window = WindowGenerator(input_width=120,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                              label_columns=['close'])

multi_window.plot()
multi_window


# In[133]:


#Now to develope a "baseline"
#Which only repeats the previous's price

#This is where the model literal just continously predicts the last value
#Not very good, as you would expect

class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Baseline'] = last_baseline.evaluate(multi_window.val)
multi_performance['Baseline'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(last_baseline)


# In[134]:


#As one would expect, the models are not very good
#They do not do much learning, but this is used as a baseline
#for the future predctions (mean_absolute_error: 3.4402)
#Below we will start on the actual models that have usage


# In[135]:


#Single Shot Multi Step Model
#Note:
#Return_sequences is set to false because...
#In a single shot format, the model onlt needs to produce an output  at the last timestep
#For instance
#inputs = 1:90
#outputs = 91:120
#No need to return the sequences to the model since it is only making ONE 30-day prediction
#If return sequences was TRUE, then it would train on 1:90 -> make prediction, 2:91->make prediction & so on


# In[136]:


#Model 1
multi_lstm_model1 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
                         return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features), #Default Dense Layer
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

random.seed(2112)
history = compile_and_fit_adam(multi_lstm_model1, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_1'] = multi_lstm_model1.evaluate(multi_window.val)
multi_performance['LSTM_1'] = multi_lstm_model1.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model1)


# In[137]:


#The default model started out all over the place.
#After adjusting the LSTM Layer Units, the predictions got much better
#Started at 32 & work our way up to 256

#The unit size


# In[138]:


multi_lstm_model2 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh", #Default, has to be TANH
                         kernel_initializer="VarianceScaling",
                         recurrent_initializer="orthogonal",
                         return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.VarianceScaling()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit_adam(multi_lstm_model2, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_2'] = multi_lstm_model2.evaluate(multi_window.val)
multi_performance['LSTM_2'] = multi_lstm_model2.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model2)


# In[139]:


#For this iteration we experimented with a few different parameters
#We changed the initializers for the LSTM layer
#Also learned that leaving the Dense Layer's activation function to "linear"(none) worked best


# In[140]:


multi_lstm_model3 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh", #Default, has to be TANH
                         kernel_initializer="VarianceScaling",
                         recurrent_initializer="orthogonal",
                         return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.Constant()),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.VarianceScaling()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit_adam(multi_lstm_model3, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_3'] = multi_lstm_model3.evaluate(multi_window.val)
multi_performance['LSTM_3'] = multi_lstm_model3.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model3)


# In[141]:


multi_lstm_model4 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh", #Default, has to be TANH
                         kernel_initializer="VarianceScaling",
                         recurrent_initializer="orthogonal",
                         return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.RandomUniform()),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.VarianceScaling()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit_adam(multi_lstm_model4, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_4'] = multi_lstm_model4.evaluate(multi_window.val)
multi_performance['LSTM_4'] = multi_lstm_model4.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model4)


# In[142]:


multi_lstm_model5 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh", #Default, has to be TANH
                         kernel_initializer="VarianceScaling",
                         recurrent_initializer="TruncatedNormal",
                         return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.RandomUniform()),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.VarianceScaling()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit_adam(multi_lstm_model5, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_5'] = multi_lstm_model5.evaluate(multi_window.val)
multi_performance['LSTM_5'] = multi_lstm_model5.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model5)


# In[148]:


multi_lstm_model6 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh", #Default, has to be TANH
                         kernel_initializer="VarianceScaling",
                         recurrent_initializer="TruncatedNormal",
                         return_sequences=False),
    tf.keras.layers.Dropout(rate = 0.01),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.RandomUniform()),
    tf.keras.layers.Dropout(rate = 0.001),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.VarianceScaling()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit_adam(multi_lstm_model6, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_6'] = multi_lstm_model6.evaluate(multi_window.val)
multi_performance['LSTM_6'] = multi_lstm_model6.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model6)


# In[154]:


multi_lstm_model7 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh", #Default, has to be TANH
                         kernel_initializer="VarianceScaling",
                         recurrent_initializer="TruncatedNormal",
                         return_sequences=False),
    tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.RandomUniform()),
    tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          activation = None,
                          kernel_initializer=tf.initializers.VarianceScaling()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit_adam(multi_lstm_model7, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_7'] = multi_lstm_model7.evaluate(multi_window.val)
multi_performance['LSTM_7'] = multi_lstm_model7.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model7)


# In[155]:


#Results:
single_shot_results = pd.DataFrame(multi_val_performance)
single_shot_results = single_shot_results.transpose()
single_shot_results.columns = ["Val Loss", "Val MAE"]
single_shot_results


# In[145]:


#Results 2:


# In[146]:


#We can look at the MAE and see which model was comparitively the most accurate.
#But we can also look at the plotted predictions to see how the models behaved.
#This will allows us to make a more informed decision rather than basing our
#approach off of the MAE


# In[147]:


#Tomorrow making closing remarks about which one is the best
#And make a summary about lstm models and the varaible choices and thought processes
#what is an lstm & how do they work - what are they used for
#what is the structure of an lstm and how do the models flow - what are they made of
#Which model is the best according to MAE, which model is the best according to you
#which model looks like it would perform best considering the goal of predicting price movement?

