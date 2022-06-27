#LLY Stock Price Forecasting with LSTMs
#Zak Massey

------------------------------------------------------------------------------------
  
  
#This project used multiple online resources
#Listed in the summary file

#Import libraries
....

#Adding technical indicators libraries
...


#Project Goal: Predict Stock Price Movement
#First we retrive stock prive data on LLY
#Then we caluate technical inidators which help identify movement
#Then we preprocess the data for our models
#First we make a Single-Step LSTM
#Then we move to a Single-Shot Multi-Step LSTM
#We then evaluate the performance & deploy the model for the next months predictions


------------------------------------------------------------------------------------


#Set the API key
settings.set_apikey('Your API Key')

#Pull stock price data "lly"
stock = sts.historical_stock_data('LLY', dailytype = 'change', start = '2015-04-15',
                                  end = '2022-06-20')
#Need to reverse the order
stock = stock[::-1] 
stock.close


#Look at a plot
plt.plot(stock.close) #or plt.plot(lly["close"])


------------------------------------------------------------------------------------


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


------------------------------------------------------------------------------------


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


------------------------------------------------------------------------------------


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


------------------------------------------------------------------------------------


#Normalizing the data for the NN
#Basing all normalizations off the training since
#we will not know the testing std/mean

#Preping the data for the tensorflow windowing functions

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


#Now the data is prepped for the tensorflow functions
#https://www.tensorflow.org/tutorials/structured_data/time_series
#sourced out the functions from another scrpit
get_ipython().run_line_magic('run', 'tf_windowing.ipynb')



#Tensorflow Functions:

#Referenced from https://www.tensorflow.org

#These functions were developed by TensorFlow
#The functoins are:
#1. Windowing the data into TF datasets
#2. A compile & fit function for each model
    #For this we changed the source code to allow for different optimizers, etc
#3. Plotting the outputs

------------------------------------------------------------------------------------

#Notes on the functions:

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


#4. Plotting the outputs

#SINGLE STEP Models:
#Predicts ONE time step in the future based ONLY ON CURRENT CONDITIONS
#Single step models DO NOT take a full windodw of data into account

#For our purposes of predicting future pricing movement, these models
#are pretty useless.


------------------------------------------------------------------------------------


#LSTM Model1 (Single Step)
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

------------------------------------------------------------------------------------

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

#***See summary for output and explainations

------------------------------------------------------------------------------------


#Multi step models

#Multistep Models
    #Learns & predicts a RANGE/SEQUENCE of future values
    #Unlike a single-step model, which only predicts one step at a time
    
    #For MUTLI-STEP MODELS
#There are two approaches
    #Single Shot - Entire Future sequence is predicted at once
    #AutoRegressive - Models makes single step predictions which are fed back as input to the model    


------------------------------------------------------------------------------------


#Single Shot Multiple Step Predictions

#SINGLE SHOT PREDICTIONS
#THE 120 DAYS ARE USED ALL AT ONCE TO PREDICT THE NEXT 30
#In a "single shot",  the model will predict a future sequence

MAX_EPOCHS = 50
OUT_STEPS = 30
multi_window = WindowGenerator(input_width=120,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                              label_columns=['close'])

multi_window.plot()
multi_window


------------------------------------------------------------------------------------

#Baseline Model

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


------------------------------------------------------------------------------------

#Single Shot Multi Step Model

#The following is quite repetitive.
#It is just the adhoc experimentation mentioned in the summary
#Chaning parameters/adding layers etc.

#Model 1
multi_lstm_model1 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
                         return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

random.seed(2112)
history = compile_and_fit_adam(multi_lstm_model1, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM_1'] = multi_lstm_model1.evaluate(multi_window.val)
multi_performance['LSTM_1'] = multi_lstm_model1.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model1)


------------------------------------------------------------------------------------


multi_lstm_model2 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
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


------------------------------------------------------------------------------------


multi_lstm_model3 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
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

------------------------------------------------------------------------------------

multi_lstm_model4 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
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


------------------------------------------------------------------------------------


multi_lstm_model5 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
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


------------------------------------------------------------------------------------


multi_lstm_model6 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
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

------------------------------------------------------------------------------------


multi_lstm_model7 = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256,
                         activation="tanh",
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


------------------------------------------------------------------------------------

#Results:
single_shot_results = pd.DataFrame(multi_val_performance)
single_shot_results = single_shot_results.transpose()
single_shot_results.columns = ["Val Loss", "Val MAE"]
single_shot_results


