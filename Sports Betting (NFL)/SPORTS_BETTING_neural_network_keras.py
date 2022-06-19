#Fav Won
#Zak Massey
#Reference:
#Keras Neural Network

----------------------------------------------------------------------

 #This project used multiple online resources
  #References:
  #Keras Documentation
  #Tensorflow Documentation
  #StackExchange
  #Github
  
 ----------------------------------------------------------------------
  
  
#Load libraries
import numpy as np               
import pandas as pd         
from sklearn import preprocessing 
import keras
import tensorflow as tf
from keras.layers import Dense    
from keras.layers import BatchNormalization 
from keras.layers import Dropout        
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras import regularizers  
from keras.wrappers.scikit_learn import KerasRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner import Objective
from keras_tuner import HyperModel
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from sklearn.utils import compute_class_weight
from keras import metrics
import sklearn
import matplotlib.pyplot as plot

----------------------------------------------------------------------

#Import Data
nfl_data = pd.read_csv("filename")
nfl_data.head()


#Remove Variables
nfl_data.drop(["home_rookie_coach", "away_rookie_coach", "home_ovr_avg_att",
              "away_ovr_avg_att", "home_team_overall", "away_team_overall", "team_home",
              "team_away", "team_favorite_id", "score_home", "score_away", "SuperBowl", 
              "over", "dog_covered", "fav_covered", "winner_yards", "loser_yards", "winner_turnover", "loser_turnovers",
               "away_fav", "winner_turnover", "fav_won6", "idx", "dog", "winner", "loser", "margin_of_victory"],
              axis=1, inplace=True)

#View the Data
#Set display to see all columns with 4 decimal places
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 4)
nfl_data.head()


#Shuffle the dataset
nfl_data = nfl_data.sample(frac=1)
nfl_data.head()


#Split the Dataset into test and train

#Create the training and testing data
#Splitting the data 90/10
import random
random.seed(2112)
train, test = train_test_split(nfl_data, test_size=0.10)

----------------------------------------------------------------------

#Define the target and predictors
labels = np.array(train['fav_won'])
features = train.drop('fav_won', axis = 1)

#Store a list of the predictors
feature_list = list(features.columns)
features = np.array(features)

----------------------------------------------------------------------

#Variable Reduction

#Define the rf model (variable importance)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 500)

import random
random.seed(2112)

#Train the model
rf.fit(features, labels);

----------------------------------------------------------------------

#Variable selection & importance

#Gini/Mean Decrease of Impurity Importance

#Extract the feature importance based on Mean Decrease of Impurity
#Decrease of Impurity == Gain of Purity
#AKA Gini Importance (Node Impurity)

#Pull importance list from the rf object
importances = list(rf.feature_importances_)

#List variable and importance
#Four decimal places rounded
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]

#Order by importance
#key = lambda x: x[1] -> selected the column to be sorted by (Col1 = Importance)
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

#View features and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#Extract the variables selected

#Define the new list
varimp = pd.DataFrame(feature_importances, columns=['Variable','Importance'])

#Select variables which have an importance above 0.005
#(The impurity-based feature importances)
#Mean Decrease of Impurity
#If we remove LOGspread_favorite, the mean IMPURITY would decrease by 0.0703 etc  
varimp_rf_gini = varimp[varimp.Importance > 0.005]
varimp_rf_gini_features = list(varimp_rf_gini["Variable"])
print(varimp_rf_gini_features)


#Permutation Based Feature Importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rf, features, labels)


#Extract the importance from the perm_importance object
varimp_perm = pd.DataFrame(perm_importance.importances_mean, columns=['Importance'], index = feature_list)
varimp_perm["Variable"] = (varimp_perm.index)
#Sort the values
varimp_perm.sort_values("Importance", ascending = False)


#Filter out unimportant variables & sort accordingly
varimp_perm = varimp_perm[varimp_perm.Importance > 0.005]
varimp_perm.sort_values("Importance", ascending = False)


#Create a list of the selected variables
varimp_rf_perm_features = list(varimp_perm["Variable"])
print(varimp_rf_perm_features)


varimp_perm.sort_values("Importance", ascending=False)

----------------------------------------------------------------------

#Combine the lists & remove the duplicates
selected_vars = list(varimp_rf_gini_features+varimp_rf_perm_features)
selected_vars2 = list()
for item in selected_vars:
    if item not in selected_vars2:
        selected_vars2.append(item)
print(selected_vars2) 


#Reduce the dataset
#Keep columns selected from Gini/Perm importance - along with the target

#Train
train2 = pd.DataFrame(train, columns = selected_vars2)
train2["fav_won"] = train["fav_won"]
train2

#Test
test2 = pd.DataFrame(test, columns = selected_vars2)
test2["fav_won"] = test["fav_won"]
test2

----------------------------------------------------------------------

#Define inputs/target
x_train = train2.drop("fav_won", axis = 1).values
y_train = train2["fav_won"].values
x_test = test2.drop("fav_won", axis = 1).values
y_test = test2["fav_won"].values

#Shape
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

----------------------------------------------------------------------

#Standardizing the data --- look at this - only training data was scaled?
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)
x_train = pd.DataFrame(data=x_train)
x_test = pd.DataFrame(data=x_test)

----------------------------------------------------------------------

#Define the model
def build_model(hyperparams):   #Define the model
        model = Sequential()       #Start the model
        model.add(layers.Input(shape=(x_train.shape[1],)))  #Add the input layer/shape (x dimenstions)
        model.add(layers.Dense(units=hyperparams.Int("units_l1", 64, 256, step=16), #First layer units
                           use_bias=hyperparams.Boolean("bias_l1"),  #Make weight bias tunable
                           activation=hyperparams.Choice("act_l1", ["tanh", "LeakyReLU", "relu", "swish"]), #Activation function
                           kernel_initializer=hyperparams.Choice("kern_l1", ["GlorotUniform", "HeUniform", "he_normal"])
                          ))
        model.add(layers.Dense(units=hyperparams.Int("units_l2", 64, 256, step=16), #Second layer units
                           use_bias=hyperparams.Boolean("bias_l2"), #Make weight bias tunable
                           activation=hyperparams.Choice("act_l2", ["tanh", "LeakyReLU", "relu", "swish"]), #Activation function
                           kernel_initializer=hyperparams.Choice("kern_l2", ["GlorotUniform", "HeUniform", "he_normal"])
                          ))
        model.add(layers.Dense(units=hyperparams.Int("units_l3", 8, 128, step=8), #Second layer units
                           use_bias=hyperparams.Boolean("bias_l3"), #Make weight bias tunable
                           activation=hyperparams.Choice("act_l3", ["tanh", "LeakyReLU", "relu", "swish"]), #Activation function
                           kernel_initializer=hyperparams.Choice("kern_l3", ["GlorotUniform", "HeUniform", "he_normal"])
                          ))
        model.add(Dense(1, activation='sigmoid'))
        #optim=tf.keras.optimizers.SGD(learning_rate = hyperparams.Float("learning_rate",0.00001,0.0001,step=0.01),
        #                          momentum = hyperparams.Float("momentum",0.01,0.1,step=0.01))
        optim=tf.keras.optimizers.Adagrad(learning_rate = hyperparams.Float("learning_rate",0.0001,0.001),
                                         epsilon = 1e-05)#Brought epsilon up from 1e-07 to make the learning rate smaller

        model.compile(optim, loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])

        return model
    
----------------------------------------------------------------------

#Define the Tuner
#Using Bayesian Optimization Tuning
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model, #Model referenced above
    objective="val_loss",
    max_trials = 200, #number of combinations from tuning parameters
    beta = 3.35, #Increased this to have more exploration since we increased epsilon, which slowed the learning rate
    project_name="tuner1",
    overwrite=True)


#Balance class weights
from sklearn.utils.class_weight import compute_class_weight
classWeight = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train),  y= y_train)
classWeight = dict(enumerate(classWeight))
classWeight



#Define final tuning options

#Early stopping if val_loss does not improve (get smaller) for 20 iterations
earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)


#Start the tuning
tuner.search(x_train,
              y_train,
              batch_size=102,
              epochs=100, 
              validation_data=(x_test, y_test),
              class_weight=classWeight,
              callbacks =[earlystopping])

----------------------------------------------------------------------

#Look at the tuned parameters
best_params2 = tuner.get_best_hyperparameters()
best_params2[0].values


#Apply to the best model
best_model = tuner.get_best_models()[0]
best_model.summary()

----------------------------------------------------------------------

#Predict using the best_model & the Testing Data Inputs
random.seed(2112)
preds = best_model.predict(x_test)
#Round the predictions so we can make a confusion matrix
preds2 = pd.DataFrame(data = preds, columns = ['Pred'])
y_test2 = pd.DataFrame(data = y_test, columns = ["Truth"])
output = pd.concat([preds2, y_test2], axis = 1)
output['Pred2'] = output.apply(lambda row: round(row.Pred), axis = 1) 

----------------------------------------------------------------------

#Confusion Matrix with standard probability predictions (0.5 threshold)
confusion_matrix1 = sklearn.metrics.confusion_matrix(output.Pred2, output.Truth)
print(confusion_matrix1)
print("Accuracy %:")
print((70+37)/(70+37+18+34)*100)

----------------------------------------------------------------------

#Using Classification Thresholds

#Create the 2 outcome probability variables
#159 obs - entire testing dataset
output["Prob1"] = output.Pred
output["Prob2"] = 1-output.Pred
output


#Create the probability threshold
#87 Obs - Only kept obs with predictions above the 0.6 threshold
idx = (output['Prob1'] > 0.61) | (output['Prob2'] > 0.61)
output2 = output[idx]
output2


#Fill out the threshold based predictions
#The obs were already filtered out to only keep >0.6(1) preds for either 0,1
#So we can only use one column as a reference
output2['Prob_pred'] = np.where(output2['Prob1'] > 0.61,1, 0)  
output2


#Threshold: 0.6
confusion_matrix2 = sklearn.metrics.confusion_matrix(output2.Prob_pred, output2.Truth)
confusion_matrix2


#Threshold: 0.61
confusion_matrix3 = sklearn.metrics.confusion_matrix(output2.Prob_pred, output2.Truth)
confusion_matrix3


#View the output
print("Confusion Matrix (0.6 Threshold)")
print(confusion_matrix2)
print("Accuracy % (0.6 Threshold):")
print((18+45)/(45+18+5+13)*100)
print("Bettable Instances:")
print(81)
print("Percent of Bettable Instances:")
print((81/159)*100)
print("77.77% accuracy with a decision threshold of 0.6, able to bet on 50.74% of games")


print("Confusion Matrix (0.61 Threshold)")
print(confusion_matrix3)
print("Accuracy % (0.61 Threshold):")
print((60/76)*100)
print("Bettable Instances:")
print(76)
print("Percent of Bettable Instances:")
print((76/159)*100)
print("78.94% accuracy with a decision threshold of 0.61, able to bet on 47.79% of games")

----------------------------------------------------------------------

#SAVE THE MODEL TO JSON FORMAT
# serialize model to JSON
model_json = best_model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("model2.h5")
print("Saved model to disk")


#Save the testing and training data
x_train_saved = pd.DataFrame(x_train)
y_train_saved = pd.DataFrame(y_train)


x_test_saved = pd.DataFrame(x_test)
y_test_saved = pd.DataFrame(y_test)

training = pd.concat([x_train_saved, y_train_saved], axis=1, join='inner')
testing = pd.concat([x_test_saved, y_test_saved], axis=1, join='inner')

training.to_csv("filename")
testing.to_csv("filename")

----------------------------------------------------------------------

#Later... To load the model & environment back to the notebook...


#Load the model back

#Load the model back;
from keras.models import model_from_json
# load json and create model
json_file = open("filename", 'r')
loaded_model_json = json_file.read()
json_file.close()
best_model = model_from_json(loaded_model_json)
# load weights into new model
best_model.load_weights("filename")
print("Loaded model from disk")


#Load the environment back

#Load data
training = pd.read_csv("filename")
testing = pd.read_csv("filename")

#Drop the column that populated as an index
training.drop(training.columns[[0]], axis=1, inplace=True)
testing.drop(testing.columns[[0]], axis=1, inplace=True)

#Define inputs/target
x_train = training.drop("fav_won", axis = 1).values
y_train = training["fav_won"].values
x_test = testing.drop("fav_won", axis = 1).values
y_test = testing["fav_won"].values

----------------------------------------------------------------------

#Model Interpretations


#Permutation Based Importance
#Top Variables

varimp_perm = varimp_perm.sort_values("Importance", ascending=False)
varimp_perm = pd.DataFrame(varimp_perm)

varimp_perm15 = varimp_perm.loc[varimp_perm["Importance"] > 0.03]
varimp_perm15

varimp_perm15.plot.bar(x = "Variable", y = "Importance")


#Model interpretation explained in the Keras Summary
