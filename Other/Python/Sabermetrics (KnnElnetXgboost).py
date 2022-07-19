#Sabermetrics is the empirical analysis of baseball statistics
#The goal of this projects is to use this approach and devlop
#ML models which can accurately predict a players runs per game
#We can use the model devloped for scouting purposes
#What features impact a players runs per game?


------------------------------------------------------------------

#Import Required Starting Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pylab
import scipy.stats as stats
import random
import sklearn
from sklearn.model_selection import train_test_split

------------------------------------------------------------------

#Import Data
baseball = pd.read_csv("your/file/here")
baseball.drop(["Pos Summary", "Name-additional", "Rk", "szn", "Tm", "Lg"], axis=1, inplace=True)

#Clean the data
#Drop values that don't meet critera & duplicates
baseball = baseball.drop_duplicates()

#Dropping NA values & Name columns
baseball = baseball.dropna()
baseball.drop(["Name"], axis = 1, inplace=True)
baseball.drop(["TB"], axis = 1, inplace=True)
baseball.drop(["PA"], axis = 1, inplace=True)
baseball.drop(["AB"], axis = 1, inplace=True)

#We could create a new target variable
baseball["runs_per_game"] = baseball.R/baseball.G

#Shuffle the dataset
baseball = baseball.sample(frac=1)

#View the data
baseball

------------------------------------------------------------------

#We can split it into test/train/validation
random.seed(2112)
train, test = train_test_split(baseball, test_size=0.15)
test, validation = train_test_split(test, test_size=0.10)
#Train = 5378 
#Test = 855 
#Validation = 95 

#Look for outliers in the training data
#Probably should drop the outliers
sns.boxplot(train.runs_per_game)


------------------------------------------------------------------

#Verify the outliers

#To verify there are outliers according to the zscore
train["z"] = np.abs(stats.zscore(train.runs_per_game))

#We can see all the values that fall outside the z score range
#This is good because most of them only have 1-3 games played, which is not the best sample
train[train.z > 3]


------------------------------------------------------------------

#Dropping the vales above
train = train[train.z < 3]
train = pd.DataFrame(train)


#Drop the Runs and Zscore columns
train.drop(["z", "R"], axis = 1, inplace=True)
test.drop(["R"], axis = 1, inplace=True)
validation.drop(["R"], axis = 1, inplace=True)


#We can also set a minimum number of games played for each player
#Because a small amount of games could skew the runs/game target
#There are 162 games are played over a 6-month season
#Requirement to play at least 20% of the season


#Drop all obs who played less than 20 games
train = train[train["G"]> 20]

#Drop the games variable
train.drop(["G"], axis = 1, inplace=True)
test.drop(["G"], axis = 1, inplace=True)
validation.drop(["G"], axis = 1, inplace=True)

------------------------------------------------------------------

#View distributions

fig1 = train.hist()
fig1 = fig1.figure
fig1.set_size_inches(8, 10)

#Viewing the ditributions
train.hist()
train.hist(["H", "2B", "HR", "SLG"])

------------------------------------------------------------------

#Test if data has a normal distribution
from scipy.stats import jarque_bera
from scipy.stats import shapiro

result_jb = (jarque_bera(train['runs_per_game']))
result_sh = (shapiro(train['runs_per_game']))
print(result_jb)
print(result_sh)

#Null is that there is no difference from normally distributed data
#We can reject that based on the pval
#Conclude that the data is not normally distributed


------------------------------------------------------------------

#Normalizing the data

#For out first model - KNN it is not necessary to transform the data in any way, because it is a non parametic model
#But for the subsquent models we would want to transform the data. 
#So we are just gonna do it now so we can keep the labels
#Since we are going to do this all in the same script

#Pull the training data mean/std/min/max
train_mean = train.mean()
train_std = train.std()
train_min = train.min()
train_max = train.max()


#Normalize the train/testing/validation based off the training features
train2 = ((train - train_min)/(train_max-train_min))+1
test2 = ((test - train_min)/(train_max-train_min))+1
validation2 = ((validation - train_min)/(train_max-train_min))+1


#Using the log transofmation
train2 = pd.DataFrame((np.log(train2)))
test2 = pd.DataFrame((np.log(test2)))
validation2 = pd.DataFrame((np.log(validation2)))


#View new distributions after the transformation
train2.columns = train.columns
train2.hist()


#Rename the columns
test2.columns = test.columns
validation2.columns = validation.columns


#Need to define the labels
#Define inputs/target
x_train = train2.drop("runs_per_game", axis = 1).values
y_train = train2["runs_per_game"].values
x_test= test2.drop("runs_per_game", axis = 1).values
y_test = test2["runs_per_game"].values
x_val= validation2.drop("runs_per_game", axis = 1).values
y_val = validation2["runs_per_game"].values


------------------------------------------------------------------


#Modeling

#KKN model 1
#Running a for loop to pass all the Ks defined in our range
#Recording the score so we can visualize what K is best
#Almost like gridsearch, but is in a for loop since it is only one feature

from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

k_range = range(1,51)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.mean_absolute_error(y_test, y_pred)
    scores_list.append(metrics.mean_absolute_error(y_test, y_pred))

#Plotting the results
plt.plot(k_range, scores_list)
plt.xlabel("K")
plt.ylabel("MAE")


------------------------------------------------------------------


#Checking for optimal K based on a different metric (R2)

#KNN Model 2
k_range = range(1,51)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.r2_score(y_test, y_pred)
    scores_list.append(metrics.r2_score(y_test, y_pred))

plt.plot(k_range, scores_list)
plt.xlabel("K")
plt.ylabel("R2")


------------------------------------------------------------------


#We see K = 10 got the lowest R2/MAE on the testing data
#Fit on the training data
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)

#Make predictions on the validation dataset using the model above
y_pred = knn.predict(x_val)

#Evaulate the models performance via MSE
def mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean()

mse(y_val, y_pred)

------------------------------------------------------------------

#KNN Model Tuning:

#Setting parameters to tune
leaf_size = list(range(1,50))
n_neighbors = list(range(5,25))
p=[1,2]
#Weights is going to stay uniform
#Algorithim will stay auto
#Metric is the default minkowski metric

#Set params, define the model and tune
from sklearn.model_selection import GridSearchCV
hp = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
knn2 = KNeighborsRegressor()

#Use a Gridsearch to tune
knn_tuned = GridSearchCV(knn2, hp, cv=10)

#Fit the tuned model & view parameters
best_model = knn_tuned.fit(x_train, y_train)
best_model.best_estimator_.get_params()


#Apply the tuned parameters
#Train the model
#Then predict on the validation data
knn_tuned2 = KNeighborsRegressor(algorithm='auto', leaf_size = 1,
                            metric = 'minkowski', n_neighbors = 14,
                            p = 1, weights = 'uniform')

knn_tuned2.fit(x_train, y_train)

#Make predictions on the validation dataset
y_pred_knn = knn_tuned2.predict(x_val)

#View Results
mse(y_val, y_pred_knn)


------------------------------------------------------------------


#Transform the validation data & predictions back to their true values
#So we can compare actual & predicted values based on the true values
#(train2-1)*(train_max-train_min)+train_min

#Inverse Log
y_val2 = pd.DataFrame(np.exp(y_val))
y_pred_knn2 = pd.DataFrame(np.exp(y_pred_knn))

#Inverse Normalization
y_val2_actual = (y_val2-1)*(train_max["runs_per_game"]-train_min["runs_per_game"])+train_min["runs_per_game"]
y_pred_knn3 = (y_pred_knn2-1)*(train_max["runs_per_game"]-train_min["runs_per_game"])+train_min["runs_per_game"]
results = pd.concat([y_val2_actual, y_pred_knn3], axis = 1)

#Look at the true values and the predicted values (actual)
pd.set_option('display.max_rows', None)
results

#Verify the reverse transformations were correct.
#validation["runs_per_game"]


------------------------------------------------------------------


#Next Model: Elastic Net
#See summary for explainations of the tuning
#We are going straight into the L1/L2 tuning for this model


#Elastic Net Model
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

#Building the model
#Define parameters to tune
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
l1_ratio = np.arange(0.0, 1.0, 0.01)
alpha =alpha
#Set the parameters
hp = dict(alpha=alpha, l1_ratio=l1_ratio)
#Set model
elastic_net = ElasticNet()
#Set resampling
cv = RepeatedKFold(n_splits=10, n_repeats=3)

#Set gridsearch as the tuning method
elnet = GridSearchCV(elastic_net, hp, cv=cv, n_jobs=-1)

#Tune/fit the model on the training data
elnet_tuned = elnet.fit(x_train, y_train)

#View the best parameters
elnet_tuned.best_params_


------------------------------------------------------------------


#Set the tuned parameters
#Fit the model with tuned paramters
#Elastic Net from sklearn already implemented coordinate descent
best_model_elnet = ElasticNet(alpha = 0.0001, l1_ratio=0.0)
best_model_elnet2 = best_model_elnet.fit(x_train, y_train)

#Make predictions on the validation dataset
y_pred_elnet = best_model_elnet2.predict(x_val)

#View the results
mse(y_val, y_pred_elnet)


------------------------------------------------------------------


#Transform the validation data & predictions back to their true values
y_val2 = pd.DataFrame(np.exp(y_val))
y_pred_elnet2 = pd.DataFrame(np.exp(y_pred_elnet))
y_val2_actual = (y_val2-1)*(train_max["runs_per_game"]-train_min["runs_per_game"])+train_min["runs_per_game"]
y_pred_elnet3 = (y_pred_elnet2-1)*(train_max["runs_per_game"]-train_min["runs_per_game"])+train_min["runs_per_game"]
results = pd.concat([y_val2_actual, y_pred_elnet3], axis = 1)
results

#Look at the Coefficients
print(best_model_elnet2.intercept_, best_model_elnet2.coef_)

#Transform the output so we can interpret model
#Transform so we can make graphs and tables
features = pd.DataFrame(validation.columns[0:24])
coefs = pd.DataFrame(best_model_elnet2.coef_)
res = pd.concat([features, coefs], axis = 1)
res.columns = ["Feature", "Coefficient"]
intercept = best_model_elnet2.intercept_


------------------------------------------------------------------


#Coefficicent table
import tabulate
from tabulate import tabulate
res2 = res
res2.loc[len(res2.index)] = ["Intercept",intercept]
print(tabulate(res2, headers=res2.columns, tablefmt="grid"))



#Make the coefficient graph
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.bar(res["Feature"],res["Coefficient"])
plt.xticks(rotation=45)
plt.title('Elastic Net Coefficients')
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.show()


------------------------------------------------------------------


#Model 3: XGBoost
import xgboost as xgb

#Need to make the data into the xgbmatrix objects

#Training
train_features = train2.loc[:,"Age":"IBB"]
train_label = train2["runs_per_game"]
dtrain = xgb.DMatrix(data = train_features, label = train_label)
#Testing
test_features = test2.loc[:,"Age":"IBB"]
test_label = test2["runs_per_game"]
dtest = xgb.DMatrix(data = test_features, label = test_label)
#Validation
val_features = validation2.loc[:,"Age":"IBB"]
val_label = validation2["runs_per_game"]
dval = xgb.DMatrix(data = val_features, label = val_label)


#Fitting a standard model
xgb_model = xgb.XGBRegressor(objective="reg:linear")
xgb_model.fit(X=train_features, y = train_label)
xgb_preds = xgb_model.predict(val_features)
mse(val_label, xgb_preds)

------------------------------------------------------------------


#Tuning the XGB model
#Using Randomsearch for this tuning since Gridsearch was used in the previous 2 models

#Set the tuning parameters
from sklearn.model_selection import RandomizedSearchCV
#Using random search to get an idea of where parameters need to be set
#Then we will probably go back and run a more focuesed gridsearch

#Set the model
xgb_model = xgb.XGBRegressor()

#Dont need to include L1 regularization because
#1. There is not high dimensionalty
#Define the parameters
parameters = {
    'max_depth': [*range(6, 12, 1)],
    'learning_rate': np.linspace(0.01,1.0,11),
    'gamma': np.linspace(0,1.0,11),
    'colsample_bytree': np.linspace(0.5,1.0,6),
    'lambda': np.linspace(0.5,1.0,6)
}


xgb_rand = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=parameters,
    n_iter = 500,
    scoring = 'r2',
    cv = 10,
    verbose=True
)

#Tune the model on the training data:
xgb_tuned = xgb_rand.fit(X=train_features, y = train_label)

#View the tuned parameters
xgb_tuned.best_params_
xgb_tuned.set_params = xgb_tuned.best_params_

#Predict with the tuned model:
xgb_preds = xgb_tuned.predict(val_features)
mse(val_label, xgb_preds)


------------------------------------------------------------------


#View the final predictions on validation data
y_val2 = pd.DataFrame(np.exp(y_val))
y_pred_xbg2 = pd.DataFrame(np.exp(xgb_preds))

y_val2_actual = (y_val2-1)*(train_max["runs_per_game"]-train_min["runs_per_game"])+train_min["runs_per_game"]
y_pred_xbg3 = (y_pred_xbg2-1)*(train_max["runs_per_game"]-train_min["runs_per_game"])+train_min["runs_per_game"]
results = pd.concat([y_val2_actual, y_pred_xbg3], axis = 1)
results


------------------------------------------------------------------


#Formatting a model in a format we can interpret better
#(Same model, just differnt format)

#Get the model in a format that we can use
xgb_regr = xgb.train(params=xgb_tuned.best_params_, dtrain=dtrain)

#Predict with the tuned model:
xgb_preds = xgb_regr.predict(dval)
mse(val_label, xgb_preds)


------------------------------------------------------------------


#Model Interpretations:

#Variable Importance
varimp1 = xgb.plot_importance(xgb_regr)
varimp1 = varimp1.figure
varimp1.set_size_inches(8, 10)

varimp2 = xgb.plot_importance(xgb_regr, importance_type="gain")
varimp2 = varimp2.figure
varimp2.set_size_inches(8, 10)

#SHAP Plots
import shap
explainer = shap.TreeExplainer(xgb_regr)
shap_values = explainer.shap_values(dtrain)
shap.summary_plot(shap_values, train_features)

#Shap Dependence Plots:
for name in train_features.columns:
    shap.dependence_plot(name, shap_values, train_features)

