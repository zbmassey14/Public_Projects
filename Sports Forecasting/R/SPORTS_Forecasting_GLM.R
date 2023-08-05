#Fav Won
#Zak Massey
#Reference:
#GLM
#Gradient Boosting with Component-wise (Generalized Linear) Models

------------------------------------------------------------------------------------
  

  #This project used multiple online resources
  #References:
  #https://cran.r-project.org/web/packages/mboost/index.html
  #mlr3extralearners.mlr-org.com
  #StackExchange
  #Github  
  
  
#The goal is to predict whether or not the favorite will win an NFL football 
#game & interpret the results for betting insights.
#The data was built out using online betting data, season data, & madden data

#Utilizing a Boosted Generalized Linear Classification Model

#Please find the summary in a separate txt file with the same name.
#All data preprocessing, merging & cleaning was done in a separate script  
  
  
  
#Clear Space
rm(list = ls())
rm(list = ls(all.names = TRUE))
gc()

#Load Packages
library(e1071)
library(caTools)
library(class)
library(parallel)
library(parallelMap)
library(future)
library(dplyr)
library(readr)
library(tictoc)
library(randomForest)
library(caret)
library(purrr)
library(mlr3)
library(mlr3verse)
library(mlr3tuning)
library(mlr3viz)
library(flashlight)
library(caret)
library(iml)
library(DALEX)
library(DALEXtra)
library(ggplot2)

------------------------------------------------------------------------------------
  
  
  #*Loaded Environment & Data*
  

  ------------------------------------------------------------------------------------
  
  
  #Since this data is chronological
  #We need to shuffle it & then split it
nfl_data <- as.data.frame(nfl_data)
rows <- sample(nrow(nfl_data))
nfl_data <- nfl_data[rows, ]
str(nfl_data)
head(nfl_data)



#Split the data into test and train
set.seed(1221)
parts <- createDataPartition(nfl_data$fav_won, p = 0.90, list = F)
train <- nfl_data[parts, ]
test <- nfl_data[-parts, ]
#Check data structure & order
str(train)
str(test)
head(train)
head(test)


------------------------------------------------------------------------------------
  
  #Feature selection for random forest
  #(Data/Dimension Reduction)
  
  #Random Forest variable selection
  set.seed(5414)
tic()
map(c(
  rf_model <- randomForest(train$fav_won~., data = train,
                           mtry = 11, ntree = 500, importance = TRUE,
                           proximity = TRUE)
),  ~Sys.sleep(.x))
toc()
imp <- as.data.frame(round(importance(rf_model), 2))
imp$vars <- rownames(imp)
#Only want to keep if vairbles helps accuracy
imp$x <- ifelse(imp$MeanDecreaseAccuracy > 0 & imp$MeanDecreaseGini > 5, 1, 0)
imp2 <- subset(imp, x == 1, select = c("vars"))
as.list(imp2$vars)
train2 <- subset(train, select = imp2$vars)
#Want to keep the variables below
train2$home_favored <- train$home_favored
train2$fav_won <- train$fav_won
str(train2)


--------------------------------------------------------------------------------


#Append the testing data for future use
test2 <- subset(test, select = colnames(train2))
ncol(test2)
ncol(train2)


#Check the balance
sum(train2$fav_won == "1")
sum(train2$fav_won == "0")
sum(test2$fav_won == "1")
sum(test2$fav_won == "0")


--------------------------------------------------------------------------------
  
  #MLR3
  
#Create the learning task
task <- mlr3::TaskClassif$new(id = "train2",
                              backend = train2, target = "fav_won")

#Load learner & hyperparms
learner <- mlr3::lrn("classif.glmboost", predict_type = "prob")


#Create a training task and a testing task
task_train <- mlr3::TaskClassif$new(id = "task_train",
                                    backend = train2, target = "fav_won")

task_test <- mlr3::TaskClassif$new(id = "task_test",
                                   backend = test2, target = "fav_won")



--------------------------------------------------------------------------------
  
  
#Setting the model parameters
  
learner$param_set


#Parameters - starting--- to find the best kernel
params <- ps(offset = p_dbl(lower = 0.0, upper = 0.01),
             mstop = p_int(lower = 100, upper = 500),
             family = p_fct(list("Binomial")),
             type = p_fct(list("adaboost", "glm")),
             link = p_fct(list("logit", "probit")))

#Evaluation Metrics/Resampling
measure <- mlr3::msr("classif.logloss")
resamp <- rsmp("repeated_cv", repeats = 3, folds = 10L)

#Tuning Budget
evals <- trm("evals", n_evals = 200)

#Setting the model
instance <- mlr3tuning::TuningInstanceSingleCrit$new(
  task = task_train,
  learner = learner,
  resampling = resamp,
  measure = measure,
  search_space = params,
  terminator = evals)


--------------------------------------------------------------------------------
  
  
  #Tuning and evaluation
  
#Tuning
mlr3tuning::tnrs("irace")
tuner1 <- tnr("irace")
tuner1$optimize(instance)


#Optimal Parameters
instance$result_learner_param_vals

#Best result on training data
instance$result_y



--------------------------------------------------------------------------------
  


#Test Evaluation
set.seed(5414)
preds <- learner$predict(task_test)
preds$score(measure)
preds$confusion
#We can use the log loss as well 
confusionMatrix(data = as.factor(preds$data$response),
                reference = as.factor(preds$data$truth))


#Classification Thresholds
probs_glmboost <- as.data.frame(cbind(preds$data$row_ids,
                                      preds$data$truth,
                                      preds$data$response,
                                      preds$data$prob))

colnames(probs_glmboost) <- list("id", "truth", "response", "prob1", "prob2")

probs_glmboost$prob_pred1 <- ifelse(probs_glmboost$prob1 > 0.7, 1, 0)
probs_glmboost$prob_pred2 <- ifelse(probs_glmboost$prob2 > 0.7, 2, 0)
probs_glmboost$prob_preds <- probs_glmboost$prob_pred1 + probs_glmboost$prob_pred2


probs_glmboost2 <- probs_glmboost[!(probs_glmboost$prob_preds == 0),]
confusionMatrix(data = as.factor(probs_glmboost2$prob_preds),
                reference = as.factor(probs_glmboost2$truth))

#86.67% accuracy on 37.73% of data (0.7 threshold)
#81.43% accuracy on 44.02% of data (0.68 threshold)

#Confusion Matrix and Statistics

#            Reference
#Prediction    1  2
#1             0  0
#2             8 52

#Accuracy : 0.8667 

#Given, our model only predicted one class.
#But there was a large class imbalance once we subsetted the data to only
#include games which earned a threshold above 0.7

#Therefore we can conclude that our model is good at predicting
#when the favorite won (positive instances). This COULD be used to our advantage. 

#For instance, in a setting where a model is tasked with predicting
#yes/nos for an ebola outbreak - it would be much more acceptable
#to have a false positive than a false negative. 

#The same logic can apply to this situation since we are using it for betting
#If the goal is parlay multiple bets together:
#The goal is then  just to have a higher accuracy because multiple bets will
#compound the odds, even if all the bets placed are for the favorite to win

#Negative scenarios would yield a higher return if they hit, but that may 
#even out if more bets were parlayed together, even if you chose the 
#favorite to win each game

#For example, the following 2 parlays  MAY yield the same return
#1. Betting 1 underdog to win & 1 favorite to win (2 games)
#3. Betting 3 favorites to win (3 games)

--------------------------------------------------------------------------------
  
#Interpretation & Visualization

--------------------------------------------------------------------------------

#Visualizing the tuning (mlr3viz)

#LogLoss vs Batch Number
mlr3viz::autoplot(object = instance,
                  type = "performance",
                  learner = mlr3::lrn("classif.glmboost"))


#Parallel coordinates (parameters)
mlr3viz::autoplot(object = instance,
                  type = "parallel",
                  learner = mlr3::lrn("classif.glmboost"))

#Shows each parameter and how its adjustments impacted the logloss by batch #
mlr3viz::autoplot(object = instance,
                  type = "marginal",
                  learner = mlr3::lrn("classif.glmboost"))

--------------------------------------------------------------------------------
  
#IML Package
x_train <- train2[which(names(train2) != "fav_won")]
int_model <- Predictor$new(learner, data = x_train, y = train2$fav_won)


#Feature Importance - Permutation Error
feat_imp_ce = FeatureImp$new(int_model, loss = "ce") 
feats = c("LOGspread_favorite", "fav_total_points_against",
          "fav_avg_margin_of_defeat", "away_off_overall", "over_under_line")
feat_imp_ce$plot(features = feats)


#PDPS
effects  <- FeatureEffects$new(int_model, method = "pdp")
plot(effects$effects$LOGspread_favorite)
plot(effects$effects$fav_total_points_against)

--------------------------------------------------------------------------------
  
#DALEX Package
  
#Model Summary
exp = explain_mlr3(learner, data = train2, y = as.numeric(train2$fav_won)-1,
                   label = "GLMBoost",colorize = FALSE)

#Permutation Based Importance
perm_varimp <- DALEX::model_parts(exp)
head(perm_varimp)
plot(perm_varimp, max_vars = 10, show_boxplots = FALSE)

feats2 = c("LOGspread_favorite", "fav_total_points_against",
          "fav_wr_score", "fav_def_score", "fav_ovrl_score")

#PDPs
pdps = model_profile(exp, variables = feats2)$agr_profiles
pdps
plot(pdps) +
  scale_y_continuous("Predicted fav_won Probability") +
  ggtitle("Partial Dependence profiles for selected features")


#SHAP plots/values
shap1 = predict_parts(exp, new_observation = train2, type = "shap")

plot(shap1) +
  scale_y_continuous("SHAP Value")


