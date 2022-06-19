#Fav Won
#Zak Massey
#Reference:
#Bayesian XGB

------------------------------------------------------------------------------------
  
  
  #This project used multiple online resources
  #References:
  #XGBoost Documentation (xgboost.readthedocs.io)
  #Complete Guide to Parameter Tuning in Xgboost (Shegyg)
  #mysmu.edu - Wang Jiwei
  #StackExchange
  #Github
  
  

#The goal is to predict whether or not the favorite will win an NFL football 
#game & interpret the results for betting insights.
#The data was built out using online betting data, season data, & madden data

#For this iteration we will use an XGBoost model with Bayesian Optimization
#Please find the summary in a separte txt file with the same name.

#All data preprocessing, merging & cleaning was done in a separate script  

  
  
  #Clear Space
rm(list = ls())
rm(list = ls(all.names = TRUE))
gc()

#Load Packages
library(parallel)
library(parallelMap)
library(future)
library(dplyr)
library(readr)
library(tictoc)
library(randomForest)
library(purrr)
library(caret)
library(xgboost)
library(ParBayesianOptimization)
library(furrr)

------------------------------------------------------------------------------------
  
  
 #Data is loaded & variables are dropped
  
  
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
  

#Training Data - Turn into matrix
x_train <- data.matrix((train2 %>% dplyr::select(-fav_won)))
y_train <- train2$fav_won
str(y_train)
y_train <- as.numeric(as.factor(train2$fav_won)) - 1
#Testing Data - Turn into matrix
x_test <- data.matrix((test2 %>% dplyr::select(-fav_won)))
y_test <- test2$fav_won
str(y_test)
y_test <- as.numeric(as.factor(test2$fav_won)) - 1
#Define them as xgb objects
xgboost_train <- xgb.DMatrix(data = x_train, label = y_train)
xgboost_test <- xgb.DMatrix(data = x_test, label = y_test)



--------------------------------------------------------------------------------
  

  #Bayesian Tuning Version3
  #DART booster with AUC measurement
  #And cross fold validation
  
--------------------------------------------------------------------------------
  
  #Define the folds
  
#3 fold cv
folds <- list(
  Fold1 = as.integer(seq(1, nrow(train), by = 3)),
  Fold2 = as.integer(seq(2, nrow(train), by = 3)),
  Fold3 = as.integer(seq(3, nrow(train), by = 3))
  )


#5 fold cv
folds <- list(
  Fold1 = as.integer(seq(1, nrow(train), by = 5)),
  Fold2 = as.integer(seq(2, nrow(train), by = 5)),
  Fold3 = as.integer(seq(3, nrow(train), by = 5)),
  Fold4 = as.integer(seq(4, nrow(train), by = 5)),
  Fold5 = as.integer(seq(5, nrow(train), by = 5))
)


--------------------------------------------------------------------------------
  
  #Define the scoring function  & Cross validation methods
  
  
  scoring_function <- function(
    eta, gamma, max_depth, min_child_weight, subsample, nfold,
    lambda, alpha, rate_drop, kappa) {
    
    dtrain <- xgboost_train
    
    pars <- list(
      eta = eta,
      gamma = gamma,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      lambda = lambda,
      alpha = alpha,
      nfold = nfold,
      booster = "dart",
      normalize_type = "forest",
      rate_drop = rate_drop,
      kappa = kappa,
      objective = "binary:logistic",
      eval_metric = "auc",
      sampling_method = "uniform",
      tree_method = "approx",
      verbosity = 0
    )
    xgbcv <- xgb.cv(
      params = pars,
      niter = 50,
      data = dtrain,
      folds = folds,
      nrounds = 50,
      prediction = TRUE,
      showsd = TRUE,
      early_stopping_rounds = 10,
      maximize = TRUE,
      stratified = TRUE
    )

    return(
      list(
        Score = max(xgbcv$evaluation_log$test_auc_mean),
        nrounds = xgbcv$best_iteration
      )
    )
  }


--------------------------------------------------------------------------------
  
  #Set the tuning parameters
  
  bounds <- list(
    eta = c(0.001, 0.1),
    gamma = c(1, 2.75),
    max_depth = c(3L, 16L),
    min_child_weight = c(10, 18),
    subsample = c(0.5, 1),
    nfold = c(3L, 8L),
    lambda = c(0.2, 1),
    alpha = c(0.5, 1),
    rate_drop = c(0.0, 0.075),
    kappa = c(2.56, 4.5)
  )


--------------------------------------------------------------------------------
  
  #Set the tuning model
  
future::nbrOfWorkers()
plan(multisession, workers = 8)
set.seed(5414)
tic()
future_map(c(
  time_noparallel <- system.time(
    opt_obj <- bayesOpt(
      FUN = scoring_function,
      bounds = bounds,
      initPoints = 12,
      iters.n = 25,
      iters.k = 1,
      acq = "poi",
      eps = 0.1,
      plotProgress = TRUE
    ))),  ~Sys.sleep(Inf))
toc()

opt_obj$scoreSummary


--------------------------------------------------------------------------------
  
  
  #Parameter Retirivers
  
  #Maximized (AUC)
  get_maximized_pars <- function (optObj, N = 1){
    if (N > nrow(optObj$scoreSummary))
      stop("N is greater than the iterations that have been run.")
    if (N == 1) {
      return(as.list(head(optObj$scoreSummary[order(-get("Score"))],
                          1))[names(optObj$bounds)])
    }
    else {
      head(optObj$scoreSummary[order(-get("Score"))],
           N)[,names(optObj$bounds), with = FALSE]
    }
  }

#Minimized Parameters (LogLoss)
get_minimized_pars <- function (optObj, N = 1){
  if (N > nrow(optObj$scoreSummary)) 
    stop("N is greater than the iterations that have been run.")
  if (N == 1) {
    return(as.list(head(optObj$scoreSummary[order(get("Score"))],
                        1))[names(optObj$bounds)])
  }
  else {
    head(optObj$scoreSummary[order(get("Score"))],
         N)[,names(optObj$bounds), with = FALSE]
  }
}

--------------------------------------------------------------------------------
  
  #Get Parameters
  get_minimized_pars(opt_obj)


#Take the optimal parameters for xgboost()
params <- list(eta = get_maximized_pars(opt_obj)[1],
               gamma = get_maximized_pars(opt_obj)[2],
               max_depth = get_maximized_pars(opt_obj)[3],
               min_child_weight = get_maximized_pars(opt_obj)[4],
               subsample = get_maximized_pars(opt_obj)[5],
               nfold = get_maximized_pars(opt_obj)[6],
               lambda = get_maximized_pars(opt_obj)[7],
               alpha = get_maximized_pars(opt_obj)[8],
               rate_drop = get_maximized_pars(opt_obj)[9],
               kappa = get_maximized_pars(opt_obj)[10],
               objective = "binary:logistic")


--------------------------------------------------------------------------------
  
  
  #Fit the tuned model
numrounds <- opt_obj$scoreSummary$nrounds[
  which(opt_obj$scoreSummary$Score
         == max(opt_obj$scoreSummary$Score))]

set.seed(2112)
fit_tuned <- xgboost(params = params,
                     data = x_train,
                     label = y_train,
                     nrounds = 50,
                     base_score = 0.5,
                     eval_metric = "auc")



#Model Evaluation:

#Plots the AUC over iteration
plot(fit_tuned$evaluation_log$train_auc)

#ROC/AUC plots
plot(pROC::roc(response = true,
               predictor = predict(fit_tuned, xgboost_test),
               levels=c(0, 1)), lwd=1.5)


--------------------------------------------------------------------------------

  
  #Prediction & Confusion Matrix
  
#Standard (all obs with)
set.seed(5414)
pred_test <- predict(fit_tuned, xgboost_test)
pred <- as.factor(round(pred_test))
true <- as.factor(y_test)
confusionMatrix(pred, true)


#Confusion Matrix and Statistics
#
             #Reference
#Prediction    0  1
#0            15  5
#1            39 99

#Accuracy : 0.7215    
#95% CI : (0.6447, 0.7898)


--------------------------------------------------------------------------------
  
  
#Using the classification thresholds

#The idea with classification thresholds:
#Each prediction has a probability
#For instance:
# [0.54, 0.46]
# [0.70, 0.30]
# [0.25, 0.75]

#If predictions were required with every testing observation, we would
#essentially round the probabilities & determine a [0,1] outcome

#However in this case (place bets), it is not mandatory to bet on every game
#Therefore it may be beneficial to only place bets in instances where the 
#outcome predicted is above a certain threshold of certainty

#For example:
#If the model returns the following predictions for 2 games:
#Game 1: [0.51, 0.49]
#Game 2: [0.75, 0.25]

#It would make sense to disregard game 1, & only focus on game 2
#Since the prediction is more "confident" for lack of a better term

#This is essentially what is going on below

#A "threshold" is selected.
#A bettor would only place a bet if the prediction for 0 or 1 was above 
#a level of certainty

#However the fallback of this approach is that the number of bettable
#instances decreases, but for our goal of correctly parlaying bets, 
#it would be more beneficial


#Now lets use the classification thresholds
preds_probs <- as.data.frame(cbind(pred_test, 1-pred_test, y_test))
colnames(preds_probs) <- list("prob_1", "prob_0", "truth")

preds_probs$prob_pred1 <- ifelse(preds_probs$prob_1 > 0.64, 2, 0)
preds_probs$prob_pred2 <- ifelse(preds_probs$prob_0 > 0.64, 1, 0)
preds_probs$prob_preds <- preds_probs$prob_pred1 + preds_probs$prob_pred2

preds_probs2 <- preds_probs[!(preds_probs$prob_preds == 0),]

confusionMatrix(data = as.factor(preds_probs2$prob_preds),
                reference = as.factor(preds_probs2$truth+1))

#80% Accuracy with 0.69 classification threshold

#With a threshold of 0.69, we were only able to place bets on 47.46% of the data
#Whereas before, we had 70% accuracy when betting on all instances

#We can see that the accuracy greatly increased, but at the expense
#of bettable instances


--------------------------------------------------------------------------------
  
  
#Model interpretations Feature Importance, SHAP plots, PDPs

#Feature importance of final model:
imp = xgb.importance(colnames(xgboost_train), model = fit_tuned)
imp
xgb.plot.importance(imp[1:15,])
sum(imp[,c(2)])



#Shap Plots:
shap <- shap.values(xgb_model = fit_tuned, X_train = xgboost_train)
shap$mean_shap_score
shap_vals <- shap$shap_score
shap_long <- shap.prep(xgb_model = fit_tuned, X_train = x_train)

# **SHAP summary plot**
shap.plot.summary(shap_long, scientific = TRUE)
shap.plot.summary(shap_long, x_bound  = 0.5, dilute = 10)



#Partial Dependency Plots for the most important variables
partial(fit_tuned, pred.var = "LOGspread_favorite", train = x_train, plot = T)
partial(fit_tuned, pred.var = "fav_wr_score", train = x_train, plot = T)
partial(fit_tuned, pred.var = "fav_avg_point_scored_away", train = x_train,
        plot = T)
partial(fit_tuned, pred.var = "dog_avg_yds_agnst", train = x_train, plot = T)
partial(fit_tuned, pred.var = c("fav_wr_score", "fav_avg_point_scored_away"),
        train = x_train, plot = T)


#Interpretation is in the summary document
