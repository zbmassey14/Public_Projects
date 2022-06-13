#Fav Won
#Zak Massey
#Reference:
#Bayesian XGB

------------------------------------------------------------------------------------
  

#The goal is to predict whether or not the favorite will win an NFL football 
  #game
  #The data was built out using online betting data, season data, & madden data
    
  #For this iteration we will use an XGBoost model with Bayesian Optimization
  #We will select variables based on random forest Gini & Mean decrease Acc.
  

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
  
#**Loaded Environment & Data

------------------------------------------------------------------------------------
  
  
  
  #Feature selection for random forest
  #(Data/Dimension Reduction)
  
#Random Forest variable selection
set.seed(5414)
tic()
map(c(
  rf_model <- randomForest(nfl_data$fav_won~., data = nfl_data,
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
nfl_data_rf <- subset(nfl_data, select = imp2$vars)
#Want to keep the variables below
nfl_data_rf$home_favored <- nfl_data$home_favored
nfl_data_rf$fav_won <- nfl_data$fav_won
str(nfl_data_rf)

#Use the mean decrease gini as the baseline
#Use above 0 for mean decrease accuracy

--------------------------------------------------------------------------------
  
  
  
  #Preprocessing
  
  #Since this data is chronological
  #We need to shuffle it & then split it
nfl_data <- as.data.frame(nfl_data)
rows <- sample(nrow(nfl_data))
nfl_data <- nfl_data[rows, ]
str(nfl_data)#Data is shuffled
head(nfl_data)

nfl_data_rf <- as.data.frame(nfl_data_rf)
rows <- sample(nrow(nfl_data_rf))
nfl_data_rf <- nfl_data_rf[rows, ]
str(nfl_data_rf)#Data is shuffled
head(nfl_data_rf)

#Check the outcome balance
sum(nfl_data_rf$fav_won == "1")
sum(nfl_data_rf$fav_won == "0")



--------------------------------------------------------------------------------
  
  #Split the data into test and train
  #Class.if does no support factor inputs
set.seed(5414)
parts <- createDataPartition(nfl_data_rf$fav_won, p = 0.90, list = F)
train <- nfl_data_rf[parts, ]
test <- nfl_data_rf[-parts, ]
#Check data structure & order
str(train)
str(test)
head(train)
head(test)


#Training Data - Turn into matrix
x_train <- data.matrix((train %>% dplyr::select(-fav_won)))
y_train <- train$fav_won
str(y_train)
y_train <- as.numeric(as.factor(train$fav_won)) - 1
#Testing Data - Turn into matrix
x_test <- data.matrix((test %>% dplyr::select(-fav_won)))
y_test <- test$fav_won
str(y_test)
y_test <- as.numeric(as.factor(test$fav_won)) - 1
#Define them as xgb objects
xgboost_train <- xgb.DMatrix(data = x_train, label = y_train)
xgboost_test <- xgb.DMatrix(data = x_test, label = y_test)




--------------------------------------------------------------------------------
  
  
  #Bayesian Tuning
  
  --------------------------------------------------------------------------------
  
  #Define the folds
  
  #3 fold cv
  folds <- list(
    Fold1 = as.integer(seq(1, nrow(train), by = 3)),
    Fold2 = as.integer(seq(2, nrow(train), by = 3)),
    Fold3 = as.integer(seq(3, nrow(train), by = 3))
  )


--------------------------------------------------------------------------------
  
  #Define the scoring function  & Cross validation methods
  #Using a DART booster with xgb cross validation
  #Implement early stopping
  #Set maximize false since we are using logloss as the eval metric
  
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
      normalize_type = "forest", #Dart
      rate_drop = rate_drop, #Dart
      kappa = kappa, #Dart
      objective = "binary:logistic",
      eval_metric = "logloss",
      sampling_method = "uniform", #Set subsmaple=>0.5
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
      maximize = FALSE, #We do not want a large eval score
      stratified = TRUE
    )
    # required by the package, the output must be a list
    # with at least one element of "Score", the measure to optimize
    # Score must start with capital S
    # For this case, we also report the best num of iteration
    return(
      list(
        Score = min(xgbcv$evaluation_log$test_logloss_mean),
        nrounds = xgbcv$best_iteration
      )
    )
  }


--------------------------------------------------------------------------------
  
  #Set the tuning parameters
  #Parameters based off XGB documentation & recommendations
  #Also based off previous tests
  
  bounds <- list(
    eta = c(0.01, 0.1),
    gamma = c(2, 2.75), #Let the model be aggressive
    max_depth = c(3L, 16L),
    min_child_weight = c(10, 18),
    subsample = c(0.6, 1),
    nfold = c(5L, 8L),
    lambda = c(0.25, 0.85), #Let the model be aggressive
    alpha = c(0.5, 1), #Let the model be aggressive
    rate_drop = c(0.0, 0.075),
    kappa = c(2.96, 4.5)
  )


--------------------------------------------------------------------------------
  
  #Set the tuning model
  #Using multiple cores (all 8) since this takes a while to tune
  
  #Going for 50 iterations & running the scoring function one time per iteration
  
future::nbrOfWorkers()
plan(multisession, workers = 8)
set.seed(5414)
tic()
future_map(c(
  time_noparallel <- system.time(
    opt_obj <- bayesOpt(
      FUN = scoring_function,
      bounds = bounds,
      initPoints = 11,
      iters.n = 50,
      iters.k = 1,
      acq = "poi", #Not using EI - want exploration
      plotProgress = TRUE
    ))),  ~Sys.sleep(Inf))
toc()


opt_obj$scoreSummary

--------------------------------------------------------------------------------
  
  
  #Parameter Retirivers
  
  #Maximized (use with AUC)
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

#Minimized Parameters (use with LogLoss)
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


# take the optimal parameters for xgboost()
#Set the parmeters
params <- list(eta = get_minimized_pars(opt_obj)[1],
               gamma = get_minimized_pars(opt_obj)[2],
               max_depth = get_minimized_pars(opt_obj)[3],
               min_child_weight = get_minimized_pars(opt_obj)[4],
               subsample = get_minimized_pars(opt_obj)[5],
               nfold = get_minimized_pars(opt_obj)[6],
               lambda = get_minimized_pars(opt_obj)[7],
               alpha = get_minimized_pars(opt_obj)[8],
               rate_drop = get_minimized_pars(opt_obj)[9],
               kappa = get_minimized_pars(opt_obj)[10],
               objective = "binary:logistic")


--------------------------------------------------------------------------------
  
  
  #Fit the tuned model
  numrounds <- opt_obj$scoreSummary$nrounds[
    which(opt_obj$scoreSummary$Score
          == max(opt_obj$scoreSummary$Score))]

fit_tuned <- xgboost(params = params,
                     data = x_train,
                     label = y_train,
                     nrounds = numrounds,
                     base_score = 0.5,
                     eval_metric = "logloss")



--------------------------------------------------------------------------------
  
  
  
  #Prediction & Confusion Matrix
  
  #Standard (all obs with)
set.seed(5414)
pred_test <- predict(fit_tuned, xgboost_test)
pred <- as.factor(round(pred_test))
true <- as.factor(y_test)
confusionMatrix(pred, true)
#62.89% Accuracy
#64.15% Accuracy when increasing the testing amounts


#70.89% accuracy with the xgboost on all obs

--------------------------------------------------------------------------------


#Now lets use the probability thresholds

#The idea with probability thresholds:
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
#instances decreases, but for our goal of correctly parlarying bets, 
#it would be more benefical
  
#Cleaning the predictions
preds_probs <- as.data.frame(cbind(pred_test, 1-pred_test, y_test))
colnames(preds_probs) <- list("prob_1", "prob_0", "truth")

#Setting the threshold
preds_probs$prob_pred1 <- ifelse(preds_probs$prob_1 > 0.7, 2, 0)
preds_probs$prob_pred2 <- ifelse(preds_probs$prob_0 > 0.7, 1, 0)
preds_probs$prob_preds <- preds_probs$prob_pred1 + preds_probs$prob_pred2

preds_probs2 <- preds_probs[!(preds_probs$prob_preds == 0),]

confusionMatrix(data = as.factor(preds_probs2$prob_preds),
                reference = as.factor(preds_probs2$truth+1))


#83.33% -- 22.7% of data (0.7 thres)

#We can see that the accuracy greatly increased, but at the expense
#of bettable instances

#With a threshold of 0.70, we were only able to place bets on 22.7% of the data
#Whereas before, we had 70% accuracy when betting on all instances




