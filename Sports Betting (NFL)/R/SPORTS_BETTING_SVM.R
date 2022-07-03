#Over
#SVM
#Support Vector Machine

------------------------------------------------------------------------------------
  
  #Clear Space
rm(list = ls())
rm(list = ls(all.names = TRUE))
gc()

#Load Pacckages
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
library(nnet)
library(caret)

------------------------------------------------------------------------------------
  
  
  #Data Load
  nfl_data <- read_csv("MS Econ Analytics/MS-Economic-Analytics/SportsBetting/NFL/Masters/FINALDATASETS/Final Data Sets/modelready/master_may16.csv", 
                       col_types = cols(idx = col_number(),
                                        week = col_number(),
                                        season = col_number(),
                                        playoff = col_factor(levels = c("0","1")),
                                        score_home = col_number(),
                                        score_away = col_number(),
                                        home_favored = col_factor(levels = c("0","1")),
                                        margin_of_victory = col_number(),
                                        over_under_line = col_number(),
                                        fav_won = col_factor(levels = c("0","1")),
                                        over = col_factor(levels = c("0","1")),
                                        dog_covered = col_factor(levels = c("0","1")),
                                        fav_covered = col_factor(levels = c("0","1")),
                                        away_fav = col_factor(levels = c("0","1")),
                                        fav_won6 = col_factor(levels = c("0","1")),
                                        fav_prev_fav = col_factor(levels = c("0","1")),
                                        curr_fav_won_lst_wk_prev_fav = col_factor(levels = c("0","1")),
                                        dog_prev_dog = col_factor(levels = c("0","1")),
                                        curr_dog_won_lst_wk_prev_dog = col_factor(levels = c("0","1")),
                                        fav_won_lst_wk = col_factor(levels = c("0","1")),
                                        dog_won_lst_wk = col_factor(levels = c("0","1")),
                                        fav_record = col_number(),
                                        dog_record = col_number(),
                                        fav_avg_point_scored_home = col_number(),
                                        fav_avg_point_scored_away = col_number(),
                                        fav_avg_point_scored = col_number(),
                                        dog_avg_point_scored_home = col_number(),
                                        dog_avg_point_scored_away = col_number(),
                                        dog_avg_point_scored = col_number(),
                                        fav_avg_pts_agnst_home = col_number(),
                                        fav_avg_pts_agnst_away = col_number(),
                                        fav_avg_pts_agnst = col_number(),
                                        dog_avg_pts_agnst_home = col_number(),
                                        dog_avg_pts_agnst_away = col_number(),
                                        dog_avg_pts_agnst = col_number(),
                                        fav_covered_pct = col_number(),
                                        dog_covered_pct = col_number(),
                                        fav_avg_margin_victory = col_number(),
                                        dog_avg_margin_victory = col_number(),
                                        fav_avg_margin_of_defeat = col_number(),
                                        dog_avg_margin_of_defeat = col_number(),
                                        fav_total_points_for = col_number(),
                                        dog_total_points_for = col_number(),
                                        fav_total_points_against = col_number(),
                                        dog_total_points_against = col_number(),
                                        stadium_neutral = col_factor(levels = c("0","1")),
                                        home_avg_att = col_number(),
                                        home_trvl_avg_att = col_number(), 
                                        home_ovr_avg_att = col_number(), 
                                        away_avg_att = col_number(),
                                        away_trvl_avg_att = col_number(), 
                                        away_ovr_avg_att = col_number(), 
                                        home_team_overall = col_number(), 
                                        home_def_overall = col_number(), 
                                        home_off_overall = col_number(), 
                                        away_team_overall = col_number(), 
                                        away_def_overall = col_number(), 
                                        away_off_overall = col_number(),
                                        home_QB = col_number(), home_HB = col_number(), 
                                        home_WR = col_number(), home_OL = col_number(), 
                                        home_TE = col_number(), home_DL = col_number(), 
                                        home_LB = col_number(), home_SCDY = col_number(), 
                                        home_SPT = col_number(), away_QB = col_number(), 
                                        away_HB = col_number(), away_WR = col_number(), 
                                        away_TE = col_number(), away_OL = col_number(), 
                                        away_DL = col_number(), away_LB = col_number(), 
                                        away_SCDY = col_number(), away_SPT = col_number(), 
                                        wildcard = col_factor(levels = c("0","1")),
                                        Divisionalround = col_factor(levels = c("0","1")),
                                        Conferencechamp = col_factor(levels = c("0","1")), 
                                        SuperBowl = col_factor(levels = c("0","1")), 
                                        home_cold = col_factor(levels = c("0","1")), 
                                        home_warm = col_factor(levels = c("0","1")),
                                        home_moderate = col_factor(levels = c("0","1")),
                                        home_outdoor = col_factor(levels = c("0", "1")),
                                        home_GrassField = col_factor(levels = c("0","1")),
                                        away_cold = col_factor(levels = c("0","1")),
                                        away_warm = col_factor(levels = c("0","1")),
                                        away_moderate = col_factor(levels = c("0","1")),
                                        away_outdoor = col_factor(levels = c("0","1")),
                                        away_GrassField = col_factor(levels = c("0","1")),
                                        DivisionalGame = col_factor(levels = c("0","1")),
                                        ConfreneceGame = col_factor(levels = c("0","1")),
                                        freezing = col_factor(levels = c("0","1")),
                                        breezy = col_factor(levels = c("0","1")),
                                        windy = col_factor(levels = c("0","1")),
                                        home_rookie_coach = col_factor(levels = c("0","1")),
                                        away_rookie_coach = col_factor(levels = c("0","1")), 
                                        SCALEdistance_traveled = col_number(),
                                        rain = col_factor(levels = c("0","1")),
                                        home_stdm_age = col_number(),
                                        away_stdm_age = col_number(),
                                        home_AFC = col_factor(levels = c("0","1")),
                                        away_AFC = col_factor(levels = c("0","1")),
                                        home_AFCeast = col_factor(levels = c("0","1")),
                                        home_AFCsouth = col_factor(levels = c("0","1")),
                                        home_AFCwest = col_factor(levels = c("0","1")),
                                        home_NFCeast = col_factor(levels = c("0","1")),
                                        home_NFCsouth = col_factor(levels = c("0","1")),
                                        home_NFCwest = col_factor(levels = c("0","1")),
                                        away_AFCeast = col_factor(levels = c("0","1")),
                                        away_AFCsouth = col_factor(levels = c("0","1")),
                                        away_AFCwest = col_factor(levels = c("0","1")),
                                        away_NFCeast = col_factor(levels = c("0","1")),
                                        away_NFCsouth = col_factor(levels = c("0","1")),
                                        away_NFCwest = col_factor(levels = c("0","1")),
                                        fav_ovrl_score = col_number(),
                                        fav_off_score = col_number(),
                                        fav_def_score = col_number(),
                                        fav_wr_score = col_number(),
                                        fav_scdy_score = col_number(),
                                        fav_ol_score = col_number(),
                                        fav_dl_score = col_number(),
                                        fav_coach_score = col_number(),
                                        SCALEweather_temperature = col_number(),
                                        LOGspread_favorite = col_number(),
                                        LOGhome_ch_tenure = col_number(),
                                        LOGaway_ch_tenure = col_number(),
                                        SCALEdistance_traveled = col_number(),
                                        NORMhome_stadium_capacity = col_number(), 
                                        NORMwaway_stadium_capacity = col_number(),
                                        SCALEelevation_difference = col_number()
                       ))

#Data Load
nfl_data <- read_csv("MS Econ Analytics/MS-Economic-Analytics/SportsBetting/NFL/Masters/FINALDATASETS/Final Data Sets/modelready/june7_verified.csv", 
                     col_types = cols(idx = col_number(),
                                      week = col_number(),
                                      season = col_number(),
                                      playoff = col_factor(levels = c("0","1")),
                                      score_home = col_number(),
                                      score_away = col_number(),
                                      home_favored = col_factor(levels = c("0","1")),
                                      fav_tot_szn_yds = col_number(),
                                      dog_tot_szn_yds = col_number(),
                                      fav_avg_yds = col_number(),
                                      dog_avg_yds = col_number(),
                                      fav_tot_to = col_number(),
                                      fav_avg_to = col_number(),
                                      dog_tot_to = col_number(),
                                      dog_avg_to = col_number(),
                                      fav_tot_yds_agnst = col_number(),
                                      fav_avg_yds_agnst = col_number(),
                                      dog_tot_yds_agnst = col_number(),
                                      dog_avg_yds_agnst = col_number(),
                                      fav_tot_takeaways = col_number(),
                                      fav_avg_takeways = col_number(),
                                      dog_tot_takeaways = col_number(),
                                      dog_avg_takeways = col_number(),
                                      margin_of_victory = col_number(),
                                      over_under_line = col_number(),
                                      fav_won = col_factor(levels = c("0","1")),
                                      over = col_factor(levels = c("0","1")),
                                      dog_covered = col_factor(levels = c("0","1")),
                                      fav_covered = col_factor(levels = c("0","1")),
                                      away_fav = col_factor(levels = c("0","1")),
                                      fav_won6 = col_factor(levels = c("0","1")),
                                      fav_prev_fav = col_factor(levels = c("0","1")),
                                      curr_fav_won_lst_wk_prev_fav = col_factor(levels = c("0","1")),
                                      dog_prev_dog = col_factor(levels = c("0","1")),
                                      curr_dog_won_lst_wk_prev_dog = col_factor(levels = c("0","1")),
                                      fav_won_lst_wk = col_factor(levels = c("0","1")),
                                      dog_won_lst_wk = col_factor(levels = c("0","1")),
                                      fav_record = col_number(),
                                      dog_record = col_number(),
                                      fav_avg_point_scored_home = col_number(),
                                      fav_avg_point_scored_away = col_number(),
                                      fav_avg_point_scored = col_number(),
                                      dog_avg_point_scored_home = col_number(),
                                      dog_avg_point_scored_away = col_number(),
                                      dog_avg_point_scored = col_number(),
                                      fav_avg_pts_agnst_home = col_number(),
                                      fav_avg_pts_agnst_away = col_number(),
                                      fav_avg_pts_agnst = col_number(),
                                      dog_avg_pts_agnst_home = col_number(),
                                      dog_avg_pts_agnst_away = col_number(),
                                      dog_avg_pts_agnst = col_number(),
                                      fav_covered_pct = col_number(),
                                      dog_covered_pct = col_number(),
                                      fav_avg_margin_victory = col_number(),
                                      dog_avg_margin_victory = col_number(),
                                      fav_avg_margin_of_defeat = col_number(),
                                      dog_avg_margin_of_defeat = col_number(),
                                      fav_total_points_for = col_number(),
                                      dog_total_points_for = col_number(),
                                      fav_total_points_against = col_number(),
                                      dog_total_points_against = col_number(),
                                      stadium_neutral = col_factor(levels = c("0","1")),
                                      home_avg_att = col_number(),
                                      home_trvl_avg_att = col_number(), 
                                      home_ovr_avg_att = col_number(), 
                                      away_avg_att = col_number(),
                                      away_trvl_avg_att = col_number(), 
                                      away_ovr_avg_att = col_number(), 
                                      home_team_overall = col_number(), 
                                      home_def_overall = col_number(), 
                                      home_off_overall = col_number(), 
                                      away_team_overall = col_number(), 
                                      away_def_overall = col_number(), 
                                      away_off_overall = col_number(),
                                      home_QB = col_number(), home_HB = col_number(), 
                                      home_WR = col_number(), home_OL = col_number(), 
                                      home_TE = col_number(), home_DL = col_number(), 
                                      home_LB = col_number(), home_SCDY = col_number(), 
                                      home_SPT = col_number(), away_QB = col_number(), 
                                      away_HB = col_number(), away_WR = col_number(), 
                                      away_TE = col_number(), away_OL = col_number(), 
                                      away_DL = col_number(), away_LB = col_number(), 
                                      away_SCDY = col_number(), away_SPT = col_number(), 
                                      wildcard = col_factor(levels = c("0","1")),
                                      Divisionalround = col_factor(levels = c("0","1")),
                                      Conferencechamp = col_factor(levels = c("0","1")), 
                                      SuperBowl = col_factor(levels = c("0","1")), 
                                      home_cold = col_factor(levels = c("0","1")), 
                                      home_warm = col_factor(levels = c("0","1")),
                                      home_moderate = col_factor(levels = c("0","1")),
                                      home_outdoor = col_factor(levels = c("0", "1")),
                                      home_GrassField = col_factor(levels = c("0","1")),
                                      away_cold = col_factor(levels = c("0","1")),
                                      away_warm = col_factor(levels = c("0","1")),
                                      away_moderate = col_factor(levels = c("0","1")),
                                      away_outdoor = col_factor(levels = c("0","1")),
                                      away_GrassField = col_factor(levels = c("0","1")),
                                      DivisionalGame = col_factor(levels = c("0","1")),
                                      ConfreneceGame = col_factor(levels = c("0","1")),
                                      freezing = col_factor(levels = c("0","1")),
                                      breezy = col_factor(levels = c("0","1")),
                                      windy = col_factor(levels = c("0","1")),
                                      home_rookie_coach = col_factor(levels = c("0","1")),
                                      away_rookie_coach = col_factor(levels = c("0","1")), 
                                      SCALEdistance_traveled = col_number(),
                                      rain = col_factor(levels = c("0","1")),
                                      home_stdm_age = col_number(),
                                      away_stdm_age = col_number(),
                                      home_AFC = col_factor(levels = c("0","1")),
                                      away_AFC = col_factor(levels = c("0","1")),
                                      home_AFCeast = col_factor(levels = c("0","1")),
                                      home_AFCsouth = col_factor(levels = c("0","1")),
                                      home_AFCwest = col_factor(levels = c("0","1")),
                                      home_NFCeast = col_factor(levels = c("0","1")),
                                      home_NFCsouth = col_factor(levels = c("0","1")),
                                      home_NFCwest = col_factor(levels = c("0","1")),
                                      away_AFCeast = col_factor(levels = c("0","1")),
                                      away_AFCsouth = col_factor(levels = c("0","1")),
                                      away_AFCwest = col_factor(levels = c("0","1")),
                                      away_NFCeast = col_factor(levels = c("0","1")),
                                      away_NFCsouth = col_factor(levels = c("0","1")),
                                      away_NFCwest = col_factor(levels = c("0","1")),
                                      fav_ovrl_score = col_number(),
                                      fav_off_score = col_number(),
                                      fav_def_score = col_number(),
                                      fav_wr_score = col_number(),
                                      fav_scdy_score = col_number(),
                                      fav_ol_score = col_number(),
                                      fav_dl_score = col_number(),
                                      fav_coach_score = col_number(),
                                      SCALEweather_temperature = col_number(),
                                      LOGspread_favorite = col_number(),
                                      LOGhome_ch_tenure = col_number(),
                                      LOGaway_ch_tenure = col_number(),
                                      SCALEdistance_traveled = col_number(),
                                      NORMhome_stadium_capacity = col_number(), 
                                      NORMwaway_stadium_capacity = col_number(),
                                      SCALEelevation_difference = col_number()
                     ))

str(nfl_data) #1606 Games. 134 Variables.
sum(is.na(nfl_data)) #No NAs



#Data Clean
#Need to create a separate dataset with only the selected variables
nfl_data <- nfl_data
nfl_data$winner_turnover <- NULL
nfl_data$loser_turnovers <- NULL
nfl_data$winner_yards <- NULL
nfl_data$loser_yards <- NULL
nfl_data$home_rookie_coach <- NULL
nfl_data$away_rookie_coach <- NULL
nfl_data$home_ovr_avg_att <- NULL
nfl_data$away_ovr_avg_att <- NULL
nfl_data$home_team_overall <- NULL
nfl_data$away_team_overall <- NULL
nfl_data$team_home <- NULL
nfl_data$team_away <- NULL
nfl_data$team_favorite_id <- NULL
nfl_data$score_home <- NULL
nfl_data$score_away <- NULL
nfl_data$SuperBowl <- NULL
nfl_data$winner <- NULL
nfl_data$dog <- NULL
nfl_data$idx <- NULL
nfl_data$loser <- NULL
nfl_data$margin_of_victory <- NULL
nfl_data <- nfl_data

#Outcome Variables
nfl_data$fav_won6 <- NULL
nfl_data$home_win <- NULL
nfl_data$home_fav <- NULL
nfl_data$away_win <- NULL
nfl_data$away_fav <- NULL
nfl_data$over <- NULL
nfl_data$fav_won 
nfl_data$dog_covered <- NULL
nfl_data$fav_covered <- NULL
str(nfl_data)
nfl_data$SCALEelevation_difference

------------------------------------------------------------------------------------
  
  #Feature selection for random forest
  #(Data/Dimension Reduction)
  
  #Random Forest variable selection
set.seed(5414)
tic()
map(c(
  rf_model <- randomForest(nfl_data$over~., data = nfl_data,
                           mtry = 15, ntree = 500, importance = TRUE, proximity = TRUE)
),  ~Sys.sleep(.x))
toc()
imp <- as.data.frame(round(importance(rf_model), 2))
imp$vars <- rownames(imp)
#Only want to keep if vairbles helps accuracy
imp$x <- ifelse(imp$MeanDecreaseAccuracy > 0 & imp$MeanDecreaseGini > 5, 1, 0)
imp2 <- subset(imp, x == 1, select = c("vars"))
as.list(imp2$vars)
nfl_data_rf <- subset(nfl_data, select = imp2$vars)
nfl_data_rf$over <- nfl_data$over
str(nfl_data_rf)

------------------------------------------------------------------------------------
  
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
str(nfl_data_rf)

#Make the data have en even outcome split
sum(nfl_data_rf$over == "1")
sum(nfl_data_rf$over == "0")


--------------------------------------------------------------------------------
  
  #MLR3
  #MLR3
  
  #Create the learning task
  task <- mlr3::TaskClassif$new(id = "nfl_data_rf",
                                backend = nfl_data_rf, target = "over")

#Load learner & hyperparms
#Want probabilities so we can use logloss as eval metric
#learner <- mlr3::lrn("classif.multinom")
learner <- mlr3::lrn("classif.svm", predict_type = "prob")

#Split the data into test and train
set.seed(1221)
parts <- createDataPartition(nfl_data_rf$over, p = 0.90, list = F)
train <- nfl_data_rf[parts, ]
test <- nfl_data_rf[-parts, ]

#Convert to numeric
train <- as.data.frame(lapply(train, as.numeric))
train$over <- as.factor(train$over)
str(train)

test <- as.data.frame(lapply(test, as.numeric))
test$over <- as.factor(test$over)
str(test)


#Create a training task and a testing task
task_train <- mlr3::TaskClassif$new(id = "task_train",
                                    backend = train, target = "over")

task_test <- mlr3::TaskClassif$new(id = "task_test",
                                   backend = test, target = "over")




--------------------------------------------------------------------------------
  
#Setting Tuning Parameters
  
params <- ps(kernel = p_fct(list("radial", "polynomial", "sigmoid")),
               gamma = p_dbl(lower = 0.001, upper = 0.1),
               epsilon = p_dbl(lower = 0.01, upper = 0.1),
               cost = p_dbl(lower = 0.001, upper = 1),
               tolerance = p_dbl(lower = 0.0001, upper = 0.01),
               type = p_fct(list("C-classification")))

#Evaluation Metrics/Resampling
#measure = msr("classif.ce") - classification error
#measure <- mlr3::msr("classif.acc")
measure <- mlr3::msr("classif.logloss")
resamp <- rsmp("repeated_cv", repeats = 3, folds = 10L)

#Tuning Budget
evals <- trm("evals", n_evals = 300)

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
  #Setting the gridsearch
mlr3tuning::tnrs("irace")
tuner1 <- tnr("irace")
tuner1$optimize(instance)
tuner2 <- tnr("gensa")
tuner3 <- tnr("cmaes")


#Optimal Parameters
instance$result_learner_param_vals

#Best result on training data
instance$result_y



--------------------------------------------------------------------------------
  
  
  #Applying the optimized parameters
  #And creating the Final Model
  #Set learner params as the optimal params
  #Train the model with the optimal parameters
learner$param_set$values <- instance$result_learner_param_vals
learner$train(task_train)
#Setting the nested resampling
learner_resmap <- resample(task_train, learner, resamp, store_models = TRUE)
#Compare outer/inner resampling  performance
learner_resmap$score()
learner_resmap$learners
learner #Make sure this has the best parameters
#Look at aggregate performance
learner_resmap$aggregate()

#Preferably use the log loss from the training data as the model weights
#when comparing

--------------------------------------------------------------------------------
  
  #Test Evaluation
set.seed(5414)
preds <- learner$predict(task_test)
preds$score(measure)
preds$confusion
#We can use the log loss as well 
caret::confusionMatrix(data = as.factor(preds$data$response),
                       reference = as.factor(preds$data$truth))

#Confusion Matrix and Statistics

#Reference
#Prediction  0  1
#0 71 54
#1 12 23

#Accuracy : 0.5875 


#Classification Thresholds
probs_svm <- as.data.frame(cbind(preds$data$row_ids,
                                      preds$data$truth,
                                      preds$data$response,
                                      preds$data$prob))

colnames(probs_svm) <- list("id", "truth", "response", "prob1", "prob2")

probs_svm$prob_pred1 <- ifelse(probs_svm$prob1 > 0.54, 1, 0)
probs_svm$prob_pred2 <- ifelse(probs_svm$prob2 > 0.54, 2, 0)
probs_svm$prob_preds <- probs_svm$prob_pred1 + probs_svm$prob_pred2


probs_svm2 <- probs_svm[!(probs_svm$prob_preds == 0),]
caret::confusionMatrix(data = as.factor(probs_svm2$prob_preds),
                       reference = as.factor(probs_svm2$truth))


#Confusion Matrix and Statistics
#
#Reference
#Prediction  1  2
#1 16  8
#2  0  5

#Accuracy : 0.7241 



--------------------------------------------------------------------------------
  
  #Interpretation & Visualization
  
--------------------------------------------------------------------------------
  
  #Visualizing the tuning (mlr3viz)
  
  #LogLoss vs Batch Number
mlr3viz::autoplot(object = instance,
                  type = "performance",
                  learner = mlr3::lrn("classif.svm"))


#Parallel coordinates (parameters)
mlr3viz::autoplot(object = instance,
                  type = "parallel",
                  learner = mlr3::lrn("classif.svm"))

#Shows each parameter and how its adjustments impacted the logloss by batch #
mlr3viz::autoplot(object = instance,
                  type = "marginal",
                  learner = mlr3::lrn("classif.svm"))


--------------------------------------------------------------------------------

train2 <- nfl_data_rf
  
  
  #IML Package
x_train <- train2[which(names(train2) != "over")]
int_model <- Predictor$new(learner, data = x_train, y = as.numeric(train2$over))


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

#Model Summary
exp = explain_mlr3(learner, data = x_train, y = as.numeric(train2$over)-1,
                   label = "SVM",colorize = FALSE)


#Dalex Package
perm_varimp <- DALEX::model_parts(exp)
head(perm_varimp)
plot(perm_varimp, max_vars = 10, show_boxplots = FALSE)

feats2 = c("dog_avg_point_scored_home", "dog_avg_point_scored",
           "home_off_overall", "dog_avg_pts_agnst", "dog_covered_pct",
           "over_under_line", "fav_avg_point_scored")

#PDPs
pdps = model_profile(exp, variables = feats2)$agr_profiles
pdps
plot(pdps) +
  scale_y_continuous("Predicted over Probability") +
  ggtitle("Partial Dependence profiles for selected features")


#SHAP plots/values & BreakDown Profile
shap1 = predict_parts(exp, new_observation = train2, type = "shap")
shap2 = predict_parts(exp, new_observation = train2)

plot(shap1) +
  scale_y_continuous("SHAP Value")

plot(shap2) +
  scale_y_continuous("SHAP Value")