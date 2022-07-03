#Over
#SVM
#Support Vector Machine

------------------------------------------------------------------------------------
  
  #Clear Space
rm(list = ls())
rm(list = ls(all.names = TRUE))
gc()

#Load Pacckages
#****

------------------------------------------------------------------------------------
  
 #*Import data and do minor cleaning
 
  ------------------------------------------------------------------------------------
  
  
 #Since this data is chronological
 #We need to shuffle it & then split it
nfl_data <- as.data.frame(nfl_data)
rows <- sample(nrow(nfl_data))
nfl_data <- nfl_data[rows, ]


#Split the data into test and train
set.seed(1221)
parts <- createDataPartition(nfl_data$over = 0.90, list = F)
train <- nfl_data[parts, ]
test <- nfl_data[-parts, ]
#Check data structure & order
str(train)
str(test)
head(train)
head(test)


------------------------------------------------------------------------------------


 #Random Forest variable selection
set.seed(5414)
tic()
map(c(
  rf_model <- randomForest(train$over~., data = train,
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
train2$over <- train$over
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
#Convert to numeric for SVM
train2 <- as.data.frame(lapply(train2, as.numeric))
train2$over <- as.factor(train2$over)
str(train2)

test2 <- as.data.frame(lapply(test2, as.numeric))
test2$over <- as.factor(test2$over)
str(test2)


#Create the learning task
task <- mlr3::TaskClassif$new(id = "train2",
                              backend = train2, target = "over")

#Load learner & hyperparms
learner <- mlr3::lrn("classif.svm", predict_type = "prob")


#Create a training task and a testing task
task_train <- mlr3::TaskClassif$new(id = "task_train",
                                    backend = train2, target = "over")

task_test <- mlr3::TaskClassif$new(id = "task_test",
                                   backend = test2, target = "over")


--------------------------------------------------------------------------------
  
#Setting Tuning Parameters
  
params <- ps(kernel = p_fct(list("radial", "polynomial", "sigmoid")),
               gamma = p_dbl(lower = 0.001, upper = 0.1),
               epsilon = p_dbl(lower = 0.01, upper = 0.1),
               cost = p_dbl(lower = 0.001, upper = 1),
               tolerance = p_dbl(lower = 0.0001, upper = 0.01),
               type = p_fct(list("C-classification")))


#Evaluation Metrics/Resampling
measure <- mlr3::msr("classif.logloss")
resamp <- rsmp("repeated_cv", repeats = 3, folds = 10L)

#Tuning Budget
evals <- trm("evals", n_evals = 200

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
caret::confusionMatrix(data = as.factor(preds$data$response),
                       reference = as.factor(preds$data$truth))

#Confusion Matrix and Statistics

#Reference
#Prediction  0  1
#0          71 54
#1          12 23
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
#1          16  8
#2           0  5
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
  scale_y_continuous("iBreakDown")
