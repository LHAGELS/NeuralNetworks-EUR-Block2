library(nnet)
library(readr)
library(caret)
library(ROSE)
library(tidyverse)
library(DMwR)
####################################################################################
#### 1. Data Preparation
## Read Dataframe
Churn <- read_csv("Churn.csv")

## Inspect Data
summary(Churn)
#> Booleans: HasCrCard, IsActiveMember, Exited

## Read Dataframe and change Datatypes
Churn <- read_csv("Churn.csv", 
                  col_types = cols(Exited = col_logical(), 
                                   HasCrCard = col_logical(), 
                                   IsActiveMember = col_logical()))
## Convert columns to integers
Churn[, c(2:4,6)] <- lapply(Churn[,c(2:4,6)], as.integer)

## Convert Dependent Variable to a factor
Churn$Exited <- as.factor(ifelse(Churn$Exited == TRUE, "Exited","Not_Exited"))
Churn$HasCrCard <- as.factor(ifelse(Churn$HasCrCard == TRUE, 1,0))
Churn$IsActiveMember <- as.factor(ifelse(Churn$IsActiveMember == TRUE, 1,0))

## Drop first column
Churn <- Churn[,-1]

#####################################################################
#### 2. Preprocessing
summary(Churn$Exited)

#Min-Max Scaler
normalize <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

## Normalize numeric values
Churn[,c(1:5,8)] <- sapply(Churn[,c(1:5,8)],normalize)
colnames(Churn)

## Correlationplot
par(mfrow=c(1,3), mar=c(4,4,1,1))
plot(Churn$CreditScore, Churn$Age, xlab="CreditScore", ylab="Age", col = rgb(red = 0.1, green = 0.1, blue = 0.1, alpha = 0.1), pch=20, cex=2)
abline(lm(Churn$CreditScore ~ Churn$Age), col="blue")
plot(Churn$CreditScore, Churn$Balance,xlab="CreditScore", ylab="Balance",col = rgb(red = 0.1, green = 0.1, blue = 0.1, alpha = 0.1), pch=20, cex=2)
abline(lm(Churn$CreditScore ~ Churn$Balance), col="blue")
plot(Churn$Age, Churn$Balance,xlab="Age", ylab="Balance",col = rgb(red = 0.1, green = 0.1, blue = 0.1, alpha = 0.1), pch=20, cex=2)
abline(lm(Churn$Age ~ Churn$Balance), col="blue")

## Sort the Dataframe and create formula
Churn_sorted <- data.frame(Churn[,9],Churn[,1:8])
formula <-as.formula('Exited~CreditScore+Age+Tenure+Balance+NumOfProducts+HasCrCard+IsActiveMember+EstimatedSalary')

## Stratify Data --> train/test (80/20)
set.seed(1)
train.index <- createDataPartition(Churn_sorted[,1], p = .7, list = FALSE)
train_data <- Churn_sorted[train.index,]
test_data  <- Churn_sorted[-train.index,]
colnames(train_data) <- c("Exited", "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary")

# Check the split 70% train data (80% NO, 20% YES), 30% test data (80% NO, 20% YES)
table(train_data[,1])

#######################################################################
#### 3. Analysis 
# part a: set range of tuning parameters (layer size and weight decay)
tune_grid_neural <- expand.grid(size = c(7,8,9,10,12,14,16,20),
                                decay = c(0, 0.00001, 0.0001, 0.0003, 0.001,0.01,0.1))
# part b: constraint calculation
max_size_neural <- max(tune_grid_neural$size)
max_weights_neural <- max_size_neural*(nrow(train_data) + 1) + max_size_neural + 1
# -----------------------------------------------------------------------------
# STEP 2: SELECT TUNING METHOD
# set up train control object, which specifies training/testing technique
train_control_neural <- trainControl(method = "LGOCV",
                                     number = 10,
                                     classProbs = TRUE, 
                                     verboseIter = TRUE)
# -----------------------------------------------------------------------------
# STEP 0: set seed, so that statistics don't keep changing for every analysis
set.seed(1)
# -----------------------------------------------------------------------------
# STEP 1: decide how many times to run the model
rounds <- 10
# -----------------------------------------------------------------------------
# STEP 2: set up object to store results
# part a: create names of results to store
result_cols <- c("model_type", "round", "accuracy", "kappa", "accuracy_LL", "accuracy_UL",
                 "sensitivity", "specificity", "precision", "npv","F1", "n")
# part b: create matrix
results <-
  matrix(nrow = rounds,
         ncol = length(result_cols))
# part c: actually name columns in results marix
colnames(results) <- result_cols
# part d: convert to df (so multiple variables of different types can be stored)
results <- data.frame(results)
# -----------------------------------------------------------------------------
# STEP 2: start timer
start_time <- Sys.time()
# -----------------------------------------------------------------------------
# STEP 3: create rounds number of models, and store results each time
for (i in 1:rounds){
  # part c: use caret "train" function to train logistic regression model
  model <- train(form = formula,
                 data = train_data,
                 method = "nnet",
                 tuneGrid = tune_grid_neural,
                 trControl = train_control_neural,
                 metric = "Accuracy", # how to select among models
                 trace = FALSE,
                 maxit = 100,
                 MaxNWts = max_weights_neural)
  # part d: make predictions
  preds <- predict(object = model,
                   newdata = test_data,
                   type = "raw")
  # part e: store model performance
  conf_m <- confusionMatrix(data = as.factor(preds),
                            reference = test_data$Exited,
                            positive ="Exited")
  # part f: store model results
  # model type
  results[i, 1] <- "neural"
  # round
  results[i, 2] <- i
  # accuracy
  results[i, 3] <- conf_m$overall[1]
  # Kappa 
  results[i, 4] <- conf_m$overall[2]
  # accuracy LL
  results[i, 5] <- conf_m$overall[3]
  # accuracy UL
  results[i, 6] <- conf_m$overall[4]
  # sensitivity
  results[i, 7] <- conf_m$byClass[1]
  # specificity
  results[i, 8] <- conf_m$byClass[2]
  # precision
  results[i, 9] <- conf_m$byClass[3]
  # negative predictive value
  results[i, 10] <- conf_m$byClass[4]
  # F1 Score
  results[i, 11] <- conf_m$byClass[7]
  # sample size (of test set)
  results[i, 12] <- sum(conf_m$table)
  
  # part g: print round and total elapsed time so far
  cumul_time <- difftime(Sys.time(), start_time, units = "mins")
  print(paste("round #", i, ": cumulative time ", round(cumul_time, 2), " mins",
              sep = ""))
  print("--------------------------------------")
}
#> Imbalanced Classification problem

##################################################################################################################
## Oversampling
#> Keep majority class equal, oversample (4x) minority class
# -----------------------------------------------------------------------------
# STEP 0: set seed, so that statistics don't keep changing for every analysis
set.seed(1)
data_balanced_over <- ovun.sample(formula, data = train_data, method = "over",N = 11000)$data
table(data_balanced_over$Exited)
# -----------------------------------------------------------------------------
# STEP 1: decide how many times to run the model
rounds <- 10
# -----------------------------------------------------------------------------
# STEP 2: set up object to store results
# part a: create names of results to store
result_cols <- c("model_type", "round", "accuracy", "kappa", "accuracy_LL", "accuracy_UL",
                 "sensitivity", "specificity", "precision", "npv","F1", "n")
# part b: create matrix
results_over <- matrix(nrow = rounds,
                       ncol = length(result_cols))
# part c: actually name columns in results marix
colnames(results_over) <- result_cols
# part d: convert to df (so multiple variables of different types can be stored)
results_over <- data.frame(results_over)
# -----------------------------------------------------------------------------
# STEP 2: start timer
start_time <- Sys.time()
for (i in 1:rounds){
  # part c: use caret "train" function to train logistic regression model
  model_over <- train(form = formula,
                      data = data_balanced_over,
                      method = "nnet",
                      tuneGrid = tune_grid_neural,
                      trControl = train_control_neural,
                      metric = "Accuracy", # how to select among models
                      trace = FALSE,
                      maxit = 100,
                      MaxNWts = max_weights_neural)
  # part d: make predictions
  preds_over <- predict(object = model_over,
                        newdata = test_data,
                        type = "raw")
  # part e: store model performance
  conf_m_over <- confusionMatrix(data = as.factor(preds_over),
                                 reference = test_data$Exited,
                                 positive = "Exited")
  # part f: store model results
  # model type
  results_over[i, 1] <- "neural"
  # round
  results_over[i, 2] <- i
  # accuracy
  results_over[i, 3] <- conf_m_over$overall[1]
  # Kappa 
  results_over[i, 4] <- conf_m_over$overall[2]
  # accuracy LL
  results_over[i, 5] <- conf_m_over$overall[3]
  # accuracy UL
  results_over[i, 6] <- conf_m_over$overall[4]
  # sensitivity
  results_over[i, 7] <- conf_m_over$byClass[1]
  # specificity
  results_over[i, 8] <- conf_m_over$byClass[2]
  # precision
  results_over[i, 9] <- conf_m_over$byClass[3]
  # negative predictive value
  results_over[i, 10] <- conf_m_over$byClass[4]
  # F1 Score
  results_over[i, 11] <- conf_m_over$byClass[7]
  # sample size (of test set)
  results_over[i, 12] <- sum(conf_m_over$table)
  
  # part g: print round and total elapsed time so far
  cumul_time <- difftime(Sys.time(), start_time, units = "mins")
  print(paste("round #", i, ": cumulative time ", round(cumul_time, 2), " mins",
              sep = ""))
  print("--------------------------------------")
}

###############################################################
## ROSE
#> Undersample majority class, oversample (2x) minority class
# -----------------------------------------------------------------------------
# STEP 0: set seed, so that statistics don't keep changing for every analysis
set.seed(1)
data_rose <- ROSE(formula, data = train_data, seed = 1)$data
table(data_rose$Exited)
# -----------------------------------------------------------------------------
# STEP 1: decide how many times to run the model
rounds <- 10
# -----------------------------------------------------------------------------
# STEP 2: set up object to store results
# part a: create names of results to store
result_cols <- c("model_type", "round", "accuracy", "kappa", "accuracy_LL", "accuracy_UL",
                 "sensitivity", "specificity", "precision", "npv","F1", "n")
# part b: create matrix
results_rose <-
  matrix(nrow = rounds,
         ncol = length(result_cols))
# part c: actually name columns in results marix
colnames(results_rose) <- result_cols
# part d: convert to df (so multiple variables of different types can be stored)
results_rose <- data.frame(results_rose)
# -----------------------------------------------------------------------------
# STEP 2: start timer
set.seed(1)
start_time <- Sys.time()
for (i in 1:rounds){
  # part c: use caret "train" function to train logistic regression model
  model_rose <- 
    train(form = formula,
          data = data_rose,
          method = "nnet",
          tuneGrid = tune_grid_neural,
          trControl = train_control_neural,
          metric = "Accuracy", # how to select among models
          trace = FALSE,
          maxit = 100,
          MaxNWts = max_weights_neural)
  # part d: make predictions
  preds_rose <- predict(object = model_rose,
                        newdata = test_data,
                        type = "raw")
  # part e: store model performance
  conf_m_rose <- confusionMatrix(data = as.factor(preds_rose),
                                 reference = test_data$Exited,
                                 positive = "Exited")
  # part f: store model results
  # model type
  results_rose[i, 1] <- "neural"
  # round
  results_rose[i, 2] <- i
  # accuracy
  results_rose[i, 3] <- conf_m_rose$overall[1]
  # Kappa 
  results_rose[i, 4] <- conf_m_rose$overall[2]
  # accuracy LL
  results_rose[i, 5] <- conf_m_rose$overall[3]
  # accuracy UL
  results_rose[i, 6] <- conf_m_rose$overall[4]
  # sensitivity
  results_rose[i, 7] <- conf_m_rose$byClass[1]
  # specificity
  results_rose[i, 8] <- conf_m_rose$byClass[2]
  # precision
  results_rose[i, 9] <- conf_m_rose$byClass[3]
  # negative predictive value
  results_rose[i, 10] <- conf_m_rose$byClass[4]
  # F1 Score
  results_rose[i, 11] <- conf_m_rose$byClass[7]
  # sample size (of test set)
  results_rose[i, 12] <- sum(conf_m_rose$table)
  
  # part g: print round and total elapsed time so far
  cumul_time <- difftime(Sys.time(), start_time, units = "mins")
  print(paste("round #", i, ": cumulative time ", round(cumul_time, 2), " mins",
              sep = ""))
  print("--------------------------------------")
}
model$bestTune
#> size = 12, decay = 0.1
model_over$bestTune
#> size = 20, decay = 0.001
model_rose$bestTune
#> size = 20, decay = 0.0001

#################################################################################
#### Final Model
# part a: set range of tuning parameters (layer size and weight decay)
grid_final <- expand.grid(size = c(model_rose$bestTune[1,1]),
                          decay = c(model_rose$bestTune[1,2]))
set.seed(1)
# -----------------------------------------------------------------------------
# STEP 2: SELECT TUNING METHOD
# set up train control object, which specifies training/testing technique
train_control_neural <- trainControl(method = "LGOCV",
                                     number = 50,
                                     classProbs = TRUE, 
                                     verboseIter = TRUE)
fit_final <- train(formula, 
                   data = data_rose, method = 'nnet', 
                   trControl = train_control_neural, 
                   tuneGrid= grid_final,
                   metric = "Accuracy", 
                   trace = FALSE,
                   maxit = 100,
                   MaxNWts = max_weights_neural)

results_test <- predict(fit_final, newdata=test_data)
conf_test <- confusionMatrix(results_test, test_data$Exited)

###########################################################################
## Plot Variable Importance
library(devtools)
#source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
gar <- garson(fit_final) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
###########################################################################
## Plot Neural Network
plotnet(fit_final)