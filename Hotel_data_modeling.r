Cancelation3<-read.csv('Hospitality_final2.csv')
#^This csv file was made after creating dummy variables on our filtered data set.

#Turning off scientific notation:
options(scipen=999)#Turn off scientific notation as global setting

library(caret)
library(tidyverse)
library(dplyr)
library(skimr)
library(lubridate)


#LASSO:

#Partitioning the data:
set.seed(99) #set random seed
index <- createDataPartition(Cancelation3$is_canceled, p = .8,list = FALSE)
Cancel_train <-Cancelation3[index,]
Cancel_test <- Cancelation3[-index,]

#Train/Fitting the Model:
library(e1071)
library(glmnet)
library(Matrix)
set.seed(10)#set the seed again since within the train method the validation set is randomly selected
Cancel_model <- train(is_canceled ~ .,
                      data = Cancelation3,
                      method = "glmnet",
                      standardize =T,
                      tuneGrid = expand.grid(alpha =1, #lasso
                                             lambda = seq(0.0001, 1, length = 20)),
                      trControl =trainControl(method = "cv",
                                              number = 5,
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")
Cancel_model         

#list coefficients selected
coef(Cancel_model$finalModel, Cancel_model$bestTune$lambda)

#Step 3: Get Predictions using Testing Set Data:
#First, get the predicted probabilities of the test data.
predprob_lasso<-predict(Cancel_model , Cancel_test, type="prob")

#Step 4: Evaluate Model Performance:
library(ROCR)

pred_lasso <- prediction(predprob_lasso$canceled, Cancel_test$is_canceled,label.ordering =c("notcanceled","canceled") )
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)

#Get the AUC
auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso



#Forward Subset-Selection:

set.seed(10)#set the seed again since within the train method the validation set is randomly selected
Cancel_model <- train(is_canceled ~ .,
                      data = Cancelation3,
                      method = "glmStepAIC",
                      direction="forward",
                      trControl =trainControl(method = "none",
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")

Cancel_model 

coef(Cancel_model$finalModel)

predprob_lasso<-predict(Cancel_model , Cancel_test, type="prob")

library(ROCR)

pred_lasso <- prediction(predprob_lasso$canceled, Cancel_test$is_canceled,label.ordering =c("notcanceled","canceled") )
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)

auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso

#Backward Subset-Selection:

set.seed(10)#set the seed again since within the train method the validation set is randomly selected
Cancel_model <- train(is_canceled ~ .,
                      data = Cancelation3,
                      method = "glmStepAIC",
                      direction="backward",
                      trControl =trainControl(method = "none",
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")

Cancel_model 

coef(Cancel_model$finalModel)

predprob_lasso<-predict(Cancel_model , Cancel_test, type="prob")

library(ROCR)

pred_lasso <- prediction(predprob_lasso$canceled, Cancel_test$is_canceled,label.ordering =c("notcanceled","canceled") )
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)

auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso

# XGBOOST:

#Partitioning the data:
set.seed(99) #set random seed
index <- createDataPartition(Cancelation3$is_canceled, p = .8,list = FALSE)
Cancel_train <-Cancelation3[index,]
Cancel_test <- Cancelation3[-index,]

#Fit/train the model:
library(xgboost)

library(doParallel)

#total number of cores on your computer
num_cores<-detectCores(logical=FALSE)
num_cores

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)

set.seed(8)
Cancel_XG_Model <- train(is_canceled~.,
                   data = Cancelation,
                   method = "xgbTree",
                   # provide a grid of parameters
                   tuneGrid = expand.grid(
                     nrounds = c(50,200),
                     eta = c(0.025, 0.05),
                     max_depth = c(2, 3),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   trControl= trainControl(method = "cv",
                                           number = 5,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary),
                   metric = "ROC"
)
#stop parallel processing
stopCluster(cl)
registerDoSEQ()

#only print top 10 important variables
plot(varImp(Cancel_XG_Model), top=10)

library(SHAPforxgboost)

Xdata<-as.matrix(select(Cancel_train,-is_canceled)) # change data to matrix for plots

# Calculate SHAP values
shap <- shap.prep(Cancel_XG_Model$finalModel, X_train = Xdata)

# SHAP importance plot for top 15 variables
shap.plot.summary.wrap1(Cancel_XG_Model$finalModel, X = Xdata, top_n = 10)

# Use 4 most important predictor variables. Still need to run.
top4<-shap.importance(shap, names_only = TRUE)[1:4]

for (x in top4) {
  p <- shap.plot.dependence(
    shap, 
    x = x, 
    color_feature = "auto", 
    smooth = FALSE, 
    jitter_width = 0.01, 
    alpha = 0.4
  ) +
    ggtitle(x)
  print(p)
}

#See the performance based on various tuning parameters.
plot(Cancel_XG_Model)

#To get a printout of the best tuning parameters use
Cancel_XG_Model$bestTune

#Get Predictions:
predprob_lasso<- predict(Cancel_XG_Model, Cancel_test, type="prob")

#Evaluate model:
pred_lasso <- prediction(predprob_lasso$canceled, Cancel_test$is_canceled,label.ordering =c("notcanceled","canceled") )
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)

auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso


#Random Forest:

#Train/Fitting the Model:
library(e1071)
library(glmnet)
library(Matrix)
set.seed(20)#set the seed again since within the train method the validation set is randomly selected
Cancel_model <- train(is_canceled ~ .,
                      data = Cancelation3,
                      method = "rf",
                      standardize =T,
                      #tuneGrid= expand.grid(mtry = c(1,2)),
                      tuneGrid= expand.grid(mtry = c(1, 3,6,9)),
                      #tuneGrid = expand.grid(alpha =1,lambda = seq(0.0001, 1, length = 20)),
                      trControl =trainControl(method = "cv",
                                              number = 5,
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")
#list coefficients selected
coef(Cancel_model$finalModel, Cancel_model$bestTune$lambda)



#Step 3: Get Predictions using Testing Set Data:
#First, get the predicted probabilities of the test data.
predprob_lasso<-predict(Cancel_model , Cancel_test, type="prob")



#Step 4: Evaluate Model Performance:
library(ROCR)



pred_lasso <- prediction(predprob_lasso$canceled, Cancel_test$is_canceled,label.ordering = c("notcanceled","canceled"))
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)



#Get the AUC
auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))





auc_lasso

