library(caret) #Install as package first if you do not have.
library(tidyverse)
options(scipen=999)#Turn off scientific notation as global setting

#Loading Data Files:
df_1 <- read.csv(file = "C:/Users/mahami13/Documents/R Predictive Modeling/Red_Wine.csv", header=T)
df_2 <- read.csv(file = "C:/Users/mahami13/Documents/R Predictive Modeling/White_Wine.csv", header=T)

#Merge/concat the two wine files:

df <- rbind(df_1, df_2)

#Step 1: Partition our Data:

set.seed(99) #set random seed
index <- createDataPartition(df$quality, p = .8,list = FALSE)
quality_train <-df[index,]
quality_test <- df[-index,]

#Step 2: Train/Fit the model:

set.seed(10) #set the seed again since within the train method the validation set is randomly selected

wine_lasso_model <- train(quality ~ .,
                     data = quality_train,
                     method = "glmnet",
                     standardize=TRUE,#standardize coefficients
                     tuneGrid=expand.grid(alpha=1,
                                          lambda=seq(0, 3, by = 0.1)),#add in grid of lambda values
                     trControl =trainControl(method = "cv",number=5))#5-fold cross validation

#list coefficients selected
coef(wine_lasso_model$finalModel, wine_lasso_model$bestTune$lambda)

# Step 3: Get predictions using testing set data:
wine_lasso_pred<-predict(wine_lasso_model, quality_test)

#Step 4: Evaluate model performance:
MSE<-mean((wine_lasso_pred - quality_test$quality)^2)
MSE

#Step 5: Combine predicted quality with original test data
quality_test$predicted_quality <- wine_lasso_pred

#Step 6: Sort the combined data by predicted quality in descending order
top_5_wines <- quality_test[order(quality_test$predicted_quality, decreasing = TRUE), ][1:5, ]

#Step 7: Print the top 5 wines
print(top_5_wines)

#Step 8: Print top 5 wines for new/holdout data-set:

new_wine_data <- read.csv(file = "C:/Users/mahami13/Downloads/StudentFinalWineSet.csv", header=T)

#Get Predictions on new dataset/holdout set
new_wine_pred <- predict(wine_lasso_model, newdata = new_wine_data)

#Combine the predicted quality with the original data
new_wine_data$predicted_quality <- new_wine_pred

#Sort the new dataset by predicted quality in descending order
sorted_new_wine_data <- new_wine_data[order(new_wine_data$predicted_quality, decreasing = TRUE), ]

#Get the top 5 wines for the new/holdout dataset
top_5_wines_2 <- sorted_new_wine_data[1:5, ]



