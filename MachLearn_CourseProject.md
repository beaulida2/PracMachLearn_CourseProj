Coursera: Practical Machine Learning (course project)
========================================================

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of this project is to build a machine learning algorithm to predict activity quality (classe) from activity monitors.


# Creating a prediction model
## Loading data

First I will load all of the libraries need for the analyses.

```r
setwd("G:/Data Science projects - 2015/Coursera/Data Science Specialization/8) Practical Machine Learning/Course Project")

library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

Now I will go ahead and download both the training and testing data files and take a look at the data provided to build our model. The goal of the model is to use any variables provided to predict the manner in which a person did the exercise (classe).


```r
#download files from the urls provided
#train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#download.file(url=train_url, destfile="training.csv")

#test_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download.file(url=test_url, destfile="testing.csv")

#read in training and testing data
train <- read.csv("training.csv", na.strings=c("NA","#DIV/0!",""))
test <- read.csv("testing.csv", na.strings=c("NA","#DIV/0!",""))

names(train)
str(train)
summary(train)
summary(train$classe)#this is the outcome we want to predict
```

## Split training/testing data
Before we do anything, we will set aside a subset of our training data for cross validation (40%). 

```r
#we want to predict the 'classe' variable using any other variable to predict with

inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
myTrain <- train[inTrain, ]
myTest <- train[-inTrain, ]
dim(myTrain)
```

```
## [1] 11776   160
```

```r
dim(myTest)
```

```
## [1] 7846  160
```

```r
#some exploratory plots
#featurePlot(x=train[, 150:159], y = train$classe, plot = 'pairs')
```

## Feature selection
Now we can tranform the data to only include the variables we will need to build our model. We will remove variables with near zero variance, variables with mostly missing data, and variables that are obviously not useful as predictors.


```r
#first we will remove variables with mostly NAs (use threshold of >75%)
mytrain_SUB <- myTrain
for (i in 1:length(myTrain)) {
  if (sum(is.na(myTrain[ , i])) / nrow(myTrain) >= .75) {
    for (j in 1:length(mytrain_SUB)) {
      if (length(grep(names(myTrain[i]), names(mytrain_SUB)[j]))==1) {
        mytrain_SUB <- mytrain_SUB[ , -j]
      }
    }
  }
}

dim(mytrain_SUB)
```

```
## [1] 11776    60
```

```r
#names(mytrain_SUB)

#remove columns that are obviously not predictors
mytrain_SUB2 <- mytrain_SUB[,8:length(mytrain_SUB)]

#remove variables with near zero variance
NZV <- nearZeroVar(mytrain_SUB2, saveMetrics = TRUE)
NZV #all false, none to remove
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.162264    8.62771739   FALSE FALSE
## pitch_belt            1.000000   13.83322011   FALSE FALSE
## yaw_belt              1.112540   14.58050272   FALSE FALSE
## total_accel_belt      1.089308    0.23777174   FALSE FALSE
## gyros_belt_x          1.024970    1.02751359   FALSE FALSE
## gyros_belt_y          1.172272    0.54347826   FALSE FALSE
## gyros_belt_z          1.090476    1.35020380   FALSE FALSE
## accel_belt_x          1.106383    1.33322011   FALSE FALSE
## accel_belt_y          1.123110    1.14639946   FALSE FALSE
## accel_belt_z          1.015009    2.38620924   FALSE FALSE
## magnet_belt_x         1.040359    2.57302989   FALSE FALSE
## magnet_belt_y         1.069054    2.37771739   FALSE FALSE
## magnet_belt_z         1.020548    3.60054348   FALSE FALSE
## roll_arm             48.761905   19.27649457   FALSE FALSE
## pitch_arm            75.888889   22.46093750   FALSE FALSE
## yaw_arm              34.133333   21.62024457   FALSE FALSE
## total_accel_arm       1.060377    0.56046196   FALSE FALSE
## gyros_arm_x           1.032680    5.23097826   FALSE FALSE
## gyros_arm_y           1.490323    3.09103261   FALSE FALSE
## gyros_arm_z           1.060127    1.95312500   FALSE FALSE
## accel_arm_x           1.009901    6.39436141   FALSE FALSE
## accel_arm_y           1.251969    4.40726902   FALSE FALSE
## accel_arm_z           1.074074    6.43682065   FALSE FALSE
## magnet_arm_x          1.052632   11.12432065   FALSE FALSE
## magnet_arm_y          1.058824    7.18410326   FALSE FALSE
## magnet_arm_z          1.014706   10.51290761   FALSE FALSE
## roll_dumbbell         1.046512   86.93104620   FALSE FALSE
## pitch_dumbbell        2.232558   85.13926630   FALSE FALSE
## yaw_dumbbell          1.116883   86.75271739   FALSE FALSE
## total_accel_dumbbell  1.050060    0.36514946   FALSE FALSE
## gyros_dumbbell_x      1.051136    1.96161685   FALSE FALSE
## gyros_dumbbell_y      1.240793    2.20788043   FALSE FALSE
## gyros_dumbbell_z      1.016000    1.67289402   FALSE FALSE
## accel_dumbbell_x      1.019608    3.43919837   FALSE FALSE
## accel_dumbbell_y      1.018987    3.78736413   FALSE FALSE
## accel_dumbbell_z      1.068027    3.37126359   FALSE FALSE
## magnet_dumbbell_x     1.075472    8.88247283   FALSE FALSE
## magnet_dumbbell_y     1.294118    6.86990489   FALSE FALSE
## magnet_dumbbell_z     1.153846    5.57914402   FALSE FALSE
## roll_forearm         11.321782   15.15794837   FALSE FALSE
## pitch_forearm        58.641026   20.99184783   FALSE FALSE
## yaw_forearm          15.240000   14.15591033   FALSE FALSE
## total_accel_forearm   1.172000    0.58593750   FALSE FALSE
## gyros_forearm_x       1.015060    2.30129076   FALSE FALSE
## gyros_forearm_y       1.004167    5.99524457   FALSE FALSE
## gyros_forearm_z       1.114865    2.39470109   FALSE FALSE
## accel_forearm_x       1.175439    6.55570652   FALSE FALSE
## accel_forearm_y       1.033333    8.22860054   FALSE FALSE
## accel_forearm_z       1.009524    4.64504076   FALSE FALSE
## magnet_forearm_x      1.020408   11.93953804   FALSE FALSE
## magnet_forearm_y      1.204082   15.20889946   FALSE FALSE
## magnet_forearm_z      1.102564   13.40013587   FALSE FALSE
## classe                1.469065    0.04245924   FALSE FALSE
```

```r
keep <- names(mytrain_SUB2)
```

## Random Forest Model
I decided to use the random forest model to build my machine learning algorithm as it is appropriate for a classification problem as we have and based on information provided in class lectures this model tends to be more accurate than some other classification models.

Below I fit my model on my training data and then use my model to predict classe on my subset of data used for cross validation.


```r
#fit model- RANDOM FOREST
set.seed(223)

modFit <- randomForest(classe~., data = mytrain_SUB2)
print(modFit)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = mytrain_SUB2) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3345    2    0    0    1 0.0008960573
## B    9 2265    5    0    0 0.0061430452
## C    0   18 2033    3    0 0.0102239533
## D    0    0   28 1901    1 0.0150259067
## E    0    0    2    8 2155 0.0046189376
```

```r
#cross validation on my testing data
#out of sample error
predict1 <- predict(modFit, myTest, type = "class")
confusionMatrix(myTest$classe, predict1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    2    0    0    0
##          B    4 1513    1    0    0
##          C    0   13 1355    0    0
##          D    0    0   20 1266    0
##          E    0    0    2    3 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9943          
##                  95% CI : (0.9923, 0.9958)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9927          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9902   0.9833   0.9976   1.0000
## Specificity            0.9996   0.9992   0.9980   0.9970   0.9992
## Pos Pred Value         0.9991   0.9967   0.9905   0.9844   0.9965
## Neg Pred Value         0.9993   0.9976   0.9964   0.9995   1.0000
## Prevalence             0.2847   0.1947   0.1756   0.1617   0.1832
## Detection Rate         0.2842   0.1928   0.1727   0.1614   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9989   0.9947   0.9906   0.9973   0.9996
```

```r
#in sample error
predict_train <- predict(modFit, myTrain, type = "class")
confusionMatrix(myTrain$classe, predict_train)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
#modFit <- train(classe ~., method = "rf", trControl=trainControl(method = "cv", number = 4), data = mytrain_SUB)
```

## Error
As we can see from the model summaries above, when we run the model on our test data for cross validation we get an accuracy of 99.4% that we can estimate to be our out of sample error. When the model is fitted to the training data used to build the model it shows 100% accuracy, which we can assume as our in sample error. 


## Apply to final test set
Finally, we apply our model to the final test data. Upon submission all predictions were correct! 


```r
predict_FINAL <- predict(modFit, test, type = "class")
print(predict_FINAL)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files = function(x) {
  n = length(x)
  for (i in 1:n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE,row.names=FALSE, col.names=FALSE)
  }
}

pml_write_files(predict_FINAL)
```


