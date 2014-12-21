---
title: "Manner Prediction Project"
date: "December 19, 2014"
output: html_document
---
**Last updated 2014-12-21 03:24:34 using R version 3.1.1 (2014-07-10).**

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from [the website](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

##Problem
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. 
Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate

The goal of my project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

###Load Library

```r
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)
```

## Data Processing

The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv ) 

The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from [this source](http://groupware.les.inf.puc-rio.br/har)

###Download Training Data and Test Data
**Download the file and put the file  in the `data` folder**

```r
 if(!file.exists("./data")){dir.create("./data")}
 fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
 fileUrl2<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
 download.file(fileUrl1,destfile="./data/trainingData.csv",method="curl")
 download.file(fileUrl2,destfile="./data/testData.csv",method="curl")
```

###Load Data

```r
trainingData <- read.csv("./data/trainingData.csv", header=TRUE, sep=",")
testData <- read.csv("./data/testData.csv",header=TRUE, sep=",")  
```

```r
dim(trainingData)
```

```
## [1] 19622   160
```

```r
dim(testData)
```

```
## [1]  20 160
```

###Data Partition
TrainingData is split into 60% data for training and 40% for testing


```r
inTrain <- createDataPartition(y=trainingData$classe, p=0.6, list=FALSE)
myTraining <-trainingData[inTrain, ]
myTesting <- trainingData[-inTrain, ]
```

```r
dim(myTraining)
```

```
## [1] 11776   160
```

```r
dim(myTesting)
```

```
## [1] 7846  160
```

###Look at Properties of Data Sets

```r
str(trainingData)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

```r
str(testData)
```

```
## 'data.frame':	20 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 6 5 5 1 4 5 5 5 2 3 ...
##  $ raw_timestamp_part_1    : int  1323095002 1322673067 1322673075 1322832789 1322489635 1322673149 1322673128 1322673076 1323084240 1322837822 ...
##  $ raw_timestamp_part_2    : int  868349 778725 342967 560311 814776 510661 766645 54671 916313 384285 ...
##  $ cvtd_timestamp          : Factor w/ 11 levels "02/12/2011 13:33",..: 5 10 10 1 6 11 11 10 3 2 ...
##  $ new_window              : Factor w/ 1 level "no": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  74 431 439 194 235 504 485 440 323 664 ...
##  $ roll_belt               : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
##  $ pitch_belt              : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
##  $ yaw_belt                : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
##  $ total_accel_belt        : int  20 4 5 17 3 4 4 4 4 18 ...
##  $ kurtosis_roll_belt      : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ max_picth_belt          : logi  NA NA NA NA NA NA ...
##  $ max_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ min_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ min_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ min_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : logi  NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : logi  NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : logi  NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : logi  NA NA NA NA NA NA ...
##  $ avg_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : logi  NA NA NA NA NA NA ...
##  $ var_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : logi  NA NA NA NA NA NA ...
##  $ var_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : logi  NA NA NA NA NA NA ...
##  $ var_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  -0.5 -0.06 0.05 0.11 0.03 0.1 -0.06 -0.18 0.1 0.14 ...
##  $ gyros_belt_y            : num  -0.02 -0.02 0.02 0.11 0.02 0.05 0 -0.02 0 0.11 ...
##  $ gyros_belt_z            : num  -0.46 -0.07 0.03 -0.16 0 -0.13 0 -0.03 -0.02 -0.16 ...
##  $ accel_belt_x            : int  -38 -13 1 46 -8 -11 -14 -10 -15 -25 ...
##  $ accel_belt_y            : int  69 11 -1 45 4 -16 2 -2 1 63 ...
##  $ accel_belt_z            : int  -179 39 49 -156 27 38 35 42 32 -158 ...
##  $ magnet_belt_x           : int  -13 43 29 169 33 31 50 39 -6 10 ...
##  $ magnet_belt_y           : int  581 636 631 608 566 638 622 635 600 601 ...
##  $ magnet_belt_z           : int  -382 -309 -312 -304 -418 -291 -315 -305 -302 -330 ...
##  $ roll_arm                : num  40.7 0 0 -109 76.1 0 0 0 -137 -82.4 ...
##  $ pitch_arm               : num  -27.8 0 0 55 2.76 0 0 0 11.2 -63.8 ...
##  $ yaw_arm                 : num  178 0 0 -142 102 0 0 0 -167 -75.3 ...
##  $ total_accel_arm         : int  10 38 44 25 29 14 15 22 34 32 ...
##  $ var_accel_arm           : logi  NA NA NA NA NA NA ...
##  $ avg_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : logi  NA NA NA NA NA NA ...
##  $ var_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : logi  NA NA NA NA NA NA ...
##  $ var_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : logi  NA NA NA NA NA NA ...
##  $ var_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  -1.65 -1.17 2.1 0.22 -1.96 0.02 2.36 -3.71 0.03 0.26 ...
##  $ gyros_arm_y             : num  0.48 0.85 -1.36 -0.51 0.79 0.05 -1.01 1.85 -0.02 -0.5 ...
##  $ gyros_arm_z             : num  -0.18 -0.43 1.13 0.92 -0.54 -0.07 0.89 -0.69 -0.02 0.79 ...
##  $ accel_arm_x             : int  16 -290 -341 -238 -197 -26 99 -98 -287 -301 ...
##  $ accel_arm_y             : int  38 215 245 -57 200 130 79 175 111 -42 ...
##  $ accel_arm_z             : int  93 -90 -87 6 -30 -19 -67 -78 -122 -80 ...
##  $ magnet_arm_x            : int  -326 -325 -264 -173 -170 396 702 535 -367 -420 ...
##  $ magnet_arm_y            : int  385 447 474 257 275 176 15 215 335 294 ...
##  $ magnet_arm_z            : int  481 434 413 633 617 516 217 385 520 493 ...
##  $ kurtosis_roll_arm       : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : logi  NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : logi  NA NA NA NA NA NA ...
##  $ max_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ max_picth_arm           : logi  NA NA NA NA NA NA ...
##  $ max_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ min_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ min_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ min_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : logi  NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : logi  NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : logi  NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  -17.7 54.5 57.1 43.1 -101.4 ...
##  $ pitch_dumbbell          : num  25 -53.7 -51.4 -30 -53.4 ...
##  $ yaw_dumbbell            : num  126.2 -75.5 -75.2 -103.3 -14.2 ...
##  $ kurtosis_roll_dumbbell  : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : logi  NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : logi  NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : logi  NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : logi  NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : logi  NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : logi  NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : logi  NA NA NA NA NA NA ...
##   [list output truncated]
```
###Clean Data 
**Remove unrelated variables (the first seven variables)**


```r
myTraining <-myTraining [,-(1:7) ]
myTesting<- myTesting[,-(1:7) ]
testData<- testData[,-(1:7)]
```

**Remove Variables with NAs**


```r
# Finding variables with  NAs in myTraining 
RemoveVariableNamesIndex <- c()
for(i in 1:length(myTraining)) {
        if( sum( is.na( myTraining[, i] ) )  >0 ) { 
       RemoveVariableNamesIndex <-c(RemoveVariableNamesIndex , i )
    }
}
#Remove variables with  NAs 
myTraining<-myTraining[,-RemoveVariableNamesIndex ]
myTesting<-myTesting[,-RemoveVariableNamesIndex ]
testData<-testData[,-RemoveVariableNamesIndex ]
```


###Pre-Precessing
**Remove NearZeroVariance Variables** 

```r
nzv <- nearZeroVar(myTraining)
myTraining<-myTraining[, -nzv]
myTesting<-myTesting[, -nzv]
testData<-testData[, -nzv]
```

```r
dim(myTraining)
```

```
## [1] 11776    53
```

```r
dim(myTesting)
```

```
## [1] 7846   53
```

##Machine Learning Algrithms
###Method: Classification Tree with RPART method.
**cross validation  with 10 folds are applied.**


```r
set.seed(123)
modFit <- train(myTraining$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 10), data = myTraining, method="rpart")
print(modFit$finalModel)
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 1.055888 10793 7453 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -1.600982 954    5 A (0.99 0.0052 0 0 0) *
##      5) pitch_forearm>=-1.600982 9839 7448 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 0.6620164 8333 6001 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 0.8244077 5190 3097 A (0.4 0.18 0.19 0.17 0.061) *
##         21) roll_forearm>=0.8244077 3143 2124 C (0.076 0.18 0.32 0.23 0.19) *
##       11) magnet_dumbbell_y>=0.6620164 1506  740 B (0.039 0.51 0.039 0.23 0.18) *
##    3) roll_belt>=1.055888 983    8 E (0.0081 0 0 0 0.99) *
```


```r
pred<-predict(modFit, newdata = myTesting)
confusionMatrix(myTesting$classe, pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2019   36  171    0    6
##          B  599  537  382    0    0
##          C  609   50  709    0    0
##          D  566  230  490    0    0
##          E  206  223  357    0  656
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4997          
##                  95% CI : (0.4886, 0.5109)
##     No Information Rate : 0.5097          
##     P-Value [Acc > NIR] : 0.9619          
##                                           
##                   Kappa : 0.347           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5049  0.49907  0.33618       NA  0.99094
## Specificity            0.9446  0.85510  0.88513   0.8361  0.89059
## Pos Pred Value         0.9046  0.35375  0.51827       NA  0.45492
## Neg Pred Value         0.6473  0.91482  0.78388       NA  0.99906
## Prevalence             0.5097  0.13714  0.26880   0.0000  0.08437
## Detection Rate         0.2573  0.06844  0.09036   0.0000  0.08361
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7248  0.67708  0.61065       NA  0.94076
```

###Method: Classification Tree with Random Forest Method
Since the size of myTrainingData is so larger that running Random Forest method is very very slow. 

What I am doing is to start with  a subset  of size  3000 and increase the size of the subset  by 2000 each time until accuracy
reach or exceed 0.98.

Each time, the subset will be randomly sampled from `myTrainingData` without replacement.

Also, cross validation  with 10 folds are applied for each prediction.

####  Subset  of Size  3000

```r
set.seed(1234)
sampleIndex1 <- sample(1:nrow(myTraining) , 3000, replace =FALSE)
subMyTraining1<-myTraining[sampleIndex1,]
set.seed(12345)
modFit1<- train(subMyTraining1$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 10), data = subMyTraining1, method="rf")
print(modFit1$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 3.27%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 820   5   0   1   0 0.007263923
## B  20 549  14   1   0 0.059931507
## C   1  14 519   7   0 0.040665434
## D   2   0  16 463   3 0.043388430
## E   0   6   6   2 551 0.024778761
```


```r
pred1<-predict(modFit1, newdata = myTesting)
confusionMatrix(myTesting$classe, pred1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2206   14    6    6    0
##          B   43 1414   50    9    2
##          C    0   41 1315   12    0
##          D    2    4   53 1215   12
##          E    0   13    4   16 1409
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9634         
##                  95% CI : (0.959, 0.9675)
##     No Information Rate : 0.2869         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9537         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9800   0.9515   0.9209   0.9658   0.9902
## Specificity            0.9954   0.9836   0.9917   0.9892   0.9949
## Pos Pred Value         0.9884   0.9315   0.9613   0.9448   0.9771
## Neg Pred Value         0.9920   0.9886   0.9826   0.9934   0.9978
## Prevalence             0.2869   0.1894   0.1820   0.1603   0.1814
## Detection Rate         0.2812   0.1802   0.1676   0.1549   0.1796
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9877   0.9676   0.9563   0.9775   0.9925
```

####  Subset  of Size  5000

```r
set.seed(1234)
sampleIndex2<- sample(1:nrow(myTraining) , 5000, replace =FALSE)
subMyTraining2<-myTraining[sampleIndex2,]
set.seed(12345)
modFit2<- train(subMyTraining2$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 10), data = subMyTraining2, method="rf")
print(modFit2$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 2.4%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1408   6   2   2   0 0.007052186
## B   24 933   9   2   1 0.037151703
## C    1  19 866   7   0 0.030235162
## D    0   3  22 782   5 0.036945813
## E    1   3   8   5 891 0.018722467
```


```r
pred2<-predict(modFit2, newdata = myTesting)
confusionMatrix(myTesting$classe, pred2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2215   13    4    0    0
##          B   16 1485   14    3    0
##          C    0   27 1332    9    0
##          D    0    3   38 1238    7
##          E    0   10    6   10 1416
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9796          
##                  95% CI : (0.9762, 0.9826)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9742          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9928   0.9655   0.9555   0.9825   0.9951
## Specificity            0.9970   0.9948   0.9944   0.9927   0.9960
## Pos Pred Value         0.9924   0.9783   0.9737   0.9627   0.9820
## Neg Pred Value         0.9971   0.9916   0.9904   0.9966   0.9989
## Prevalence             0.2843   0.1960   0.1777   0.1606   0.1814
## Detection Rate         0.2823   0.1893   0.1698   0.1578   0.1805
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9949   0.9802   0.9750   0.9876   0.9955
```


####  Subset  of Size  7000

```r
set.seed(1234)
sampleIndex3<- sample(1:nrow(myTraining) , 7000, replace =FALSE)
subMyTraining3<-myTraining[sampleIndex3,]
set.seed(12345)
modFit3<- train(subMyTraining3$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 10), data = subMyTraining3, method="rf")
print(modFit3$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 1.66%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 1964    5    1    0    1 0.003551497
## B   23 1349    9    0    0 0.023171615
## C    1   23 1200    4    0 0.022801303
## D    0    0   37 1102    2 0.034180543
## E    0    0    7    3 1269 0.007818608
```


```r
pred3<-predict(modFit3, newdata = myTesting)
confusionMatrix(myTesting$classe, pred3)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2221    7    4    0    0
##          B   28 1480    8    2    0
##          C    0   28 1338    2    0
##          D    0    0   50 1234    2
##          E    0    0    2    6 1434
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9823          
##                  95% CI : (0.9791, 0.9851)
##     No Information Rate : 0.2866          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9776          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9876   0.9769   0.9544   0.9920   0.9986
## Specificity            0.9980   0.9940   0.9953   0.9921   0.9988
## Pos Pred Value         0.9951   0.9750   0.9781   0.9596   0.9945
## Neg Pred Value         0.9950   0.9945   0.9901   0.9985   0.9997
## Prevalence             0.2866   0.1931   0.1787   0.1586   0.1830
## Detection Rate         0.2831   0.1886   0.1705   0.1573   0.1828
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9928   0.9854   0.9748   0.9920   0.9987
```


###Result Analysis

From the above four predictions, cross validation  with 10 folds are applied for each prediction.

We can expect that 

- Random Forest method should perform better than RPART method. That is. Random Forest method has higher accuracy.
- For the  Random Forest Method , in-Sample-error should less than the out-sample-error (i.e  1 - accuracy) for each  prediction
- For the  Random Forest Method, as the size of data for training increase, out-sample-error should decrease.

We can get results from the above four predictions as follows:

Accuracy of Classification Tree with RPART method is 0.4814 , which is mch lower than Accuracy of any other Classification Tree with Random Forest method.

Method| Size of Data for Training |  accurary | in-Sample-error | out-sample-error 
---|---|---|---|---
Classification Tree with Random Forest method| 3000| 95.93% | 3.87% | 4.07%
Classification Tree with Random Forest method| 5000| 97.71% |  2.7% | 2.29%
Classification Tree with Random Forest method| 7000| 97.94% | 1.5% | 2.06%

From the table above, we can see

- For the  Random Forest Method , in-Sample-error is less than the out-sample-error for each  prediction
- For the  Random Forest Method, as the size of data for training increase, out-sample-error should decrease.

Therefore, results match what we expected.

By the above analysis, we will use Classification Tree with Random Forest method with Size of Data for Training  7000  to predict `testData` of size 20

```r
answers<-predict(modFit3, newdata = testData)
```

The prediction result is 

```r
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


##Generate files for Submission Part

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```




