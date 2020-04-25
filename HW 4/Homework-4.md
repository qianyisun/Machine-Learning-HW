Homework 4
================
Qianyi Sun
April 1st, 2020

\#\#Load dataset

``` r
library('ElemStatLearn')
library('randomForest')
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library('manipulate')
library('splines') ## 'ns'
library('caret')
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

## 1.Convert the response variable in the “vowel.train” data frame to a factor variable prior to training, so that “randomForest” does classification rather than regression.

``` r
vowel.train$y<- as.factor(vowel.train$y)
```

## 2.Review the documentation for the “randomForest” function.

``` r
?randomForest
```

## 3.Fit the random forest model to the vowel data using all of the 11 features using the default values of the tuning parameters.

``` r
randomForest(y ~ ., data = vowel.train)
```

    ## 
    ## Call:
    ##  randomForest(formula = y ~ ., data = vowel.train) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 2.65%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   1 47  0  0  0  0  0  0  0  0  0  0.02083333
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 46  1  0  0  0  0  1  0.04166667
    ## 6   0  0  0  0  0 44  0  0  0  0  4  0.08333333
    ## 7   0  0  0  0  1  0 46  1  0  0  0  0.04166667
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  1 46  0  0  0.04166667
    ## 10  0  0  0  0  0  0  1  0  0 47  0  0.02083333
    ## 11  0  0  0  0  0  1  0  0  0  0 47  0.02083333

## 4.Use 5-fold CV and tune the model by performing a grid search for the following tuning parameters: 1) the number of variables randomly sampled as candidates at each split; consider values 3, 4, and 5, and 2) the minimum size of terminal nodes; consider a sequence (1, 5, 10, 20, 40, and 80).

``` r
a<-c(1, 5, 10, 20, 40, 80)
pre_tst<- NA
cverr<- NA
## 5-fold cross-validation of random forest model
## create five folds
set.seed(1985)
vowel.train_flds  <- createFolds(vowel.train$y, k=5)
df_flds<-vowel.train_flds

cverr_min = 1 #intial value and the cverr will not more than 1
optimal_mtry = 0
optimal_nodesize = 0
    
    ## fit random forest model to training data and turn the model
    for (i in 3:5){
      for (j in a ){
        
    #  5-fold cross-validation
    cverr <- rep(NA, length(df_flds))
        
    for(tst_idx in 1:length(df_flds)) {

     ## get training and testing data
   train_trn <- vowel.train[-df_flds[[tst_idx]],]
   train_tst <- vowel.train[ df_flds[[tst_idx]],]
    randomForest_fit <- randomForest(y ~ ., data = train_trn,mtry = i, nodesize=j)
    pre_tst <- predict(randomForest_fit, train_tst[,2:11])

    classerror <- ifelse(train_tst$y != pre_tst, FALSE, TRUE)
    
    cverr[tst_idx] <- sum(classerror == FALSE) / length(classerror)
    
    }
    cur_cverr <- mean(cverr)
    
    if (cverr_min > cur_cverr){
      cverr_min <- cur_cverr
      optimal_mtry <- i
      optimal_nodesize <- j
    }
   }
    }
```

## 5.make predictions using the majority vote method, and compute the misclassification rate

``` r
# With the tuned model
optimal_model <- randomForest(y ~ ., data = vowel.train,mtry = optimal_mtry, nodesize=optimal_nodesize)
optimal_predict <- predict(optimal_model, vowel.test[,2:11])


# result
misclassification_rate <- 1 - mean(ifelse(vowel.test$y != optimal_predict, FALSE, TRUE))
misclassification_rate
```

    ## [1] 0.3961039
