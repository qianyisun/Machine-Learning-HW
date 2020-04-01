Homework 4
================
Qianyi Sun
April 1st, 2020

# Load dataset

``` r
library('ElemStatLearn')
library('randomForest')
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

# keep y column in int type for further calculation

``` r
vowel.train_int_y<-vowel.train
```

1.Convert the response variable in the “vowel.train” data frame to a
factor variable prior to training, so that “randomForest” does
classification rather than
    regression.

``` r
vowel.train$y
```

    ##   [1]  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1
    ##  [24]  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2
    ##  [47]  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3
    ##  [70]  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4
    ##  [93]  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5
    ## [116]  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6
    ## [139]  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7
    ## [162]  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8
    ## [185]  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9
    ## [208] 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10
    ## [231] 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11
    ## [254]  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1
    ## [277]  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2
    ## [300]  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3
    ## [323]  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4
    ## [346]  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5
    ## [369]  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6
    ## [392]  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7
    ## [415]  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8
    ## [438]  9 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9
    ## [461] 10 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10
    ## [484] 11  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11
    ## [507]  1  2  3  4  5  6  7  8  9 10 11  1  2  3  4  5  6  7  8  9 10 11

``` r
vowel.train$y<- as.factor(vowel.train$y)
str(vowel.train)
```

    ## 'data.frame':    528 obs. of  11 variables:
    ##  $ y   : Factor w/ 11 levels "1","2","3","4",..: 1 2 3 4 5 6 7 8 9 10 ...
    ##  $ x.1 : num  -3.64 -3.33 -2.12 -2.29 -2.6 ...
    ##  $ x.2 : num  0.418 0.496 0.894 1.809 1.938 ...
    ##  $ x.3 : num  -0.67 -0.694 -1.576 -1.498 -0.846 ...
    ##  $ x.4 : num  1.779 1.365 0.147 1.012 1.062 ...
    ##  $ x.5 : num  -0.168 -0.265 -0.707 -1.053 -1.633 ...
    ##  $ x.6 : num  1.627 1.933 1.559 1.06 0.764 ...
    ##  $ x.7 : num  -0.388 -0.363 -0.579 -0.567 0.394 0.217 0.322 -0.435 -0.512 -0.466 ...
    ##  $ x.8 : num  0.529 0.51 0.676 0.235 -0.15 -0.246 0.45 0.992 0.928 0.702 ...
    ##  $ x.9 : num  -0.874 -0.621 -0.809 -0.091 0.277 0.238 0.377 0.575 -0.167 0.06 ...
    ##  $ x.10: num  -0.814 -0.488 -0.049 -0.795 -0.396 -0.365 -0.366 -0.301 -0.434 -0.836 ...

2.Review the documentation for the “randomForest” function. 3.Fit the
random forest model to the vowel data using all of the 11 features using
the default values of the tuning parameters.

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
    ##         OOB estimate of  error rate: 2.08%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   0 48  0  0  0  0  0  0  0  0  0  0.00000000
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 47  1  0  0  0  0  0  0.02083333
    ## 6   0  0  0  0  0 44  0  0  0  0  4  0.08333333
    ## 7   0  0  0  0  1  0 45  2  0  0  0  0.06250000
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  0 47  0  0  0.02083333
    ## 10  0  0  0  0  0  0  0  0  1 47  0  0.02083333
    ## 11  0  0  0  0  0  0  0  0  0  0 48  0.00000000

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

4.Use 5-fold CV and tune the model by performing a grid search for the
following tuning parameters: 1) the number of variables randomly sampled
as candidates at each split; consider values 3, 4, and 5, and 2) the
minimum size of terminal nodes; consider a sequence (1, 5, 10, 20, 40,
and 80).

``` r
a<-c(1, 5, 10, 20, 40, 80)
pre_tst<- NA
count <- 0
cverr<- NA
## 5-fold cross-validation of random forest model
## create five folds
set.seed(1985)
vowel.train_flds  <- createFolds(vowel.train$y, k=5)
vowel.train_flds_test  <- createFolds(vowel.train_int_y$y, k=5)
str(vowel.train_flds)
```

    ## List of 5
    ##  $ Fold1: int [1:105] 2 8 13 18 20 24 27 28 32 38 ...
    ##  $ Fold2: int [1:104] 3 4 7 9 19 23 43 46 49 60 ...
    ##  $ Fold3: int [1:106] 11 12 16 17 21 37 40 42 48 50 ...
    ##  $ Fold4: int [1:107] 1 5 6 15 26 29 33 34 35 36 ...
    ##  $ Fold5: int [1:106] 10 14 22 25 30 31 41 44 57 58 ...

``` r
print(vowel.train$y)
```

    ##   [1] 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1 
    ##  [24] 2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2 
    ##  [47] 3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3 
    ##  [70] 4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4 
    ##  [93] 5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5 
    ## [116] 6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6 
    ## [139] 7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7 
    ## [162] 8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8 
    ## [185] 9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9 
    ## [208] 10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10
    ## [231] 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11
    ## [254] 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1 
    ## [277] 2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2 
    ## [300] 3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3 
    ## [323] 4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4 
    ## [346] 5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5 
    ## [369] 6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6 
    ## [392] 7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7 
    ## [415] 8  9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8 
    ## [438] 9  10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9 
    ## [461] 10 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10
    ## [484] 11 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11
    ## [507] 1  2  3  4  5  6  7  8  9  10 11 1  2  3  4  5  6  7  8  9  10 11
    ## Levels: 1 2 3 4 5 6 7 8 9 10 11

``` r
cvknnreg <- function(kNN = 10, flds=vowel.train_flds) {
  for(tst_idx in 1:length(flds)) { ## for each fold
    
    ## get training and testing data
    vowel.train_trn <- vowel.train[-flds[[tst_idx]],]
    vowel.train_tst <- vowel.train[ flds[[tst_idx]],]
    vowel.train_tst$y<-as.integer(vowel.train_tst$y)
    
    ## fit random forest model to training data and turn the model
    for (i in 3:5){
      for (j in a ){
        count = count +1
    randomForest_fit <- randomForest(y ~ ., data = vowel.train,mtry = i, nodesize=j)
    
        ## compute test error on testing data
    pre_tst[count] <- predict(randomForest_fit, vowel.train_tst)
    pre_tst[count]<-as.integer(pre_tst[count])
    cverr[count] <- mean((vowel.train_tst$y - pre_tst[count])^2)
      }
    }
  }
  return(cverr)
}
```
