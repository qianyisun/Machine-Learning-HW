Homework 2
================
Qianyi Sun
February 5, 2020

# Install library

``` r
library('ElemStatLearn')  ## for 'prostate'
library('splines')        ## for 'bs'
library('dplyr')          ## for 'select', 'filter', and others
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library('magrittr')       ## for '%<>%' operator
library('glmnet')         ## for 'glmnet'
```

    ## Loading required package: Matrix

    ## Loaded glmnet 3.0-2

``` r
data('prostate')
cor(prostate)
```

    ##              lcavol      lweight       age         lbph         svi
    ## lcavol   1.00000000  0.280521386 0.2249999  0.027349703  0.53884500
    ## lweight  0.28052139  1.000000000 0.3479691  0.442264395  0.15538491
    ## age      0.22499988  0.347969112 1.0000000  0.350185896  0.11765804
    ## lbph     0.02734970  0.442264395 0.3501859  1.000000000 -0.08584324
    ## svi      0.53884500  0.155384906 0.1176580 -0.085843238  1.00000000
    ## lcp      0.67531048  0.164537146 0.1276678 -0.006999431  0.67311118
    ## gleason  0.43241706  0.056882099 0.2688916  0.077820447  0.32041222
    ## pgg45    0.43365225  0.107353790 0.2761124  0.078460018  0.45764762
    ## lpsa     0.73446033  0.433319385 0.1695928  0.179809404  0.56621822
    ## train   -0.04654347 -0.009940651 0.1776155 -0.029939957  0.02679950
    ##                  lcp     gleason      pgg45        lpsa        train
    ## lcavol   0.675310484  0.43241706 0.43365225  0.73446033 -0.046543468
    ## lweight  0.164537146  0.05688210 0.10735379  0.43331939 -0.009940651
    ## age      0.127667752  0.26889160 0.27611245  0.16959284  0.177615517
    ## lbph    -0.006999431  0.07782045 0.07846002  0.17980940 -0.029939957
    ## svi      0.673111185  0.32041222 0.45764762  0.56621822  0.026799505
    ## lcp      1.000000000  0.51483006 0.63152825  0.54881317 -0.037427296
    ## gleason  0.514830063  1.00000000 0.75190451  0.36898681 -0.044171456
    ## pgg45    0.631528246  0.75190451 1.00000000  0.42231586  0.100516371
    ## lpsa     0.548813175  0.36898681 0.42231586  1.00000000 -0.033889743
    ## train   -0.037427296 -0.04417146 0.10051637 -0.03388974  1.000000000

## split prostate into testing and training subsets

``` r
prostate_train <- prostate %>%
  filter(train == TRUE) %>% 
  select(-train)

summary(prostate_train)
```

    ##      lcavol           lweight           age             lbph         
    ##  Min.   :-1.3471   Min.   :2.375   Min.   :41.00   Min.   :-1.38629  
    ##  1st Qu.: 0.4883   1st Qu.:3.330   1st Qu.:61.00   1st Qu.:-1.38629  
    ##  Median : 1.4679   Median :3.599   Median :65.00   Median :-0.05129  
    ##  Mean   : 1.3135   Mean   :3.626   Mean   :64.75   Mean   : 0.07144  
    ##  3rd Qu.: 2.3491   3rd Qu.:3.884   3rd Qu.:69.00   3rd Qu.: 1.54751  
    ##  Max.   : 3.8210   Max.   :4.780   Max.   :79.00   Max.   : 2.32630  
    ##       svi              lcp             gleason          pgg45       
    ##  Min.   :0.0000   Min.   :-1.3863   Min.   :6.000   Min.   :  0.00  
    ##  1st Qu.:0.0000   1st Qu.:-1.3863   1st Qu.:6.000   1st Qu.:  0.00  
    ##  Median :0.0000   Median :-0.7985   Median :7.000   Median : 15.00  
    ##  Mean   :0.2239   Mean   :-0.2142   Mean   :6.731   Mean   : 26.27  
    ##  3rd Qu.:0.0000   3rd Qu.: 0.9948   3rd Qu.:7.000   3rd Qu.: 50.00  
    ##  Max.   :1.0000   Max.   : 2.6568   Max.   :9.000   Max.   :100.00  
    ##       lpsa        
    ##  Min.   :-0.4308  
    ##  1st Qu.: 1.6673  
    ##  Median : 2.5688  
    ##  Mean   : 2.4523  
    ##  3rd Qu.: 3.3652  
    ##  Max.   : 5.4775

``` r
prostate_test <- prostate %>%
  filter(train == FALSE) %>% 
  select(-train)
```

# fit the model

``` r
fit <- lm(lpsa ~ ., data=prostate_train)
```

# Use the testing subset to compute the test error using the fitted least-squares regression model.

``` r
L2_loss <- function(y, yhat)
  (y-yhat)^2
error <- function(dat, fit, loss=L2_loss)
  mean(loss(dat$lpsa, predict(fit, newdata=dat)))
error(prostate_test, fit)
```

    ## [1] 0.521274

# Train a ridge regression model using the glmnet function, and tune the value of lambda.

``` r
form  <- lpsa ~ 0 + lweight + age + lbph + lcp + pgg45 + lcavol + svi + gleason
x_inp <- model.matrix(form, data=prostate_train)
y_out <- prostate_train$lpsa
fit <- glmnet(x=x_inp, y=y_out, lambda=seq(0.5, 0, -0.05))
print(fit$beta)
```

    ## 8 x 11 sparse Matrix of class "dgCMatrix"

    ##    [[ suppressing 11 column names 's0', 's1', 's2' ... ]]

    ##                                                                        
    ## lweight .        0.006728858 0.08802267 0.16919400 0.25029493 0.3313693
    ## age     .        .           .          .          .          .        
    ## lbph    .        .           .          .          .          .        
    ## lcp     .        .           .          .          .          .        
    ## pgg45   .        .           .          .          .          .        
    ## lcavol  0.307213 0.346980360 0.37816116 0.40662942 0.42281361 0.4390147
    ## svi     .        .           .          0.01374089 0.08863503 0.1635049
    ## gleason .        .           .          .          .          .        
    ##                                                                      
    ## lweight 0.4026885474 0.44291760 0.483216170  0.532255854  0.613958878
    ## age     .            .          .           -0.002935469 -0.019001448
    ## lbph    0.0074378091 0.03989478 0.072310764  0.107605677  0.144874839
    ## lcp     .            .          .            .           -0.206198792
    ## pgg45   0.0001593199 0.00120135 0.002243793  0.003462562  0.009456602
    ## lcavol  0.4532475323 0.45792315 0.462701625  0.470160503  0.576448246
    ## svi     0.2420884074 0.32631404 0.410311213  0.490043681  0.737284632
    ## gleason .            .          .            .           -0.029295150

# Create a path diagram of the ridge regression analysis

``` r
plot(x=range(fit$lambda),
     y=range(as.matrix(fit$beta)),
     type='n',
     xlab=expression(lambda),
     ylab='Coefficients')
for(i in 1:nrow(fit$beta)) {
  points(x=fit$lambda, y=fit$beta[i,], pch=19, col='#00000055')
  lines(x=fit$lambda, y=fit$beta[i,], col='#00000055')
}
abline(h=0, lty=3, lwd=2)
```

![](Homework-2_files/figure-gfm/unnamed-chunk-7-1.png)<!-- --> \# Create
a figure that shows the training and test error associated with ridge
regression as a function of lambda

``` r
error <- function(dat, fit, lam, form, loss=L2_loss) {
  x_inp <- model.matrix(form, data=dat)
  y_out <- dat$lcavol
  y_hat <- predict(fit, newx=x_inp, s=lam)  ## see predict.elnet
  mean(loss(y_out, y_hat))
}

## compute training and testing errors as function of lambda
err_train_1 <- sapply(fit$lambda, function(lam) 
  error(prostate_train, fit, lam, form))
err_test_1 <- sapply(fit$lambda, function(lam) 
  error(prostate_test, fit, lam, form))

## plot test/train error
plot(x=range(fit$lambda),
     y=range(c(err_train_1, err_test_1)),
     type='n',
     xlab=expression(lambda),
     ylab='train/test error')
points(fit$lambda, err_train_1, pch=19, type='b', col='darkblue')
points(fit$lambda, err_test_1, pch=19, type='b', col='darkred')
legend('topleft', c('train','test'), lty=1, pch=19,
       col=c('darkblue','darkred'), bty='n')
```

![](Homework-2_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
colnames(fit$beta) <- paste('lam =', fit$lambda)
print(fit$beta %>% as.matrix)
```

    ##         lam = 0.5  lam = 0.45  lam = 0.4 lam = 0.35  lam = 0.3 lam = 0.25
    ## lweight  0.000000 0.006728858 0.08802267 0.16919400 0.25029493  0.3313693
    ## age      0.000000 0.000000000 0.00000000 0.00000000 0.00000000  0.0000000
    ## lbph     0.000000 0.000000000 0.00000000 0.00000000 0.00000000  0.0000000
    ## lcp      0.000000 0.000000000 0.00000000 0.00000000 0.00000000  0.0000000
    ## pgg45    0.000000 0.000000000 0.00000000 0.00000000 0.00000000  0.0000000
    ## lcavol   0.307213 0.346980360 0.37816116 0.40662942 0.42281361  0.4390147
    ## svi      0.000000 0.000000000 0.00000000 0.01374089 0.08863503  0.1635049
    ## gleason  0.000000 0.000000000 0.00000000 0.00000000 0.00000000  0.0000000
    ##            lam = 0.2 lam = 0.15   lam = 0.1   lam = 0.05      lam = 0
    ## lweight 0.4026885474 0.44291760 0.483216170  0.532255854  0.613958878
    ## age     0.0000000000 0.00000000 0.000000000 -0.002935469 -0.019001448
    ## lbph    0.0074378091 0.03989478 0.072310764  0.107605677  0.144874839
    ## lcp     0.0000000000 0.00000000 0.000000000  0.000000000 -0.206198792
    ## pgg45   0.0001593199 0.00120135 0.002243793  0.003462562  0.009456602
    ## lcavol  0.4532475323 0.45792315 0.462701625  0.470160503  0.576448246
    ## svi     0.2420884074 0.32631404 0.410311213  0.490043681  0.737284632
    ## gleason 0.0000000000 0.00000000 0.000000000  0.000000000 -0.029295150
