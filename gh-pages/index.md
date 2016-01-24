# Predicting Performance of an Exercise
ktrebach  
January 24, 2016  

#Executive Summary
The goal of this project is to predict the manner in which users perform a particular exercise. This is the "classe" variable in the training set. This report explains the variable selection process, the model selection process, cross validation and an estimate of the expected out of sample error. The final prediction model was used to predict 20 different test cases and I'm pleased to report an accuracy level of 100%.


#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. Data was collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

#Loading, Reading and Exploring the Data



When reading the data, we assign all missing data an NA and we are therefore able to explore the significance (or lack thereof) of when there is NA or missing data.  
 
 


Each data set has 160 variables.  The training set has 19622 observations and the testing set has 20 observations.  There are 100 variables that have exactly 97.93% of their observations as NA or missing.  It is noted that we only have NAs when new_window is "no".  According to the data documentation, for feature extraction, the authors used a sliding window approach with different lengths from 0.5 second to 2.5 seconds, with 0.5 second overlap. We will exclude these variables as they will have little predictive value.




Further exploration using the nearZeroVar function in the caret package reveals an additional variable, num_window with little to no variability.  The time stamp variables as well do not provide any predictive value and are removed as well.




We then perform a chi-squared test to see if user_name has any predictive value on classe.  Given the null hypothesis that they are independent of each other, we see from the test that the P-value is extremely tiny and we reject the null hypothesis they are independent and must retain the variable:


```
## 
## 	Pearson's Chi-squared test
## 
## data:  D
## X-squared = 242.521, df = 20, p-value < 2.2e-16
```

We will now look for very highly correlated variables.



We observe that 22 pairs have very high correlation, but it is too difficult to pick which ones to eliminate and it wouldn't be that many to eliminate anyway.

So, we are left with 53 variables with which to build our predictive model.

#Predictive Model and Testing

Per wikipedia, Random forests ... are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Given the relatively large number of variables that we are using to build our model and that we are aiming to make classifications, I have chosen the random forest method.  

In order to estimate the out of sample error estimate, a cross validation data set is established. 



Here are the dimensions of the datasets: training2:  13737, 54 and validation: 5885, 54.
 
Given the computational expense of such a large dataset, i will further subset the training2 dataset into a smaller training set (25% of the data) to build the Random Forest model.  The results of the final model call and out of sample error estimate (OOB) are shown below:



```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 29
## 
##         OOB estimate of  error rate: 2.79%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 971   5   1   0   0 0.006141249
## B  18 634  10   3   0 0.046616541
## C   1  15 575   8   0 0.040066778
## D   1   1  14 544   3 0.033747780
## E   0   4   6   6 616 0.025316456
```

The results of the model on the balance of the training set is strong with an accuracy above 95%: 



```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2910   81    0    0    2
##          B   14 1864   53    2    5
##          C    3   45 1723   53   14
##          D    1    3   21 1630   18
##          E    1    0    0    4 1854
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9689          
##                  95% CI : (0.9654, 0.9722)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9607          
##  Mcnemar's Test P-Value : 7.89e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9935   0.9353   0.9588   0.9651   0.9794
## Specificity            0.9887   0.9911   0.9865   0.9950   0.9994
## Pos Pred Value         0.9723   0.9618   0.9374   0.9743   0.9973
## Neg Pred Value         0.9974   0.9846   0.9913   0.9932   0.9954
## Prevalence             0.2843   0.1935   0.1744   0.1640   0.1838
## Detection Rate         0.2825   0.1810   0.1673   0.1582   0.1800
## Detection Prevalence   0.2906   0.1881   0.1784   0.1624   0.1805
## Balanced Accuracy      0.9911   0.9632   0.9726   0.9800   0.9894
```

We now perform the cross validation on the validation test set and the results are also strong as expected with an accuracy rate also above 95%.



```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1663   48    1    0    1
##          B   10 1068   33    0    1
##          C    1   23  970   30    8
##          D    0    0   22  929   10
##          E    0    0    0    5 1062
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9672          
##                  95% CI : (0.9623, 0.9716)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9585          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9934   0.9377   0.9454   0.9637   0.9815
## Specificity            0.9881   0.9907   0.9872   0.9935   0.9990
## Pos Pred Value         0.9708   0.9604   0.9399   0.9667   0.9953
## Neg Pred Value         0.9974   0.9851   0.9885   0.9929   0.9958
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2826   0.1815   0.1648   0.1579   0.1805
## Detection Prevalence   0.2911   0.1890   0.1754   0.1633   0.1813
## Balanced Accuracy      0.9908   0.9642   0.9663   0.9786   0.9902
```

#Final Results
The final model was applied to the test set and the predicted classes had an accuracy of 100%.



