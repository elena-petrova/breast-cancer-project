##################
# Summary
##################
# 1/ Preparing data: factors, pre-processing
# 2/ Logistic regression and Accuracy & ROC curve
# 3/ Caret: Gradient boosting, Random forest, Neural Network 


# loading packages
library(magrittr) # enables to use pipe operators
library(caret) # for pre-processing/cross-validation and ml algrythmes
library(ROCR) # for ROC curve and AUROC

# loading data + first look
cancerData <- read.csv("CancerData.csv", header = T) 
str(cancerData)

# preparing for logistic regression
cancerData$group <- factor(cancerData$group)
id <- cancerData$id; cancerData$id <- NULL # non relevant for model

# pre-processing
cancerDataPP <- preProcess(x = cancerData, 
                           method = c("center", "scale")) 
cancerDataProcessed <- predict(cancerDataPP, cancerData) # stored

# splitting data into train / test set
set.seed(1) # for reproductibility of results
trainIndex <- createDataPartition(cancerDataProcessed$diagnosis.factor, 
                                  times = 1,
                                  p = .75,
                                  list = FALSE)
cancerTrain <- cancerDataProcessed[trainIndex,]
cancerTest <- cancerDataProcessed[-trainIndex,]
testValues <- cancerTest$diagnosis.factor

# running first model with all features taken
set.seed(1)
cancerFitAll <- glm(diagnosis.factor ~., 
                    family = binomial(link = "logit"), 
                    data = cancerTrain)
summary(cancerFitAll)

# checking relative importance of features
varImp(cancerFitAll) # the importance of each variable in model

# dropping some features 
set.seed(1)
cancerFitFew <- glm(diagnosis.factor ~. -size.unif -epi, 
                    family = binomial(link = "logit"), 
                    data = cancerTrain)
summary(cancerFitFew)

# comparing resulting models
anova(cancerFitAll, 
      cancerFitFew, 
      test="Chisq")

#  adding cross-validation and by the use of caret doing the same thing as above
set.seed(1)
fitControl <- trainControl(method = "repeatedcv",
                           #number of folds is 10 by default
                           repeats = 3, 
                           savePredictions = T)

glmCancerFit <- train(diagnosis.factor ~.-size.unif -epi, 
                      data = cancerTrain,
                      method = "glm",
                      family = "binomial",
                      trControl = fitControl)

glmFitAcc <- train(diagnosis.factor ~.-size.unif -epi, 
                   data = cancerTrain,
                   method = "glm",
                   metric = "Accuracy",
                   trControl = fitControl) %>% print

# Analysis: ROC curve + AUC
probROC <- predict(cancerFitFew, 
                   newdata = cancerTest, 
                   type="response")

predROC <- prediction(probROC,
                      testValues)

perfROC <- performance(predROC, 
                       measure = 'tpr', 
                       x.measure = 'fpr')

plot(perfROC, lwd = 2, colorize = TRUE) # rainbow!
lines(x = c(0, 1), y = c(0, 1), col = "black", lwd = 1)

auc = performance(predROC, measure = "auc")
auc = auc@y.values[[1]] %>% print

# Loading additional for model comparison
library(gbm)
library(randomForest)
library(nnet)

# fitControlProb: we are going to use it later for AUROC
set.seed(1)
fitControlProb <- trainControl(method = "repeatedcv",
                               repeats = 3, 
                               savePredictions = T, 
                               classProbs = T, # probability instead of response
                               summaryFunction = twoClassSummary)
# Gradient boosting: caret
gbmCancerFit <- train(diagnosis.factor ~.-size.unif -epi, 
                      data = cancerTrain, 
                      method = "gbm", 
                      trControl = fitControl,
                      verbose = F)

# Random forest: caret
rfCancerFit <- train(diagnosis.factor ~.-size.unif -epi, 
                     data = cancerTrain,
                     method = "rf",
                     trControl = fitControl)

# Neural network: caret
nnCancerFit <- train(diagnosis.factor ~.-size.unif -epi, 
                     data = cancerTrain,
                     method = "avNNet", 
                     trControl = fitControl, 
                     linout = T)

# comparing all four models
results <- resamples(list(RF = rfCancerFit, 
                          GBM = gbmCancerFit, 
                          NNET = nnCancerFit, 
                          GLM = glmCancerFit)) 
summary(results)

# summarizing p-values for pair-wise comparisons
diffs <- diff(results) 
summary(diffs)

# testing the most accurate model (for this case) - Random Forest
rfPredict <- predict(rfCancerFit, 
                     newdata = cancerTest)
confusionMatrix(rfPredict, testValues)

# ROC for Random Forect
rfFitROC <- train(diagnosis.factor ~.-size.unif -epi, 
                  data = cancerTrain,
                  method = "rf",
                  metric = "ROC",
                  trControl = fitControlProb) %>% print

## AUROC for Random Forest model is about 99.6%

###################
# Final thougts
###################
So it up to you which model to use for this dataset. All four models seem to generalize well to unseen observations in test set. 
# Though there is still a lot of things to do with this dataset: visualizations for example. 

#Thanks for reading,

#Data Geekette