# Opioid Clinical Tool: PREDICTIVE MODELS
# Naive Bayes, XGBoost, Elastic Net, Neural Net, Random Forest Model, Gradient Boosting Model
# CLARE GRAZAL, M.S.
# FEBRUARY 2021

############
#LOAD DATA
############
data <- read.csv("completeDataset.csv")
data$X <- NULL

#Convert data types
convertFactor <- c(2:12,14:17)
convertNum <- c(1,13)
data[,convertFactor] <- data.frame(apply(data[convertFactor], 2, as.factor))
data[,convertNum] <- data.frame(apply(data[convertNum], 2, as.numeric))

################
#PREPROCESSING
################

##########################################
Split into balanced Test and Train groups
##########################################
library(caret)
set.seed(345)
trainIndex <- createDataPartition(data$longpost_dailydose, p = .8, 
                                  list = FALSE, 
                                  times = 1)
Train <- data[ trainIndex,]
Test  <- data[-trainIndex,]

#Check that outcome is proportionate
prop.table(table(Train$longpost_dailydose))
prop.table(table(Test$longpost_dailydose))

###########################################
Run MissForest imputation for missing data
###########################################
library(missForest)
#Standardize blank data
Train %>% replace_with_na_all(condition = ~.x %in% common_na_strings)
Test %>% replace_with_na_all(condition = ~.x %in% common_na_strings)

#run imputation algorithm on Train data
Train <- as.data.frame(Train)
Train.imp <- missForest(Train, maxiter = 10, ntree = 100, variablewise = FALSE,decreasing = FALSE, verbose = TRUE,
                        mtry = floor(sqrt(ncol(Train))), replace = TRUE,classwt = NULL, cutoff = NULL, strata = NULL,
                        sampsize = NULL, nodesize = NULL, maxnodes = NULL,xtrue = NA, parallelize = c('no'))
Train.complete <- Train.imp$ximp
head(Train.complete)
Train <- Train.complete

#run on Test data
Test <- as.data.frame(Test)
Test.imp <- missForest(Test, maxiter = 10, ntree = 100, variablewise = FALSE,decreasing = FALSE, verbose = TRUE,
                       mtry = floor(sqrt(ncol(Test))), replace = TRUE,classwt = NULL, cutoff = NULL, strata = NULL,
                       sampsize = NULL, nodesize = NULL, maxnodes = NULL,xtrue = NA, parallelize = c('no'))
Test.complete <- Test.imp$ximp
head(Test.complete)
Test <- Test.complete

##############################
Feature selection with Boruta
##############################
library(Boruta)
set.seed(234)
boruta.train <- Boruta(longpost_dailydose ~., data=Train, doTrace=2)
print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")
lz <- lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])

names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

getSelectedAttributes(final.boruta, withTentative = F)
boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)

selected <- getSelectedAttributes(final.boruta, withTentative = F)

library(dplyr)
Train <- Train %>% select(c(longpost_dailydose, selected))
Test <- Test %>% select(c(longpost_dailydose, selected))

#######################
Normalize numeric data
#######################
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x))) }

Train_norm_age <- as.data.frame(lapply(Train[1], normalize))
Train$PATAGE <- Train_norm_age$PATAGE
Train_norm_numdiag <- as.data.frame(lapply(Train[12], normalize))
Train$num_diagnoses <- Train_norm_numdiag$num_diagnoses
Test_norm_age <- as.data.frame(lapply(Test[1], normalize))
Test$PATAGE <- Test_norm_age$PATAGE
Test_norm_numdiag <- as.data.frame(lapply(Test[12], normalize))
Test$num_diagnoses <- Test_norm_numdiag$num_diagnoses

################################
Load necessary packages for ML
################################
set.seed(1234)
library(gbm)
library(missForest)
library(readxl)
library(dplyr)
library(caret)
library(Boruta)
library(ROSE)
library(class)
library(neuralnet)
library(xgboost)
library(Matrix)
library(glmnet)

write.csv(Train, "FinalTrainForML.csv")
write.csv(Test, "FinalTestForML.csv")

########################
#RUN PREDICTIVE MODELS
########################
Train <- read.csv("FinalTrainForML.csv")
Test <- read.csv("FinalTestForML.csv")
Train$X <- NULL
Test$X <- NULL
convertFactor <- c(2:11,13:17)
convertNum <- c(1,12)
Train[,convertFactor] <- data.frame(apply(Train[convertFactor], 2, as.factor))
Train[,convertNum] <- data.frame(apply(Train[convertNum], 2, as.numeric))
Test[,convertFactor] <- data.frame(apply(Test[convertFactor], 2, as.factor))
Test[,convertNum] <- data.frame(apply(Test[convertNum], 2, as.numeric))

#################
#1. NAIVE BAYES
#################
Train_NB <- Train
Test_NB <- Test

Train_NB$longpost_dailydose <- as.factor(Train_NB$longpost_dailydose)
Test_NB$longpost_dailydose <- as.factor(Test_NB$longpost_dailydose)

y <- Train_NB$longpost_dailydose
x <- Train_NB
x$longpost_dailydose <- NULL

levels(Train_NB$longpost_dailydose) <- c("No", "Yes")
levels(Test_NB$longpost_dailydose) <- c("No", "Yes")

#Set tuning/create model
set.seed(345)
trControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
grid <- data.frame(fL=c(0.0,0.25,0.5,0.75,1), usekernel=TRUE, adjust=c(0,0.25,0.5,0.75,1))

NBfit <- train(x=x, y=y,
               method     = "nb",
               trControl  = trControl,
               metric     = "Accuracy",
               tuneLength=10, tuneGrid=grid,
               importance=TRUE, data=Train_NB)
NBfit

#See variable list
vars <- varImp(NBfit, scale=TRUE)
varsImp <- as.matrix(vars$importance)
Features <- rownames(varsImp)
varsImp <- as.data.frame(varsImp)
varsImp$Features <- Features
varsImp$X0 <- NULL
colnames(varsImp) <- c("Importance", "Features")

#Save model
save(NBfit, file = "NB_Model.RData")

################
#2. XG BOOST
################
Train_XG <- Train
Test_XG <- Test

Train_XG$longpost_dailydose <- as.numeric(Train_XG$longpost_dailydose)
Test_XG$longpost_dailydose <- as.numeric(Test_XG$longpost_dailydose)
Train_XG$longpost_dailydose <- Train_XG$longpost_dailydose - 1
Test_XG$longpost_dailydose <- Test_XG$longpost_dailydose - 1

#Create matrix format
train_matrix_label <- (Train_XG$longpost_dailydose)
train_matrix_vars <- data.matrix(subset(Train_XG, select = -c(longpost_dailydose)))
test_matrix_label <- (Test_XG$longpost_dailydose)
test_matrix_vars <- data.matrix(subset(Test_XG, select = -c(longpost_dailydose)))

#Convert matrix to sparse matrix
train_sparse_vars <- as(train_matrix_vars, "dgCMatrix")
test_sparse_vars <- as(test_matrix_vars, "dgCMatrix")

#Create list format
train_XG <- list("data"=train_sparse_vars, "label"=train_matrix_label)
test_XG <- list("data"=test_sparse_vars, "label"=test_matrix_label)

#Check for 2 dimensions: data and label
str(train_XG)
str(test_XG)
dim(train_XG$data)
dim(test_XG$data)

#Check class; data should be "dgCMatrix" and label should be "numeric"
class(train_XG$data)[1]
class(train_XG$label)
class(test_XG$data)[1]
class(test_XG$label)

#Manually tune parameters
#try parameters: learning_rate =(), max_depth=(), min_child_weight=(), gamma=(), sub_sample=(), colsample_bytree=()

XG_cv <- xgb.cv(data = train_XG$data, label=train_XG$label, objective = "binary:logistic", 
                ntrees=2500, nrounds = 10, nthread = 5, nfold = 10, eta = 0.05, 
                learning_rate=0.1, 
                max_depth=7, 
                min_child_weight=1, 
                gamma=.01, 
                sub_sample=0.8,
                colsample_bytree=.7,
                metrics = list("auc") )

#Choose best combination
#Create final model
set.seed(10)
XG_model <- xgboost(data = train_XG$data, label=train_XG$label, objective = "binary:logistic", 
                    ntrees=2500, nrounds = 10, nthread = 5, nfold = 10, eta = 0.05, 
                    learning_rate=0.1, 
                    max_depth=7, 
                    min_child_weight=1, 
                    gamma=.01, 
                    sub_sample=0.8,
                    colsample_bytree=.7,
                    metrics = list("auc") )
XG_model

#See variable importance
importance <- xgb.importance(feature_names = colnames(train_XG), model = XG_model)
importanceRaw <- xgb.importance(feature_names = colnames(train_XG), model = XG_model, data = train_XG$data, label = train_XG$label)
importanceRaw$Gain <- importanceRaw$Gain * 100
importanceRaw$Cover <- importanceRaw$Cover * 100
importanceRaw$Frequency <- importanceRaw$Frequency * 100

#Save model
xgb.save(XG_model, 'xgb.model')

##################
#3. ELASTIC NET
##################
Train_EN <- Train
Test_EN <- Test

#Change levels to Yes and No
levels(Train_EN$longpost_dailydose)[levels(Train_EN$longpost_dailydose)=="0"] <- "No"
levels(Train_EN$longpost_dailydose)[levels(Train_EN$longpost_dailydose)=="1"] <- "Yes"
levels(Test_EN$longpost_dailydose)[levels(Test_EN$longpost_dailydose)=="0"] <- "No"
levels(Test_EN$longpost_dailydose)[levels(Test_EN$longpost_dailydose)=="1"] <- "Yes"

#Change to Matrix format
Train_EN_matrix <- model.matrix(data=Train_EN, ~ longpost_dailydose +PATAGE+COMBEN+PATRACE+RANKGRP+RSPONSVC+PATSEX+MARITAL+BEN_T3_REG+
                                  presurgical_dailydose+perisurgical_dailydose+
                                  postsurgical_dailydose+num_diagnoses+substance_abuse_diagnosis+
                                  procedure+PhysicalComorbidity+PsychologicalComorbidity)
Test_EN_matrix <- model.matrix(data=Test_EN, ~longpost_dailydose + PATAGE+COMBEN+PATRACE+RANKGRP+RSPONSVC+PATSEX+MARITAL+BEN_T3_REG+
                                 presurgical_dailydose+perisurgical_dailydose+
                                 postsurgical_dailydose+num_diagnoses+substance_abuse_diagnosis+
                                 procedure+PhysicalComorbidity+PsychologicalComorbidity)

#Create model
ENmodel <- cv.glmnet(x=Train_EN_matrix[,c(1,3:58)], y=Train_EN_matrix[,2], family="binomial", 
                     type.measure="auc", k=10, intercept=FALSE)
ENmodel

#See feature list
varImp <- function(object, lambda = NULL, ...) {
  beta <- predict(object, s = lambda, type = "coef")
  if(is.list(beta)) {
    out <- do.call("cbind", lapply(beta, function(x) x[,1]))
    out <- as.data.frame(out)
  } else out <- data.frame(Overall = beta[,1])
  out <- abs(out[rownames(out) != "(Intercept)",,drop = FALSE])
  out
}
varsImp <- varImp(ENmodel, lambda=ENmodel$lambda.min)
int <- as.matrix(coef(ENmodel,s="lambda.min"))
rownames(int)
varsImp$Variable <- rownames(int)
varsImp <- varsImp[order(varsImp$Overall, decreasing=TRUE),]

#Save model
save(ENmodel, file = "EN_Model.RData")


#################
#4. NEURAL NET
#################
Train_NN <- Train
Test_NN <- Test

Train_NN$longpost_dailydose <- as.factor(Train_NN$longpost_dailydose)
Test_NN$longpost_dailydose <- as.factor(Test_NN$longpost_dailydose)

#Set tuning/create model
trControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")

NNmodel <- train(longpost_dailydose~ 
                   PATAGE+COMBEN+PATRACE+RANKGRP+RSPONSVC+PATSEX+MARITAL+BEN_T3_REG+presurgical_dailydose+perisurgical_dailydose+
                   postsurgical_dailydose+num_diagnoses+substance_abuse_diagnosis+procedure+PhysicalComorbidity+PsychologicalComorbidity, 
                 data=Train_NN, method="nnet", trControl = trControl, importance= TRUE, tuneLength=10)
NNmodel

#See feature list
vars <- varImp(NNmodel,scale=TRUE)
varsImp <- as.matrix(vars$importance)
Features <- rownames(varsImp)
varsImp <- as.data.frame(varsImp)
varsImp$Features <- Features
varsImp$X0 <- NULL
colnames(varsImp) <- c("Importance", "Features")

#Save model
save(NNmodel, file = ".NN_Model.RData")

#See tools for NN: number of hidden layers, etc.
library(NeuralNetTools)
plotnet(NNmodel$finalModel, y_names="longpost_dailydose")
NNmodel$finalModel


#########################
#5. RANDOM FOREST MODEL
#########################
Train_RF <- Train
Test_RF <- Test

Train_RF$longpost_dailydose <- as.factor(Train_RF$longpost_dailydose)
Test_RF$longpost_dailydose <- as.factor(Test_RF$longpost_dailydose)

y <- Train_RF$longpost_dailydose
x <- Train_RF
x$longpost_dailydose <- NULL

levels(Train_RF$longpost_dailydose) <- c("No", "Yes")
levels(Test_RF$longpost_dailydose) <- c("No", "Yes")

#Set tuning/create model
trControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
tunegrid <- expand.grid(.mtry=(1:15))

RF_fit <- train(x=x, y=y, data=Train_RF, method="rf", trControl=trControl,metric="Accuracy", 
                importance=TRUE, tuneLength=10, tuneGrid=tunegrid)
RF_fit

#See feature list
vars <- varImp(RF_fit,scale=TRUE)
varsImp <- as.matrix(vars$importance)
Features <- rownames(varsImp)
varsImp <- as.data.frame(varsImp)
varsImp$Features <- Features
varsImp$X0 <- NULL
colnames(varsImp) <- c("Importance", "Features")

#Save model
save(RF_fit, file = "RF_Model.RData")


####################################
# GRADIENT BOOSTING MACHINE MODEL
####################################
Train_GBM <- Train
Test_GBM <- Test

#Change outcome format for GBM
Train_GBM$longpost_dailydose <- as.numeric(Train_GBM$longpost_dailydose)
Train_GBM$longpost_dailydose <- Train_GBM$longpost_dailydose - 1
Test_GBM$longpost_dailydose <- as.numeric(Test_GBM$longpost_dailydose)
Test_GBM$longpost_dailydose <- Test_GBM$longpost_dailydose - 1

###Tune model
hyper_grid <- expand.grid(shrinkage=c(.001, .01, .1), interaction.depth=c(1, 3, 5, 7),
                          n.minobsinnode=c(5, 10), bag.fraction=c(.8), optimal_trees=0,
                          min_CVLoss=0)
#Randomize data
random_index <- sample(1:nrow(Train_GBM), nrow(Train_GBM))
random_train <- Train_GBM[random_index,]

for (i in 1:nrow(hyper_grid)) {
  set.seed(123)
  gbm.tune <- gbm(formula=longpost_dailydose ~ ., distribution="bernoulli", 
                  data=random_train,
                  n.trees=2500, interaction.depth=hyper_grid$interaction.depth[i],
                  shrinkage=hyper_grid$shrinkage[i], 
                  n.minobsinnode=hyper_grid$n.minobsinnode[i], bag.fraction=hyper_grid$bag.fraction[i], train.fraction=.8,
                  n.cores=NULL, verbose=TRUE, cv.folds=10)
  hyper_grid$optimal_treesCV[i] <- which.min(gbm.tune$cv.error)
  hyper_grid$min_CVLoss[i] <- min(gbm.tune$cv.error) }
hyper_grid %>%
  dplyr::arrange(min_CVLoss) %>%
  head(5)

#Final Model using best parameters from above
hyper_grid <- expand.grid(shrinkage=c(.01), interaction.depth=c(7),
                          n.minobsinnode=c(5), bag.fraction=c(.8), optimal_trees=0,
                          min_ValidLoss=0)
for (i in 1:nrow(hyper_grid)) {
  set.seed(123)
  gbm.tune <- gbm(formula=longpost_dailydose ~ ., distribution="bernoulli", 
                  data=random_train,
                  n.trees=10000, interaction.depth=hyper_grid$interaction.depth[i],
                  shrinkage=hyper_grid$shrinkage[i], 
                  n.minobsinnode=hyper_grid$n.minobsinnode[i], bag.fraction=hyper_grid$bag.fraction[i], train.fraction=.8,
                  n.cores=NULL, verbose=TRUE, cv.folds=10) }

gbm.tune

#Check below for best performance
best.iter_cv <- gbm.perf(gbm.tune, method="cv")
best.iter_test <- gbm.perf(gbm.tune, method="test")

#See feature list
varsImp <- summary(gbm.tune, n.trees=best.iter_test)

#Save model
save(gbm.tune, file = "GBMmodel.RData")


#################
#GET PREDICTIONS
#################
CaretPredRF <- predict(RF_fit, newdata = Test_RF, type="prob")
Pred_RF <- CaretPredRF[,2]
roc.curve(Test_RF$longpost_dailydose, Pred_RF)
Test_RF$Preds <- Pred_RF

CaretPredNB <- predict(NBfit, newdata = Test_NB, type = "prob")
Pred_NB <- CaretPredNB[,2]
roc.curve(Test_NB$longpost_dailydose, Pred_NB)
Test_NB$Preds <- Pred_NB

Pred_XGBoost <- predict(XG_model, test_XG$data)
roc.curve(Test_XG$longpost_dailydose, Pred_XGBoost)
Test_XG$Preds <- Pred_XGBoost

Pred_EN <- predict(ENmodel, type="response", newx=Test_EN_matrix[,c(1,3:58)], s="lambda.min")
roc.curve(Test_EN$longpost_dailydose, Pred_EN)
Test_EN$Preds <- Pred_EN

CaretPredNN <- predict(NNmodel, newdata = Test_NN, type="prob")
Pred_NN <- CaretPredNN[,2]
roc.curve(Test_NN$longpost_dailydose, Pred_NN)
Test_NN$Preds <- Pred_NN

Pred_GBM <- predict(gbm.tune, newdata = Test_GBM, n.trees = best.iter_cv, type="response")
gbm.roc.area(Test_GBM$longpost_dailydose, Pred_GBM)
Test_GBM$Preds <- Pred_GBM

#####################
#RUN POST-ANALYSIS:
#####################
#1. AUC statistic with 95% CI, Delong Tests
#2. Brier Scores
#3. ROC Analysis and graphs
#4. Calibration curves
#5. DCA Graphs
#6. LIME Analysis and graphs for final model (GBM)
