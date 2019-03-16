# R Code for the Essay

## Necessary Packages, data split and variable creation ##
library(caTools)
library(sqldf)
library(glmnet)
library(kknn)

ElectionsData$Win <- ifelse(ElectionsData$Vote > ElectionsData$OpponentVote, 1, 0)

ElectionsData$SpendDiff <- ElectionsData$MoneySpent - ElectionsData$OpponentSpent

ElectionsData$SpendDiffRatio<- (ElectionsData$MoneySpent -
ElectionsData$OpponentSpent)/pmin(ElectionsData$MoneySpent,
ElectionsData$OpponentSpent)

ElectionsData$RaisedDiff <- ElectionsData$MoneyRaised - ElectionsData$OpponentRaised

ElectionsData$RaisedDiffRatio<- (ElectionsData$MoneyRaised -
ElectionsData$OpponentRaised)/pmin(ElectionsData$MoneyRaised,
ElectionsData$OpponentRaised)

set.seed(300)
sample = sample.split(ElectionsData, SplitRatio = .85)
OldData = subset(ElectionsData, sample == TRUE)
NewData = subset(ElectionsData, sample == FALSE)

## Logistic Regression ##

##Matrices of the independent variables in the regression, split into training and test set
## respectively 

LogTrainingInput <- as.matrix(sqldf('SELECT MoneyRaised, MoneySpent, OpponentRaised,
OpponentSpent, Incumbency FROM OldData'))

LogTestInput <- as.matrix(sqldf('SELECT MoneyRaised, MoneySpent, OpponentRaised,
OpponentSpent, Incumbency FROM NewData'))

## Logistic regression ##

set.seed(100)
logreg <- cv.glmnet(LogTrainingInput, OldData$Win, family = c("binomial"), alpha = 0,
type.measure = 'class' , nfolds = 10)

bestlambda <- logreg$lambda.min

NewData$logpredict <- predict(logreg, type =
c('response'), newx = LogTestInput, s=bestlambda )

NewData$logPrediction <- ifelse(NewData$logpredict > 0.5, 1, 0)

table(NewData$logPrediction, NewData$Win)

## Checking spending and raising differences for incumbents ##

sum(ElectionsData$Incumbency == 1 & ElectionsData$SpendDiffRatio > 1 |
ElectionsData$Incumbency == 0 & ElectionsData$SpendDiffRatio < -1)

sum(ElectionsData$Incumbency == 1 & ElectionsData$RaisedDiffRatio > 1 |
ElectionsData$Incumbency == 0 & ElectionsData$RaisedDiffRatio < -1)

##K-Nearest Neighbor model##

## First, set response variable into factor ##
NewData$Win <- as.factor(NewData$Win)
OldData$Win <- as.factor(OldData$Win)

## Train the KNN model ##
TrainingKNN<- train.kknn(formula = Win ~ SpendDiffRatio, data = OldData, kcv = 10, kernel =
'rectangular', kmax = 10)

## Check the quality based on the Test Set ##
NewData$Prediction <- predict(TrainingKNN , newdata = NewData)

table(NewData$Prediction, NewData$Win)

## Graph of KNN Model ##
OldData$Pred <-as.vector(TrainingKNN$fitted.values[[8]])

OldData$Prediction <- ifelse(OldData$Pred == 1, "Candidate Wins", "Challenger Wins")

OldData$Outcome <- ifelse(OldData$Win == 1, "Candidate Wins", "Challenger Wins")

GraphingDataFrame <- subset(OldData, SpendDiffRatio< 50 & SpendDiffRatio > -40)

qplot(SpendDiffRatio, RaisedDiffRatio, colour = Outcome, shape = Prediction, data =
GraphingDataFrame, xlab = "Spending Difference Ratio", ylab = "Money Raised Difference
Ratio", main = "K-Nearest Neighbor Estimation")

## Check races with similar spending ##
NewData$PredDiff <- as.numeric(NewData$Win) - as.numeric(NewData$Prediction)

NewData$LogPredDiff <- as.numeric(NewData$Win) - as.numeric(NewData$logPrediction)

sum(NewData$SpendDiffRatio > -2 & NewData$SpendDiffRatio < 2 )

sum(NewData$SpendDiffRatio > -2 & NewData$SpendDiffRatio < 2 & NewData$PredDiff == 1)

sum(NewData$SpendDiffRatio > -2 & NewData$SpendDiffRatio < 2 & NewData$LogPredDiff
== 1)
