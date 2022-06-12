
library(ElemStatLearn)
library(ROCR)
library(readxl)
UCC <- read_excel("~/Documents/6242/Final Project/UCCtrain.xlsx", 
                       sheet = "S")
UTT <- read_excel("~/Documents/6242/Final Project/UCCtest.xlsx", 
                  sheet = "S")
UTT=read.xlsx("/Users/LoveChina/Documents/6242/Final Project/UCCtest.xlsx",
              sheetName = 'S', header = TRUE)
Unpay.Train=UCC
Unpay.Test =UTT
names(UCC)

##did not put education as factor
Unpay.Train[,2]=as.factor(Unpay.Train[,2])
Unpay.Train[,4]=as.factor(Unpay.Train[,4])
Unpay.Train[,24]=as.factor(Unpay.Train[,24])

Unpay.Test[,2]=as.factor(Unpay.Test[,2])
Unpay.Test[,4]=as.factor(Unpay.Test[,4])
Unpay.Test[,24]=as.factor(Unpay.Test[,24])

LR.credi=glm(Unpay.Train$default.payment.next.month~., family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3-BILL_AMT5, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3-BILL_AMT5-BILL_AMT6, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3-BILL_AMT5-BILL_AMT6-PAY_6, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3-BILL_AMT5
            -BILL_AMT6-PAY_6-PAY_5, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3-BILL_AMT5
             -BILL_AMT6-PAY_6-PAY_5-EDUCATION, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(Unpay.Train$default.payment.next.month~.-PAY_AMT4-BILL_AMT2-PAY_AMT6-PAY_4-BILL_AMT3-SEX-PAY_AMT3-BILL_AMT5
             -BILL_AMT6-PAY_6-PAY_5-EDUCATION-LIMIT_BAL, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
###get mse
LR.train1=glm(UCC$default.payment.next.month~MARRIAGE+AGE+PAY_0+PAY_2+PAY_3+BILL_AMT1+
               BILL_AMT4+PAY_AMT1+PAY_AMT2+PAY_AMT5, family=binomial("logit"), data=UCC)
summary(LR.train1)
MX1=predict(LR.train1, UTT, type="response")
MSE1=sum((UTT$default.payment.next.month-MX1)^2)/nrow(UTT)
MSE1

pred.vals1 <- prediction(MX1, Unpay.Test$default.payment.next.month)

perf1 <- performance(pred.vals1, measure = "tpr", x.measure = "fpr")

## plot the ROC curve for the predicted response values by the fitted
##  model

plot(perf1, colorize=TRUE) 

performance(pred.vals1, measure = "auc")@y.values [[1]]



###Discriminant Analysis######
library(classifly)
Unpay.Train=UCC
Unpay.Test =UTT
Unpay.Train[,2] = as.factor(Unpay.Train[,2])
Unpay.Train[,4] = as.factor(Unpay.Train[,4])
Response1 = as.factor(Unpay.Train[,24])
Predictor1 = as.matrix(Unpay.Train[,1:23])
Predictor1 = as.matrix(Unpay.Train[,1:23])

Unpay.Test[,2] = as.factor(Unpay.Test[,2])
Unpay.Test[,4] = as.factor(Unpay.Test[,4])
Response2 = as.factor(Unpay.Test[,24])
Predictor2 = as.matrix(Unpay.Test[,1:23])

TrainD=data.frame(Response1, Predictor1)
TestD =data.frame(Response2, Predictor2)

library(MASS)
Discri = lda(Predictor1, Response1,data=TrainD)
Discri
Discri = lda(Unpay.Train[,1:23], Response1,data=TrainD)
, data=TrainD
##test predictive
discri=predict(Discri,  , prior = Discri$prior, method = "predictive")
discri=predict(Discri, Unpay.Test[,1:23], prior = Discri$prior, method = "predictive")

MSE2=sum((Unpay.Test[,24]-discri$x)^2)/length(Unpay.Test[,24])
MSE2
summary(discri)
ct <- table(discri$class,Response2)
# total percent correct
RightRate2=sum(diag(prop.table(ct)))
RightRate2

###ROC
pred.vals2 <- prediction(discri, TestD$Response2)

perf2 <- performance(pred.vals2, measure = "tpr", x.measure = "fpr")
plot(perf2, colorize=TRUE) 
performance(pred.vals2, measure = "auc")@y.values [[1]]

###do some plot
library(ggplot2)
library(scales)
library(gridExtra)

proplda = Discri$svd^2/sum(Discri$svd^2)
proplda
plda <- predict(Discri, Predictor2, prior = Discri$prior, method = "predictive")


dataset = data.frame(Response2,lda = plda$posterior)
dataset
dataset[,1]

dataset[,1]=factor(dataset[,1])

LD2=c(1:5000)
K1=dataset$lda.1
K2=dataset$LD1


p1 <- ggplot(dataset) + geom_point(aes(LD2,K1, colour = Response2, shape = Response2), size = 2.5) +
  labs(x = paste("LD1(", percent(proplda[1]), ")", sep=""),
       y = paste("LD2(", percent(proplda[1]), ")", sep=""))
p1

###K-Nearest Neighbor###########
###Hence KNN is a completely non-parametric approach: 
###no assumptions are made about the shape of the decision 
###boundary. There- fore, we can expect this approach to 
###dominate LDA and logistic regression when the decision 
###boundary is highly non-linear. On the other hand, KNN 
###does not tell us which predictors are important; we don’t
###get a table of coefficients
library(class)
library(ElemStatLearn)
Unpay.Train=UCC
Unpay.Test =UTT

Unpay.Train[,2]=as.factor(Unpay.Train[,2])
Unpay.Train[,4]=as.factor(Unpay.Train[,4])
train.X=Unpay.Train[,-24]

Unpay.Test[,2] = as.factor(Unpay.Test[,2])
Unpay.Test[,4] = as.factor(Unpay.Test[,4])
test.X =Unpay.Test [,-24]

Response1 = as.factor(Unpay.Train[,24])
Response2 = as.factor(Unpay.Test[,24])

set.seed (1)
KNum=c(1:100)
MSE3=rep(0,100)
RightRate3=rep(0,100)
for (i in 1:100) {
  knn.pred=knn(train.X,test.X,Response1, k=i)
  table(knn.pred,Response2)
  RightRate3[i]=mean(knn.pred==Response2)
  Fitted=as.numeric(knn.pred)
  Fitted=Fitted-1
  MSE3[i]=sum((Unpay.Test[,24]-Fitted)^2)/length(Unpay.Test[,24])
}

plot(RightRate3, type="p")
plot(MSE3, type='p')
MAXPoint=which.max(RightRate3)
MINPoint=which.min(MSE3)
MSE3[7]
##plot out the classification rate
RightRate3[7]
##get mse
MSE3=rep(0,100)
knn.pred=knn(train.X,test.X,Response1, k=7)
summary(knn.pred)
Fitted=as.numeric(knn.pred)
Fitted=Fitted-1
MSE3=sum((Unpay.Test[,24]-Fitted)^2)/length(Unpay.Test[,24])
MSE3

ct=table(knn.pred,Response2)
sum(diag(prop.table(ct)))
mean(knn.pred==Response2)

sum(Fitted)
5000-sum(Fitted)
sum(as.numeric(Response2)-1)
5000-sum(as.numeric(Response2)-1)


###do some plot
require(dplyr)
knn.pred=knn(train.X,test.X,Response1, k=7)
prob <- attr(knn.pred, "prob")
cl <- factor(c(rep("0",3917), rep("1",1083)))
dataf <- bind_rows(mutate(test.X,
                          prob=prob,
                          cls="1",
                          prob_cls=ifelse(classif==cls,
                                          1, 0)),
                   mutate(test.X,
                          prob=prob,
                          cls="0",
                          prob_cls=ifelse(classif==cls,
                                          1, 0)))
require(ggplot2)
ggplot(dataf) +
  geom_point(aes(train.X, col=cls),
             data = mutate(test.X, cls=classif),
             size=1.2) + 
  geom_contour(aes(train.X, z=prob_cls, group=cls, color=cls),
               bins=2,
               data=dataf) +
  geom_point(aes(x=x, y=y, col=cls),
             size=3,
             data=data.frame(train.X, cls=cl))


##do some plot
require(class)
mixture.example
knn.pred=knn(train.X,test.X,Response1, k=7)
prob <- attr(knn.pred, "prob")
prob <- ifelse(knn.pred=="2", prob, 1-prob)

prob7 <- matrix(prob, 1083 , 3917)
par(mar=rep(2,4))
contour(px1, px2, prob7, levels=0.5, labels="", xlab="", ylab="", main=
          "15-nearest neighbour", axes=FALSE)
points(x, col=ifelse(Repsonse2==1, "coral", "cornflowerblue"))
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=ifelse(prob15>0.5, "coral", "cornflowerblue"))
box()

###Support Vector Classifier
library(e1071)
set.seed(1)
Unpay.Train=UCC
Unpay.Test =UTT
Unpay.Train[,2] = factor(Unpay.Train[,2])
Unpay.Train[,4] = as.factor(Unpay.Train[,4])
Response1 = as.factor(Unpay.Train[,24])
Predictor1 = as.matrix(Unpay.Train[,1:23])

Unpay.Test[,2] = as.factor(Unpay.Test[,2])
Unpay.Test[,4] = as.factor(Unpay.Test[,4])
Response2 = as.factor(Unpay.Test[,24])
Predictor2 = as.matrix(Unpay.Test[,1:23])

TrainD=data.frame(Response1, Predictor1)
TestD =data.frame(Response2, Predictor2)

TrainD=data.frame(Response1, Unpay.Train[,1:23])
TestD =data.frame(Response2, Unpay.Test[,1:23])

tune.out1=tune(svm,Response1~.,data=TrainD,kernel="linear",
              ranges=list(cost=c(0.1,1,4,5,5.5,6)), decision.values=T)
summary(tune.out1)
tune.out1=tune(svm,Response1~.,data=TrainD,kernel="linear",
              ranges=list(cost=c(0.1, 1)))
summary(tune.out1)
tune.out2=tune(svm,Response1~.,data=TrainD,kernel="linear",
               ranges=list(cost=c(5)))
summary(tune.out2)
tune.out3=tune(svm,Response1~.,data=TrainD,kernel="linear",
              ranges=list(cost=c(5,10)))
summary(tune.out3)
tune.out4=tune(svm,Response1~.,data=TrainD,kernel="linear",
              ranges=list(cost=c(100)))
summary(tune.out4)

##choose the best cost
bestmod=tune.out1$best.model
summary(bestmod)

##turn to do predict for testing dataset
ypred=predict(bestmod, TestD,decision.values=TRUE)
SVCT=table(predict=ypred, truth=TestD$Response2)
SVCT
sum(diag(prop.table(SVCT)))

fitted=attributes(predict(bestmod,TestD,
                          decision.values=TRUE))$decision.values

pred.vals1 <- prediction(fitted, TestD[,"Response2"])

perf1 <- performance(pred.vals1, measure = "tpr", x.measure = "fpr")

## plot the ROC curve for the predicted response values by the fitted
##  model

plot(perf1, colorize=TRUE) 

performance(pred.vals1, measure = "auc")@y.values [[1]]

##get mse
yfit=as.numeric(ypred)-1
MSE4=sum((Unpay.Test[,24]-yfit)^2)/length(Unpay.Test[,24])
MSE4

##ROC
svmfit=svm(TrainD$Response1~., data=TrainD, kernel="linear", cost=4, scale=FALSE)
summary(svmfit)
ypred=predict(svmfit, TestD)

library(ROCR)

fitted=attributes(ypred)
fitted$
$decision.values
pred.vals <- prediction(fitted$decision.values, TestD$Response2)
perf <- performance(pred.vals, measure = "tpr", x.measure = "fpr")
plot(perf, colorize=TRUE)
performance(pred.vals, measure = "auc")@y.values [[1]]

rocplot(fitted,TestD$Response2)
fitted=attributes(ypred)$decision.values
rocplot(fitted,TestD$Response2,add=T,col="red")







###SIR
library(dr)
Unpay.Train=UCC
Unpay.Test =UTT
X.train=as.matrix(Unpay.Train[,-24])
X.test=as.matrix(Unpay.Test[,-24])
Y.train=as.factor(Unpay.Train[,24])
Y.test=as.factor(Unpay.Test[,24])
ttt=data.frame(Y.test,X.test)
out.sir<-dr(Y.train ~ X.train, method="sir", nslices=10)
pd=predict(out.sir, ttt)
out.sir$evectors[,1:4]
out.sir$evalues
# dimension test
summary(out.sir)
summary(out.sir)$test
# construct sufficient summary plot
b1.hat<-out.sir$evectors[,1]
xb.hat<-as.vector(X.train %*% b1.hat)
par(mfrow=c(1,2))
plot(xb, y, xlab="gamma'X", ylab="Y")
plot(xb.hat, Y.train, xlab="gamma.hat'X", ylab="Y")
plot(dr.directions(out.sir)[,1],Y.train,xlab="SIR 1")
