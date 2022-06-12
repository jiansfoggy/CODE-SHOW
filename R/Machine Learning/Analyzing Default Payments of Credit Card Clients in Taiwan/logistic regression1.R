UCC=read.xlsx("/Users/LoveChina/Documents/6242/Final Project/UCCtrain.xlsx",
              sheetName = 'S', header = TRUE)
UTT=read.xlsx("/Users/LoveChina/Documents/6242/Final Project/UCCtest.xlsx",
              sheetName = 'S', header = TRUE)
plot(UCC[,18:23])

library(MASS)
library(tree)
UCC[,2]=as.factor(UCC[,2])
UCC[,3]=as.factor(UCC[,3])
UCC[,4]=as.factor(UCC[,4])
UCC[,6]=as.factor(UCC[,6])
UCC[,7]=as.factor(UCC[,7])
UCC[,8]=as.factor(UCC[,8])
UCC[,9]=as.factor(UCC[,9])
UCC[,10]=as.factor(UCC[,10])
UCC[,11]=as.factor(UCC[,11])
UTT[,2]=as.factor(UTT[,2])
UTT[,3]=as.factor(UTT[,3])
UTT[,4]=as.factor(UTT[,4])
UTT[,6]=as.factor(UTT[,6])
UTT[,7]=as.factor(UTT[,7])
UTT[,8]=as.factor(UTT[,8])
UTT[,9]=as.factor(UTT[,9])
UTT[,10]=as.factor(UTT[,10])
UTT[,11]=as.factor(UTT[,11])


paint.tr<-tree(UCC$default.payment.next.month~.,UCC)
summary(paint.tr)

plot(paint.tr,type="uniform")
text(paint.tr)

PNP=UCC$default.payment.next.month
UCD=data.frame(UCC[,2:5],PNP)
ptree=tree(PNP~.,UCD)
summary(ptree)
plot(ptree,type="uniform")
text(ptree)

PayM=as.factor(sort(UCC[,6:11]))
BillM=apply(UCC[,12:17],1,mean)
AmtM=apply(UCC[,18:23],1,mean)
UCD=data.frame(UCC[,1:11],BillM,AmtM,PNP)
PTA=tree(PNP~.,UCD)
summary(PTA)
plot(PTA,type="uniform")
text(PTA)

for (i in 2:4) {
  UCC[,i]=as.factor(UCC[,i])
}

for (i in 6:11) {
  UCC[,i]=as.factor(UCC[,i])
}

library(ElemStatLearn)
library(ROCR)
LR.credi=glm(UCC$default.payment.next.month~., family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3-PAY_AMT6, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3-PAY_AMT6-PAY_2, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3-PAY_AMT6-PAY_2-PAY_6, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6-BILL_AMT5, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6-BILL_AMT5-PAY_AMT3, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6-BILL_AMT5-PAY_AMT3-EDUCATION, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6-BILL_AMT5-PAY_AMT3-EDUCATION-SEX, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6-BILL_AMT5-PAY_AMT3
             -EDUCATION-SEX-PAY_AMT1, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")
LR.credi=glm(UCC$default.payment.next.month~.-PAY_AMT4-PAY_4-BILL_AMT2-BILL_AMT1-BILL_AMT3
             -PAY_AMT6-PAY_2-PAY_6-BILL_AMT6-BILL_AMT5-PAY_AMT3
             -EDUCATION-SEX-PAY_AMT1-BILL_AMT4, family=binomial("logit"), data=UCC)
drop1(LR.credi, test = "Chisq")

LR.train=glm(UCC$default.payment.next.month~LIMIT_BAL+MARRIAGE+AGE+PAY_0+PAY_3+
    PAY_5+PAY_AMT2+PAY_AMT5, family=binomial("logit"), data=UCC)
MX=predict(LR.train, UTT, type="response")
MSE1=sum((UTT$default.payment.next.month-MX)^2)/nrow(UTT)
MSE1

LIMIT_BAL+MARRIAGE+AGE+PAY_0+PAY_3+PAY_5+PAY_AMT2+PAY_AMT5

imports85 <- read.csv("~/Downloads/imports-85.data.txt")
dealed=imports85[,-1:-2]
dealed=dealed[,-2:-10]
dealed=dealed[,-2:-8]
dealed=dealed[,-2:-3]