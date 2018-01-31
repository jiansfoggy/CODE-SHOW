#table between GQ and Self-assessment
n12=c(1,1,0,2,5,1,12,24,1,13,72,35,0,5,23)
qqqq=matrix(n12,nrow = 5,ncol = 3,byrow = TRUE)

#fisher's test
nnnn=fisher.test(qqqq)
print(nnnn)
#That p-value is for the null hypothesis of independence between
#the two categorical variables. We reject the null of independence here.
# p-value = 1.631e-11, this means that the null hypothesis can be 
# rejected and there is huge difference between two variables
#they are independence.

#chi-square test
chisq.test(qqqq)
# we found that p-value = 6.443e-10 < 0.05, so the null hypothesis can be rejected
# GQ is not independence of Self-Assessment.

#table between GQ and self-assessment

#LAP and LPI
n30=c(0,1,0,0,11,2,4,7,3,0,
      1,11,5,15,19,34,15,2,5,0,
      5,18,14,7,2,6,1,0,0,7)
jc2=matrix(n12,nrow = 3,ncol = 10,byrow = TRUE)
#chi-square test
chisq.test(jc2)
#p-value < 2.2e-16
#LPI is not independence of Self-Assessment.
#0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,
n50=c(0,0,0,1,5,
      0,1,0,20,9,
      0,0,0,16,3,
      0,2,2,15,3,
      1,2,11,17,1,
      0,1,11,29,1,
      0,2,7,11,0,
      0,0,2,7,0,
      1,0,4,3,0,
      0,0,0,1,6)
jc3=matrix(n50,nrow = 10,ncol = 5,byrow = TRUE)
#chi-square test
chisq.test(jc3)

#naive bayes classifier
# set mirror
options(repos=structure(c(CRAN="http://cran.rstudio.com")))

if (!("shiny" %in% names(installed.packages()[,"Package"]))) {install.packages("shiny")}
suppressMessages(library(shiny, quietly = TRUE))

if (!("openintro" %in% names(installed.packages()[,"Package"]))) {install.packages("openintro")}
suppressMessages(library(openintro, quietly = TRUE))

if (!("plotrix" %in% names(installed.packages()[,"Package"]))) {install.packages("plotrix")}
suppressMessages(library(plotrix, quietly = TRUE))
#read data
library(e1071)
library(caret)
library(readxl)
library(sqldf)
evlt=read_excel ("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/R/project 3/language master/real one/zzdraft.xlsx",sheet="D1")

# build new data frame
GQ =evlt$`General Questionnaire`
LAP=evlt$`Self Assessment`
LPI=evlt$`Language Porficiency Interview`

GQ=c("S2")
LAP=c("Category 2")
LPI=c("Advanced Low")
testPart=data.frame(GQ,LAP,LPI)
ZongPing=evlt
ZongPing=data.frame(GQ,LAP,LPI)
str(ZongPing)

# divide them into 2 groups
sample.ind <- sample(2, nrow(ZongPing),replace = T,prob = c(0.7,0.3))
ZongPing.train <- ZongPing[sample.ind==1,]
ZongPing.test  <- ZongPing[sample.ind==2,]

# build model
model = naiveBayes(LPI~., data = ZongPing)
# see the relationship between GQ and LPI, LAP and LPI
model

# result=predict(model,ZongPing.test[sample(1:50,10,replace=FALSE),],type="raw")
result1=predict(model,ZongPing.test)
result2=predict(model,ZongPing.test,type="raw")
model = naiveBayes(LPI~., data = ZongPing)
result1=predict(model,testPart)
result2=predict(model,testPart,type="raw")
czd=c()
for (i in 1:nrow(result2)) {
  czd=append(czd,max(result2[i,]))
}
summary(result1)
JZ1=matrix(result1,length(result1),1,byrow = TRUE)
JZ2=matrix(czd,length(czd),1,byrow = TRUE)
RESULT=cbind(JZ1,JZ2)
RESULT
table(result1, ZongPing.test$LPI)

############################


#method 2
x=ZongPing[,-3]
y=ZongPing$LPI

modell <- train(x,y,'nb',trControl=trainControl(method='cv',number=10))
modell
predict(modell$finalModel,ZongPing.test)$class
yc=predict(modell$finalModel,testPart[,-3])$class
yc
FenBu=table(yc,y)
sum(diag(prop.table(FenBu)))
zh=0
for (i in 1:10) {
  zh=zh+sum(FenBu[,i])
}
zh
dj=sum(diag(FenBu))
dj/zh

output$prdctrslt <- renderPrint({
  testd <- data.frame(input$GQ,input$LAP)
  modell <- train(inputt[,-3],inputt$LAP,'nb',trControl=trainControl(method='cv',number=10))
  predict(modell$finalModel,testd)$class

})

output$prdctrslt <- renderPrint({
  GQ =inputt$GQ
  LAP=inputt$LAP
  LPI=inputt$LPI
  
  ZongPing=data.frame(GQ,LAP,LPI)
  
  # divide them into 2 groups
  sample.ind <- sample(2, nrow(ZongPing),replace = T,prob = c(0.7,0.3))
  ZongPing.train <- ZongPing[sample.ind==1,]
  ZongPing.test  <- ZongPing[sample.ind==2,]
  
  
  # build model
  model = naiveBayes(LPI~., data = ZongPing.train)
  
  #testd<-data.frame(input$GQ,input$LAP)
  result=predict(model,ZongPing.test,type="raw")
  #result=predict(model,ZongPing.test)
  #summary(model)
  JZ=matrix(result,length(result),1,byrow = TRUE)
  JZ
})

output$prdctrslt <- renderPrint({
  jg()
})