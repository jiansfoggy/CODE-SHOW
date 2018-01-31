#### Data InPUTING
library(devtools)
library(ElemStatLearn)
library(ROCR)
library(tree)
library(readxl)
library(sqldf)

test1 <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/test1.xlsx")
test2 <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/test2.xlsx")
A=sqldf("select * from test1 natural join test2")
new <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/nwtry.xlsx",sheet = "Sheet3")
new$SbRgn=A$`SubRegion`
new$Fre=A$`F/TC`
names(new)


####Data Cleaning

Q=as.factor(new$Quarter)
Gender=as.factor(new$Gender)
RACE=as.factor(new$Race)
Degree=as.factor(new$Degree)
Sector=as.factor(new$Sector)
MedSort=as.factor(new$MedSort)
SbRgn=as.factor(new$SbRgn)
Legal=as.factor(new$Legal)
HRStatus=as.factor(new$HRStatus)
nrow(new)
#Dstrbt=sample(c(0,1),nrow(new),replace = TRUE)

set.seed(981)
Dstrbt <- sample(2, nrow(new),replace = T,prob = c(0.7,0.3))

TD=data.frame(new$Age,Gender,RACE,Degree,new$GPA,new$ApproveNum,HRStatus)


TD=data.frame(Q,new$VR,new$Age,Gender,RACE,Degree,new$GPA,Sector,
              MedSort,new$ApproveNum,Legal,SbRgn,new$Fre,HRStatus)
write.table(TD, "//dcna_cifs.peacecorps.gov/users/jsun/Downloads/TD.txt", sep="\t")


train=TD[Dstrbt==1,]
test =TD[Dstrbt==2,]
train=TD[Dstrbt==1,-Dstrbt]
test =TD[Dstrbt==0,-Dstrbt]

###Random Forest

options(repos='http://cran.rstudio.org')
have.packages <- installed.packages()
cran.packages <- c('devtools','plotrix','randomForest','tree')
to.install <- setdiff(cran.packages, have.packages[,1])
if(length(to.install)>0) install.packages(to.install)
library(githubinstall)

if(!('reprtree' %in% installed.packages())){
  install_github('araastat','reprtree')
}
for(p in c(cran.packages, 'reprtree')) eval(substitute(library(pkg), list(pkg=p)))

install_github('araastat/reprtree')
githubinstall("reprtree")
library(reprtree)
library(randomForest)
library(e1071)
library(caret)
new <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/nwtry.xlsx",sheet = "Sheet3")
new$SbRgn=A$`Sub Region`
new$Fre=A$`F/TC`

Q=as.factor(new$Quarter)
Gender=as.factor(new$Gender)
RACE=as.factor(new$Race)
Degree=as.factor(new$Degree)
MedSort=as.factor(new$MedSort)
Sector=as.factor(new$Sector)
SbRgn=as.factor(new$SbRgn)
Legal=as.factor(new$Legal)
HRStatus=as.factor(new$HRStatus)

TD=data.frame(Q,SbRgn,new$Age,Gender,RACE,Degree,new$GPA,
              Sector,Legal,new$Fre,MedSort,new$ApproveNum,HRStatus)


set.seed(234)
sample.ind <- sample(2, nrow(TD),replace = T,prob = c(0.7,0.3))
inv.dev <- TD[sample.ind==1,]
inv.val <- TD[sample.ind==2,]

table(inv.dev$HRStatus)/nrow(inv.dev)
table(inv.val$HRStatus)/nrow(inv.val)

class(inv.dev$HRStatus)

#make formula
varNames <- names(inv.dev)
# Exclude ID or Response variable
varNames <- varNames[!varNames %in% c("HRStatus")]
# add + sign between exploratory variables
varNames1 <- paste(varNames, collapse = "+")
# Add response variable and convert to a formula object
rf.form <- as.formula(paste("HRStatus", varNames1, sep = " ~ "))

#Build it
inv.rf <- randomForest(rf.form,inv.dev,ntree=1000,importance=T)
print(inv.rf)
attributes(inv.rf)
plot(inv.rf)
getTree(inv.rf,5)

# Variable Importance Plot
varImpPlot(inv.rf,sort = T,main="Variable Importance",
           n.var=9)
#first try
# Variable Importance Table
var.imp1 <- data.frame(importance(inv.rf,type=1))
# make row names as columns
var.imp1$Variables <- row.names(var.imp1)
var.imp1[order(var.imp1$MeanDecreaseAccuracy,decreasing = T),]

#second try
# Variable Importance Table
var.imp2 <- data.frame(importance(inv.rf,type=2))
# make row names as columns
var.imp2$Variables <- row.names(var.imp2)
var.imp2[order(var.imp2$MeanDecreaseGini,decreasing = T),]

#prediction
# Predicting response variable
inv.dev$predicted.response <- predict(inv.rf ,inv.dev)
predicted.response <- predict(inv.rf ,inv.dev)
head(predicted.response)
# Create Confusion Matrix
confusionMatrix(data=inv.dev$predicted.response,
                reference=inv.dev$HRStatus,positive='yes')
confusionMatrix(data=inv.dev$predicted.response,
                reference=inv.dev$HRStatus)


# Predicting response variable
inv.val$predicted.response <- predict(inv.rf ,inv.val)

# Create Confusion Matrix
confusionMatrix(data=inv.val$predicted.response,
                reference=inv.val$HRStatus,
                positive="yes")
confusionMatrix(data=inv.val$predicted.response,
                reference=inv.val$HRStatus)
table(predict=inv.val$predicted.response, truth=inv.val$HRStatus)

##tune tree
tune=tuneRF(inv.dev[,-13],inv.dev[,13],
       stepFactor = 0.45,plot = TRUE,ntreeTry = 400,trace = TRUE,improve = 0.05)
inv.rf <- randomForest(rf.form,inv.dev,ntree=400,mtry=12,importance=T,proximity=TRUE)
print(inv.rf)

hist(treesize(inv.rf),col="green")
varUsed(inv.rf)
importance(inv.rf)
#partial dependence plot
partialPlot(inv.rf,inv.dev,)#variable name, "response variable specific value")
MDSplot(inv.rf,inv.dev$HRStatus)
