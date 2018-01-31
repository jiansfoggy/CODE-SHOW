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


###Classification
tree = tree(train$HRStatus~., data=train)
summary(tree)

#Fitting and Pruning
plot(tree, type = "uniform")
text(tree, all=T)
opt = cv.tree(tree, FUN=prune.misclass)
opt
optimal = opt$size[opt$dev == min(opt$dev)]
par(mfrow = c(1,2))
plot(opt$size, opt$dev, pch=16, cex=0.9, 
     col="red", type="b")
plot(opt$k, opt$dev, pch=16, cex=0.9, col="red", type="b")

###pruning
prune = prune.misclass(tree, best=optimal)
summary(prune)
pred = predict(prune, test, type = "class")
summary(pred)
#as.numeric(pred)-as.numeric(test$HRStatus)

plot(prune, type="uniform")
text(prune, all=T)

# Classification Table
table=table(pred, test$HRStatus)
table

# Accuracy Rate
ar = sum(diag(prop.table(table)))
ar

# MSE
predict = predict(prune, test)
mse_class <- mean((predict - as.numeric(test$HRStatus))^2)
mse_class

#Plot pruned tree
par(mfrow = c(1,1))
plot(prune)
text(prune, pretty = 0)

### Support Vector Classification
library(e1071)
Response1=train$HRStatus
#Predictor1=as.matrix(train[,1:9])

Response2=test$HRStatus
#Predictor2=as.matrix(test[,1:9])

TrainD=data.frame(Response1, train[,1:10])
TestD=data.frame(Response2, test[,1:10])


tune.out1=tune(svm,Response1~., data=TrainD, kernel="linear",
               ranges=list(cost=c(0.001,0.01,0.1,1,5,10,50,100)),
               decision.values=T)
summary(tune.out1)

tune.out1=tune(svm,Response1~., data=TrainD, kernel="linear",
               ranges=list(cost=c(0.001,0.002,0.02,0.04,0.5,0.7,0.9,1,1.2,4,6)),
               decision.values=T)
summary(tune.out1)

tune.out1=tune(svm,Response1~., data=TrainD, kernel="linear",
               ranges=list(cost=c(0.56,0.7,0.9,0.95,1,1.1,1.15,1.2,1.3,1.5,1.53,1.55,1.82)),decision.values=T)
summary(tune.out1)

##choose the best cost
bestmod=tune.out1$best.model
summary(bestmod)

##turn to do predict for testing dataset
ypred=predict(bestmod, TestD, decision.values=TRUE)
SVCT=table(predict=ypred, truth=TestD$Response2)
SVCT
sum(diag(prop.table(SVCT)))

###Naive Bayes Classifier
library(e1071)
NB=naiveBayes(train$HRStatus~., data=train)
summary(NB)
NB$call
collect=predict(NB,test,type="raw")
pred.NaiveBayes=predict(NB,newdata=test,laplace=7)
pred.NaiveBayes
table(pred.NaiveBayes,train)
summary(pred.NaiveBayes)


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


library(party)

fit <- cforest(HRStatus ~ ., data = inv.dev, 
               controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, inv.val, OOB=TRUE, type = "response")
plot(fit,type="simple")
k=table(Prediction, truth=inv.val$HRStatus)
ark=sum(diag(prop.table(k)))
ark

library(dplyr)
library(ggraph)
library(igraph)

tree_func <- function(final_model, 
                      tree_num) {
  
  # get tree by index
  tree <- randomForest::getTree(final_model, 
                                k = tree_num, 
                                labelVar = TRUE) %>%
    tibble::rownames_to_column() %>%
    # make leaf split points to NA, so the 0s won't get plotted
    mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))
  
  # prepare data frame for graph
  graph_frame <- data.frame(from = rep(tree$rowname, 2),
                            to = c(tree$`left daughter`, tree$`right daughter`))
  
  # convert to graph and delete the last node that we don't want to plot
  graph <- graph_from_data_frame(graph_frame) %>%
    delete_vertices("0")
  
  # set node labels
  V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
  V(graph)$leaf_label <- as.character(tree$prediction)
  V(graph)$split <- as.character(round(tree$`split point`, digits = 2))
  
  # plot
  plot <- ggraph(graph, 'dendrogram') + 
    theme_bw() +
    geom_edge_link() +
    geom_node_point() +
    geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
    geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
    geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE, 
                    repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.background = element_blank(),
          plot.background = element_rect(fill = "white"),
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(size = 18))
  
  print(plot)
}



###Model Selection
library(MASS)
library(dplyr)
library(car)
library(ElemStatLearn)
library(ROCR)


##neural network
varNames <- names(inv.dev)
# Exclude ID or Response variable
varNames <- varNames[!varNames %in% c("HRStatus")]
# add + sign between exploratory variables
varNames1 <- paste(varNames, collapse = "+")
# Add response variable and convert to a formula object
rf.form <- as.formula(paste("HRStatus", varNames1, sep = " ~ "))
library(nnet)
fitnn = nnet(rf.form,inv.dev, size=1)
fitnn
summary(fitnn)
predicted=predict(fitnn, inv.val)
colnames(predicted)=c("num","per")
table(data.frame( predicted[,1] > 0.5,
                 actual=inv.val[,13]>0.5))


#https://stackoverflow.com/questions/37017165/r-plot-trees-from-h2o-randomforest-and-h2o-gbm
# # Next, we download packages that H2O depends on.
# pkgs <- c("methods","statmod","stats","graphics","RCurl","jsonlite","tools","utils")
# for (pkg in pkgs) {
#   if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
# }
# 
# # Now we download, install h2o package
# install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-turchin/3/R")))
library(h2o)

h2o.init(nthreads = -1, max_mem_size = "2G")
h2o.removeAll()  ##clean slate - just in case the cluster was already running

## Load data - available to download from link below
## https://www.dropbox.com/s/gu8e2o0mzlozbu4/SampleData.csv?dl=0
df <- h2o.importFile(path = normalizePath("../SampleData.csv"))

splits <- h2o.splitFrame(df, c(0.4, 0.3), seed = 1234)

train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test <- h2o.assign(splits[[2]], "test.hex")

predictor_col_start_pos <- 2
predictor_col_end_pos <- 169
predicted_col_pos <- 1

rf1 <- h2o.randomForest(training_frame = train, validation_frame = valid, 
                        x = predictor_col_start_pos:predictor_col_end_pos, y = predicted_col_pos, 
                        model_id = "rf_covType_v1", ntrees = 2000, stopping_rounds = 10, score_each_iteration = T, 
                        seed = 2001)

gbm1 <- h2o.gbm(training_frame = train, validation_frame = valid, x = predictor_col_start_pos:predictor_col_end_pos, 
                y = predicted_col_pos, model_id = "gbm_covType2", seed = 2002, ntrees = 20, 
                learn_rate = 0.2, max_depth = 10, stopping_rounds = 2, stopping_tolerance = 0.01, 
                score_each_iteration = T)


## Next step would be to plot trees for fitted models rf1 and gbm2
# print the model, POJO (Plain Old Java Object) to screen
h2o.download_pojo(rf1)
h2o.download_pojo(gbm1)