#split the data into training and testing
newdata<-na.omit(final_dataset)
train<-sample(1:53,33)
trainingdata<-newdata[train,]
head(trainingdata)
library(MASS)
data(newdata)
dim(newdata)
head(newdata)
gender<-factor(trainingdata$Gender)
driverlicense<-factor(trainingdata$Driver.License)
Course<-factor(trainingdata$Courses.out.stat)
ugradmajor<-factor(trainingdata$ugrad.major)
ugradschool<-factor(trainingdata$ugrad.school)
lda.fit<-lda(ugradmajor~., data=final_dataset)
lda.fit
?lda
