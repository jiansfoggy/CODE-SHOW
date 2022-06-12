x11(10,7)
set.seed(1)
X=runif(1000, -4, 4)
eps=rnorm(1000, 0, 0.01)
Y=eps+(exp(7*X)/(1+exp(7*X)))+eps
compliance <- X
improvement <- Y
bw <- 10
plot(compliance, improvement)
lines(ksmooth(compliance, improvement, bandwidth = bw, kernel = "normal"),col="blue")

out = locfit(Y~X,deg=0,alpha=c(0,h))

h = 5
k = length(h)
zero = rep(0,k)
H = cbind(zero,h)
out = gcvplot(Y~X,deg=0,alpha=H)
plot(out$df,out$values)
optband=max(out$alpha[out$values == min(out$values),2])
locfitopt= locfit(Y~X,alpha=c(0,optband),deg=1.349,maxk=1000)


m=function(X) {
  num=-4*cos(2*pi*x)
  den=x+.5
  return(num/den)
  #return(num)
}
locfit_simdata= function(f,sigma,n,h,deg,xlo,xhi)
{  # GENERATE THE DATA
  x = runif(n,xlo,xhi)
  y = m(x)  + rnorm(n,sd=sigma)
  # FIT THE MODEL USING LOCFIT
  smoo.fit= locfit(y~x,alpha=c(0,h),deg=deg,maxk=1000)
  # USE GCV TO CHOOSE BANDWIDTH
  H=seq(0.05, 5, by = 0.05)
  alphamat= matrix(0,ncol=2,nrow=length(H))
  alphamat[,2]=H
  gcvs=gcvplot(y~x,alpha=alphamat,deg=deg,maxk=1000)
  optband= max(gcvs$alpha[gcvs$values == min(gcvs$values),2])
  locfitopt=locfit(y~x,alpha=c(0,optband),deg=deg,maxk=1000)
  # MAKE PLOTS
  xg = seq(xlo,xhi,length=1000)
  plot(x,y,xlab="x",ylab="y",pch=16,cex=0.5,col=rgb(0.7,0.7,0.7))
  lines(xg,m(xg),col=2,lwd=3)
  lines(xg,predict(smoo.fit,newdata=xg),col=3,lwd=3)
  lines(xg,predict(locfitopt,newdata=xg),col=4,lwd=3)
  legend(xhi,min(y),legend=c(paste("m(x)=-4*cos(2*pi*x)/(x+.5) "),
                             paste("estimate using h =",h),
                             paste("estimate using gcv optimal h = ",round(optband,3))),col=c(2,3,4),
         lwd=2,bty="n",cex=1.2,yjust=0,xjust=1)
  mtext(paste("degree=",deg,", ", "n=",n,",
              sigma=",sigma,sep=""),line=1,adj=0,cex=1.3)
}

for (i in 1:10) {
  fit=glm(Y~X,family=gaussian, data=BDF$   , subset=TN[i])
  MSE[i]=mean((Y-predict(fit,BDF[i]))[TT[i]]^2)
}

############################
x11(10,7)

set.seed(1)
X=runif(1000, -4, 4)
eps=rnorm(1000, 0, 0.01)
Y=eps+(exp(7*X)/(1+exp(7*X)))
DF=data.frame(Y,X)
DFF=data.frame(Y,X)
set.seed(17)
cver=rep(0,10)
DF=data.frame(Y,X)
for (i in 1:10){
  glmf=glm(Y~poly(X ,i),data=DF)
  cver[i]=cv.glm(DF,glmf,K=10)$delta[1]
}
cver
summary(glmf)

set.seed(1)
S1=sample(1:nrow(DF),100,replace=FALSE)
S1=sort(S1)
S1
DF1<- DF[S1,]

for (i in 1:100) {
  DF=DF[-S1[101-i],]
}
nrow(DF)

set.seed(1)
S2=sample(1:nrow(DF),100,replace=FALSE)
S2=sort(S2)
S2
DF2<- DF[S2,]

for (i in 1:100) {
  DF=DF[-S2[101-i],]
}
nrow(DF)

set.seed(1)
S3=sample(1:nrow(DF),100,replace=FALSE)
S3=sort(S3)
S3
DF3<- DF[S3,]

for (i in 1:100) {
  DF=DF[-S3[101-i],]
}
nrow(DF)

set.seed(1)
S4=sample(1:nrow(DF),100,replace=FALSE)
S4=sort(S4)
S4
DF4<- DF[S4,]

for (i in 1:100) {
  DF=DF[-S4[101-i],]
}
nrow(DF)

set.seed(1)
S5=sample(1:nrow(DF),100,replace=FALSE)
S5=sort(S5)
S5
DF5<- DF[S5,]

for (i in 1:100) {
  DF=DF[-S5[101-i],]
}
nrow(DF)

set.seed(1)
S6=sample(1:nrow(DF),100,replace=FALSE)
S6=sort(S6)
S6
DF6<- DF[S6,]

for (i in 1:100) {
  DF=DF[-S6[101-i],]
}
nrow(DF)

set.seed(1)
S7=sample(1:nrow(DF),100,replace=FALSE)
S7=sort(S7)
S7
DF7<- DF[S7,]
DF
for (i in 1:100) {
  DF=DF[-S7[101-i],]
}
nrow(DF)

set.seed(1)
S8=sample(1:nrow(DF),100,replace=FALSE)
S8=sort(S8)
S8
DF8<- DF[S8,]

for (i in 1:100) {
  DF=DF[-S8[101-i],]
}
nrow(DF)

set.seed(1)
S9=sample(1:nrow(DF),100,replace=FALSE)
S9=sort(S9)
S9
DF9<- DF[S9,]

for (i in 1:100) {
  DF=DF[-S9[101-i],]
}
nrow(DF)

set.seed(1)
S10=sample(1:nrow(DF),100,replace=FALSE)
S10=sort(S10)
S10
DF10<- DF[S10,]

for (i in 1:100) {
  DF=DF[-S10[101-i],]
}
nrow(DF)

tn1=sample(1:nrow(DF1), 4*nrow(DF1)/5)
tt1=(-tn1)
tn2=sample(1:nrow(DF2), 4*nrow(DF2)/5)
tt2=(-tn2)
tn3=sample(1:nrow(DF3), 4*nrow(DF3)/5)
tt3=(-tn3)
tn4=sample(1:nrow(DF4), 4*nrow(DF4)/5)
tt4=(-tn4)
tn5=sample(1:nrow(DF5), 4*nrow(DF5)/5)
tt5=(-tn5)
tn6=sample(1:nrow(DF6), 4*nrow(DF6)/5)
tt6=(-tn6)
tn7=sample(1:nrow(DF7), 4*nrow(DF7)/5)
tt7=(-tn7)
tn8=sample(1:nrow(DF8), 4*nrow(DF8)/5)
tt8=(-tn8)
tn9=sample(1:nrow(DF9), 4*nrow(DF9)/5)
tt9=(-tn9)
tn10=sample(1:nrow(DF10), 4*nrow(DF10)/5)
tt10=(-tn10)

BDF=data.frame(DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9, DF10)
TN=c(tn1,tn2,tn3,tn4,tn5,tn6,tn7,tn8,tn9,tn10)
TT=c(tt1,tt2,tt3,tt4, tt5,tt6, tt7,tt8,tt9,tt10)
MSE=rep(0,10)

fit1=glm(Y~X,family=gaussian,data=DF1,subset=tn1) 
MSE1=mean((Y-predict(fit1,DF1))[-tn1]^2)
fit2=glm(Y~X,family=gaussian,data=DF2,subset=tn2) 
MSE2=mean((Y-predict(fit2,DF2))[-tn2]^2)
fit3=glm(Y~X,family=gaussian,data=DF3,subset=tn3) 
MSE3=mean((Y-predict(fit3,DF3))[-tn3]^2)
fit4=glm(Y~X,family=gaussian,data=DF4,subset=tn4) 
MSE4=mean((Y-predict(fit4,DF4))[-tn4]^2)
fit5=glm(Y~X,family=gaussian,data=DF5,subset=tn5) 
MSE5=mean((Y-predict(fit5,DF5))[-tn5]^2)
fit6=glm(Y~X,family=gaussian,data=DF6,subset=tn6) 
MSE6=mean((Y-predict(fit6,DF6))[-tn6]^2)
fit7=glm(Y~X,family=gaussian,data=DF7,subset=tn7) 
MSE7=mean((Y-predict(fit7,DF7))[-tn7]^2)
fit8=glm(Y~X,family=gaussian,data=DF8,subset=tn8) 
MSE8=mean((Y-predict(fit8,DF8))[-tn8]^2)
fit9=glm(Y~X,family=gaussian,data=DF9,subset=tn9) 
MSE9=mean((Y-predict(fit9,DF9))[-tn9]^2)
fit10=glm(Y~X,family=gaussian,data=DF10,subset=tn10) 
MSE10=mean((Y-predict(fit10,DF10))[-tn10]^2)
MSE=c(MSE1,MSE2,MSE3,MSE4,MSE5,MSE6,MSE7,MSE8,MSE9,MSE10)
MSE
mean(MSE)
##########################

A=c(1:1000)
A=c(1:700)
AA=sample(A,70)
AA=sort(AA)
for (i in 1:70) {
  A=A[-AA[i]]
}


set.seed(1)
train=sample(350,1:700,replace = FALSE)
lmf=lm(Y~X,data=DF,subset=train)
mean((Y-predict(lmf,DF))[-train]^2)

#get actual best bandwidth
library(kedd)
KNF1=kernel.fun(x = runif(1000,-4,4), deriv.order = 2, kernel = "gaussian")
KNF1
KNF2=kernel.fun(x = seq(0,100,by=0.01), deriv.order = 1, kernel = "gaussian")
KNF3=kernel.fun(x = seq(0,100,by=0.01), deriv.order = 2, kernel = "gaussian")

KNF$kx
hatf1 <- dkde(X, deriv.order = 0,kernel = "gaussian")
hatf2 <- dkde(X, deriv.order = 1,kernel = "gaussian")
hatf3 <- dkde(X, deriv.order = 2,kernel = "gaussian")
hatf1
hatf2
hatf3

h.amise(X, deriv.order = 0)
h.amise(X, deriv.order = 1)
h.amise(X, deriv.order = 2)
h.amise(X, deriv.order = 3)
h.amise(X, deriv.order = 4)
h.amise(X, deriv.order = 5)
h.amise(X, deriv.order = 6)
h.amise(X, deriv.order = 7)
h.amise(X, deriv.order = 8)
h.amise(X, deriv.order = 9)
h.amise(X, deriv.order = 10)
#Use direct plug-in methodology 
#to select the bandwidth of a kernel density estimate.
library(KernSmooth)
dpik(X,  kernel = "normal",   
     canonical = FALSE, gridsize = 401L, range.x = range(X), 
     truncate = TRUE)

#K-fold cross validation
mydata <- data.frame(ymat, xmat)
fit <- glm(Y~X, data=DF)
library(DAAG)
cv.lm(df=DF, fit, m=10)

names(bimodal)
bimodal

set.seed(1)

train=sample(1:nrow(DF), 4*nrow(DF)/5)
test=(-train)

#书上的做法
#Leave one out cv
library(boot)
cver1=rep(0,5)
for (i in 1:5){
  YXLOO=glm(Y~poly(X ,i), family=gaussian, data=DF)
  cver1[i]=cv.glm(DF,YXLOO)$delta[1] 
  }
cver1


#k-fold cv
set.seed(17)
cver=rep(0,n)
for (i in 1:n){
  YXKF=glm(logit(y/1-y)~x,family=gaussian, data=G)
  se[i]=cv.glm(DF,YXKF,K=n)$delta[1]
  }
cver

# LOOCV load the library
library(caret)
# load the iris dataset
data(iris)
# define training control
train_control <- trainControl(method="cv", number=10)
# fix the parameters of the algorithm
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
# train the model
model <- train(Species~., data=iris, trControl=train_control, method="nb", tuneGrid=grid)
# summarize results
print(model)
￼￼￼￼￼￼￼￼￼


function(x, y, h, n){
  
  smof = locfit(y~x,alpha=c(0,h),deg=n-1, maxk=1000)
  H=seq(h,10, by = 0.05)
  alphamat= matrix(0,ncol=2,nrow=length(H))
  alphamat[,2]=H
  gcvs=gcvplot(y~x,alpha=alphamat,deg=n-1,maxk=1000)
  baw=gcvs$alpha[gcvs$values == min(gcvs$values),2]
  baw
  optband= max(gcvs$alpha[gcvs$values == min(gcvs$values),2])
  locfitopt=locfit(y~x,alpha=c(0,optband),deg=n-1,maxk=1000)
  
  HS=length(x)
  MSE=numeric(HS)
  G=data.frame(x,y)
  #method 1
  for (i in 1:HS) {
    fitmod_i=glm(logit(y/1-y)~x, family = gaussian, data = G)
    predv=predict(fitmod,G)
    MSE[i]=(y-predv)[i,]^2
  }
  #mehtod 2
  set.seed(17)
  se=rep(0,n)
  for (i in 1:n){
    YXKF=glm(logit(y/1-y)~x,family=gaussian, data=G)
    se[i]=cv.glm(DF,YXKF,K=n)$delta[1]
  }
  se
}
