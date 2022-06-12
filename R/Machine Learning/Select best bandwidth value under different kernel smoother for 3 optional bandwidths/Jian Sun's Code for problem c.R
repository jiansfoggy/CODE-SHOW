LOOCV=function(x, y, h, n){
  ##get n folds
  library(plyr)
  CVgroup <- function(k, datasize) {
    cvlist <- list()
    m <- rep(1:k, ceiling(datasize/k))[1:datasize] 
    temp <- sample(m, datasize)  
    d <- 1:k
    dataseq <- 1:datasize 
    cvlist <- sapply(d, function(d) dataseq[temp==d]) 
    return(cvlist)
  }
  ##get MSE matrix
  DF=data.frame(x, y)
  DIV=CVgroup(n,nrow(DF))
  MSE=matrix(0,nrow=n,ncol=length(h), byrow = TRUE)
  ## get train and test
  for (i in 1:n) {
    for (j in 1:length(h)) {
      test=DIV[,i]
      train=DIV[,-i]
      testX=X[test]
      testY=Y[test]
      trainX=X[train]
      trainY=Y[train]
      TestX=matrix(testX,nrow=length(train),ncol = length(test), byrow=TRUE)
      TrainX=matrix(trainX, nrow = length(train), ncol = length(test), byrow=FALSE)
      ## get K(u) and m_hat(x)
      XC=TrainX-TestX
      U=XC/h[j]
      Ku=0.75*(1-U^2)*ifelse(abs(U)<=1,1,0)
      SUMKU=apply(Ku,2,sum)
      KUXY=trainY%*%Ku
      MX=KUXY/SUMKU
      ## get MSE
      MSE[i,j]=(sum((testY-MX)^2))/length(test)
    }
  }
  ## get MSE array for different Bandwidth
  MMSE=apply(MSE, 2, mean)
  ## get min MSE point
  MINP=which.min(MMSE)
  
  ##This is the first part return, I put all the return in the list function.
  A=("The actual best bandwidth is:")
  B=h[MINP]
  
  
  ##This is the second part return.
  C=("The Cross-Validation mean-squared errors of all the different bandwidth is:")
  D=MMSE
  
  ##This is the third part return.
  E=("The array of MSE for each bandwidth on each fold is:")
  FF=("Column represents each bandwidth, Row represents each fold")
  G=MSE
  
  ##The following are what I als want to get.
  plot(MMSE~h,xlab = "Bandwidth", ylab = "MSE",type="b")
  points(h[MINP],MMSE[MINP], col="red", cex=2, pch=20)
  HH="So the least MSE is:"
  I=min(MMSE)
  SUMRE=list(A,B,E,FF,G,C,D,HH,I)
  return(SUMRE)
}

library(ISLR)
attach(Wage)
names(Wage)
age=Wage$age
wage=Wage$wage
h=seq(0.01,2,0.02)
LOOCV(age, wage, h, 5)
