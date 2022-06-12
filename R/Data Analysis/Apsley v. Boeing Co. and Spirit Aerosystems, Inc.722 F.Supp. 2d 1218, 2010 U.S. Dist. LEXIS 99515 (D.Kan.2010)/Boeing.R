wants <- c('lawstat','BiasedUrn','Exact')
has <- wants %in% rownames(installed.packages())
if(any(!has)) install.packages(wants[!has])
library(Exact)

######Breslow and Day-test
breslowday.test <- function(x){
  or.hat.mh <- mantelhaen.test(x)$estimate
  K <- dim(x)[3]
  X2.HBD <- 0
  a <- tildea <- Var.a <- numeric(K)
  
  for (j in 1:K){
    mj <- apply(x[,,j],MARGIN = 1,sum)
    nj <- apply(x[,,j],MARGIN = 2,sum)
    
    coef <- c(-mj[1]*nj[1]*or.hat.mh,
              nj[2]-mj[1]+or.hat.mh*(nj[1]+mj[1]),
              1-or.hat.mh)
    sols <- Re(polyroot(coef))
    tildeaj <- sols[(0<sols) & (sols<=min(nj[1],mj[1]))]
    aj <- x[1,1,j]
    
    tildebj <- mj[1]-tildeaj
    tildecj <- nj[1]-tildeaj
    tildedj <- mj[2]-tildecj
    
    Var.aj <- (1/tildeaj+1/tildebj+1/tildecj+1/tildedj)^(-1)
    X2.HBD <- X2.HBD+as.numeric((aj-tildeaj)^2/Var.aj)
    
    a[j] <- aj
    tildea[j] <- tildeaj
    Var.a[j] <- Var.aj
  }
  
  X2.HBDT <- as.numeric(X2.HBD-(sum(a)-sum(tildea)^2)/sum(Var.aj))
  p <- 1-pchisq(X2.HBDT,df=K-1)
  
  res <- list(X2.HBD=X2.HBD,X2.HBDT=X2.HBDT,p=p)
  class(res) <- 'bdtest'
  return(res)
}

print.bdtest <- function(x){
  cat('Breslow and Day test (with Tarone correction):\n')
  cat('Breslow-Day X-squared =',x$X2.HBD,'\n')
  cat('Breslow-Day-Tarone X-squared =',x$X2.HBDT,'\n\n')
  cat('Test for a common OR: p-value = ',x$p,'\n\n')
}

###########CMH-power test###########
CMH.power <- function(data,odds,alpha,Alternative){
  s <- dim(data)[3]
  m <- n <- k <- rep(0,s)
  #m <- rep(0,s)
  #n <- rep(0,s)
  #k <- rep(0,s)
  for (i in 1:s){
    m[i] <- sum(data[1,,i])
    n[i] <- sum(data[2,,i])
    k[i] <- sum(data[,1,i])
  }
  library(BiasedUrn)
  library(lawstat)
  sim <- 10000
  pvalue.cmh <- rep(0,sim)
  pvalue.cmh.exact <- rep(0,sim)
  for (i in 1:sim){
    min <- rep(0,s)
    for (j in 1:s){
      min[j] <- rFNCHypergeo(1,m[j],n[j],k[j],odds)
    }
    maj <- k-min
    data <- NULL
    for (j in 1:s){
      data <- c(data,min[j],maj[j],m[j]-min[j],n[j]-maj[j])
    }
    sim.data <- array(data,dim=c(2,2,s))
    pvalue.cmh[i] <- mantelhaen.test(sim.data,alternative=Alternative)$p.value
    pvalue.cmh.exact[i] <- mantelhaen.test(sim.data,exact = T,alternative = Alternative)$p.value
  }
  power.cmh <- mean(pvalue.cmh<=alpha)
  power.cmh.exact <- mean(pvalue.cmh.exact<=alpha)
  return(list(power.cmh=power.cmh,power.cmh.exact=power.cmh.exact))
}

WD=read.csv("/Users/LoveChina/Documents/6253/ApsleyStrati.csv",
            header = TRUE)
head(WD)
summary(WD)
Wd=WD[,-1]
Wd=as.matrix(Wd)
wd=array(c(Wd[1,],Wd[2,],Wd[3,],Wd[4,],Wd[5,],Wd[6,],Wd[7,],Wd[8,],
           Wd[9,],Wd[10,],Wd[11,],Wd[12,],Wd[13,],Wd[14,],Wd[15,],Wd[16,],
           Wd[17,],Wd[18,],Wd[19,]), dim = c(2,2,19))
rownames(wd)=c("over40","under40")
colnames(wd)=c("fired","kept")
wd

#Doing Fisher's Exact Test for unit 1
A=matrix(Wd[1,],nrow=2,byrow = FALSE)
rownames(A)=c("over40","under40")
colnames(A)=c("fired","kept")
A
B=fisher.test(A, alternative = "greater")
B

#Doing Breslow-Day Test.  
breslowday.test(wd)
library(DescTools)
BreslowDayTest(wd)

#Doing Cochran–Mantel–Haenszel test.  
mantelhaen.test(wd)
CMH.power(wd,1.78,0.05,'two.sided')
CMH.power(wd,1.78,0.00015,'two.sided')
