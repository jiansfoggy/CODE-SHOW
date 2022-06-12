library(xlsx)
library(tseries)
library(fUnitRoots)
library(scales)
library(dplyr)
library(forecast)
###get wide form 
BJMA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/BJmonthAver.xlsx",header=TRUE,sheetName = 'W')
GZMA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/GZmonthAver.xlsx",header=TRUE,sheetName = 'W')
SYMA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/SYmonthAver.xlsx",header=TRUE,sheetName = 'W')
CDMA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/CDmonthAver.xlsx",header=TRUE,sheetName = 'W')
###get long form
BJLA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/BJonthAver.xlsx",header=TRUE,sheetName = 'L')
GZLA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/GZmonthAver.xlsx",header=TRUE,sheetName = 'L')
SYLA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/SYmonthAver.xlsx",header=TRUE,sheetName = 'L')
CDLA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/CDmonthAver.xlsx",header=TRUE,sheetName = 'L')
SHLA=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/SHdayandmonth.xlsx",header=TRUE,sheetName = 'L')

BJD=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/5deseasonal/5cities.xlsx",header=TRUE,sheetName = 'BJ')
GZD=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/5deseasonal/5cities.xlsx",header=TRUE,sheetName = 'GZ')
SYD=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/5deseasonal/5cities.xlsx",header=TRUE,sheetName = 'SY')
CDD=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/5deseasonal/5cities.xlsx",header=TRUE,sheetName = 'CD')
SHD=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/5deseasonal/5cities.xlsx",header=TRUE,sheetName = 'SH')



###monthly forecast for BJ
library(tseries)
library(fUnitRoots)
library(scales)
library(dplyr)
library(forecast)
BJTS=ts(BJD,frequency=12,start=c(2008,10), end=c(2016, 3))
mBJ=arima(BJTS[,1],order=c(1,2,0), seasonal=list(order=c(0,0,1), period=12))
plot(forecast(mBJ, 6), main="BeiJing Forecast")
plot(BJTS)
plot(decompose(BJTS))
acf(BJLA[,1],lag=12)
adfTest(BJLA[,1],lag=12,type='c')
mBJ=arima(BJTS,order=c(1,0,0), seasonal=list(order=c(0,0,3), period=12))
fit=predict(mBJ, BJLA[,3],end=c(2016, 12))
plot(fit$pred)
forecast(mBJ, BJLA[,3],end=c(2016, 12))
summary(mBJ)
tsdiag(mBJ)
rBJ <- mBJ$residuals
acf(rBJ)
Box.test(rBJ, lag = 12, type = 'L')
Box.test(rBJ ^ 2, lag = 12, type = 'L')
forecast(mBJ, 6)
par(mfrow=c(1,1))
plot(forecast(mBJ, 6), main="BeiJing Forecast")
plot(forecast(mBJ, 6), main="BeiJing Forecast")


###monthly forecast for GZ
GZD
GZTS=ts(GZD,frequency=12,start=c(2012,11), end=c(2016, 3))
mGZ=arima(GZTS[,1],order=c(1,2,0), seasonal=list(order=c(0,0,6), period=12))
GZTS=ts(GZLA[,1],frequency=12,start=c(2011,11), end=c(2016, 9))
plot(GZTS)
plot(decompose(GZTS))
acf(GZLA[,1],lag=12)
adfTest(GZLA[,1],lag=12,type='c')
mGZ=arima(GZTS,order=c(1,0,0), seasonal=list(order=c(0,0,3), period=12))
summary(mGZ)
tsdiag(mGZ)
rGZ <- mGZ$residuals
acf(rGZ)
Box.test(rGZ, lag = 12, type = 'L')
Box.test(rGZ ^ 2, lag = 12, type = 'L')
forecast(mGZ, 6)
plot(forecast(mGZ, 6), main="GuangZhou Forecast")


###monthly forecast for SY
seasonal=list(order=c(0,0,1), period=12)
SYLA=SYLA[,-1:-2]
SHD
SHTS=ts(SHD[,1],frequency=12,start=c(2012,6), end=c(2016, 3))
mSH=arima(SHTS,order=c(1,2,0),seasonal=list(order=c(0,0,1), period=12))
plot(forecast(mSH, 6), main="ShangHai Forecast")
plot(SYTS)
plot(decompose(SYTS))
acf(SYLA[,1],lag=12)
adfTest(SYLA[,1],lag=12,type='c')
mSY=arima(SYTS,order=c(1,0,0), seasonal=list(order=c(0,0,3), period=12))
summary(mSY)
tsdiag(mSY)
rSY <- mSY$residuals
acf(rSY)
Box.test(rSY, lag = 12, type = 'L')
Box.test(rSY ^ 2, lag = 12, type = 'L')
forecast(mSY, 6)
plot(forecast(mSY, 6))


###monthly forecast for SH
SYD
SYTS=ts(SYD,frequency=12,start=c(2013,10), end=c(2016, 3))
mSY=arima(SYTS[,1],order=c(1,2,0),seasonal=list(order=c(0,0,1), period=12))
plot(forecast(mSY, 6), main="ShenYang Forecast")



###monthly forecast for CD
CDLA=CDLA[,-1:-2]
CDD
CDTS=ts(CDD,frequency=12,start=c(2012,11), end=c(2016, 3))
mCD=arima(CDTS[,1],order=c(1,2,0), seasonal=list(order=c(0,0,3), period=12))
plot(forecast(mCD, 6), main="ChengDu Forecast")
plot(CDTS)
plot(decompose(CDTS))
acf(CDLA[,1],lag=12)
adfTest(CDLA[,1],lag=12,type='c')
mCD=arima(CDTS,order=c(1,0,0), seasonal=list(order=c(0,0,3), period=12))
summary(mCD)
tsdiag(mCD)
rCD <- mCD$residuals
acf(rCD)
Box.test(rCD, lag = 12, type = 'L')
Box.test(rCD ^ 2, lag = 12, type = 'L')
forecast(mCD, 6)
plot(forecast(mCD, 6), main="Cheng")


###monthly forecast for BJ
X=BJLA$MonthlyAver
ku=1:length(X)
plot(BJLA$Date,X,type="l")
reg=lm(X~BJLA$Date)
abline(reg,col="red")
Y=residuals(reg)
acf(Y,lag=36,lwd=2)
Z=diff(Y,12)
acf(Z,lag=36,lwd=2)
pacf(Z,lag=36,lwd=2)
model2b=arima(Y,order=c(1,0,0),
              seasonal = list(order = c(0, 1, 0),
              period=12)) 
summary(model2b)
model3c=arima(Y,order=c(1,0,0),
           seasonal = list(order = c(0, 0, 3), 
           period = 12))
model3c
forecast(model2b,3)
previ=function(X,h,b){
  ku=c(1:length(X))
  reg=lm(X~BJLA$Date)
  Y=residuals(reg)
  model=arima(Y,order=c(1,0,0),
                seasonal = list(order = c(0, 1, 0),
                                period=12))
  prev=forecast(model,h)
  Tfutur=length(X)+1:(length(X)+h)
  plot(ku,Y,type="l", xlim=c(0,length(X)+h),ylim=c(-80,b))
  polygon(c(Tfutur,rev(Tfutur)),c(prev$lower[,2],rev(prev$upper[,2])),col="orange",border=NA)
  polygon(c(Tfutur,rev(Tfutur)),c(prev$lower[,1],rev(prev$upper[,1])),col="yellow",border=NA)
  lines(prev$mean,col="blue")
  lines(Tfutur,prev$lower[,2],col="red")
  lines(Tfutur,prev$upper[,2],col="red")
}

previ(BJLA$MonthlyAver,4,b=120)
###get ccf
BJTS=ts(BJLA[,1],frequency=12,start=c(2008,4), end=c(2016, 9))
SYTS=ts(SYLA[,1],frequency=12,start=c(2013,4), end=c(2016, 9))
GZTS=ts(GZLA[,1],frequency=12,start=c(2011,11), end=c(2016, 9))
CDTS=ts(CDLA[,1],frequency=12,start=c(2012,5), end=c(2016, 9))
SHTS=ts(SHLA[,1],frequency=12,start=c(2011,12), end=c(2016, 9))

BJT=window(BJTS,start=c(2013,5), end=c(2016,9))
GZT=window(GZTS,start=c(2013,5), end=c(2016,9))
SYT=window(SYTS,start=c(2013,5), end=c(2016,9))
CDT=window(CDTS,start=c(2013,5), end=c(2016,9))
SHT=window(SHTS,start=c(2013,5), end=c(2016,9))

par(mfrow=c(1,1))
BJtoGZ=ccf(BJT, GZT, type = "correlation")
BJtoGZ
cvalue1$lag
summary(cvalue1)

SYtoBJ=ccf(SYT, BJT,type = "correlation")
SYtoBJ

GZtoCD=ccf(GZT, CDT,type = "correlation")
GZtoCD

BJtoSH=ccf(BJT, SHT,type = "correlation")
BJtoSH

SHtoGZ=ccf(SHT, GZT,type = "correlation")
SHtoGZ

BJDT=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/BJDaymean.xlsx",header=TRUE,sheetName = 'BJWinter')
SHDT=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/SHdayandmonth.xlsx",header=TRUE,sheetName = 'SHWinter')

BJTS=ts(BJDT[,3])
SHTS=ts(SHDT[,3])
BJTS2=ts(BJDT[1:235,3])
SHTS2=ts(SHDT[1:235,3])
BJTS3=ts(BJDT[415:535,3])
SHTS3=ts(SHDT[422:542,3])

BJtoSH1=ccf(BJTS, SHTS, type = "correlation")
BJtoSH1
BJtoSH2=ccf(BJTS2, SHTS2, type = "correlation", main="From Jan,2012 to Dec, 2013")
BJtoSH2
BJtoSH3=ccf(BJTS3, SHTS3, type = "correlation", main="From Noc,2015 to Feb, 2016")
BJtoSH3

BJDF=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/BJDaymean.xlsx",header=TRUE,sheetName = 'BJMW')
SHDF=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/cleaned data/SHdayandmonth.xlsx",header=TRUE,sheetName = 'SHMW')

BJS=ts(BJDT[,3],frequency=1,start=c(2012,1), end=c(2016,2))
SHS=ts(SHDT[,3],frequency=1,start=c(2012,1), end=c(2016,2))

BJtoSH1=ccf(BJS, SHS, type = "correlation")
BJtoSH1


###plot relative plot
C5=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/ccf.xlsx",header=TRUE,sheetName = 'CITY5')
C5Winter=read.xlsx("/Users/LoveChina/Documents/6289/Project 2/ccf.xlsx",header=TRUE,sheetName = 'ForWinter')

library(corrgram)
colorset=function(ncol){
  colorRampPalette(c("darkgoldenrod4","burlywood1","darkkhaki","darkgreen","red"))(ncol)
} 
corrgram(C5[,4:8], order=TRUE, lower.panel=panel.shade, upper.panel=panel.pie,
         diag.panel=panel.minmax,
         main="A Correlation Among 5 Cities")

corrgram(C5Winter[,4:8], order=TRUE, lower.panel=panel.shade, upper.panel=panel.pie,
         diag.panel=panel.minmax,
         main="A Correlation Among 5 Cities in Winter")

###density plot
library(ggplot2)
p <- ggplot(C5Winter, aes(x=C5Winter$BJ, y=C5Winter$SH))
p + stat_density2d(aes(fill=..density..), geom="raster", contour=FALSE)+
  ggtitle("The Winter Correlation Between BeiJing and ShangHai")
p + geom_point() +
  stat_density2d(aes(alpha=..density..), geom="tile", contour=FALSE)+
  ggtitle("The Winter Correlation Between BeiJing and ShangHai")