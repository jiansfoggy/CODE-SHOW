#PD=Playerdata, SP=shorter player, BP= bigger player
library(MASS)
PD=read.table("/Users/LoveChina/Desktop/DATA.txt", header = TRUE)
PD1=read.table("/Users/LoveChina/Desktop/partdat1.txt",header = TRUE)
PD2=read.table("/Users/LoveChina/Desktop/partata1.txt",header = TRUE)
FM<-lm(PD$OA~.,data = PD)
summary(FM)
Y=PD$OA
X1=PD$MSM
X2=PD$MST
X3=PD$DL
X4=PD$BC
X5=PD$PA
X6=PD$S
X7=PD$S.1
X8=PD$V
X9=PD$RT
SP<-data.frame(Y,Z1,X1,X2)
plot(SP)
Z1=PD$S.1
Z2=PD$MSC
Z3=PD$MSM
Z4=PD$DF
Z5=(PD$OR^2+PD$DR^2)/(PD$OR+PD$DR)
Z6=PD$S
Z7=PD$B
Z8=PD$V
Z9=PD$S.2
BP<-data.frame(Y,Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9)
plot(BP)
RM1<-lm(SP$Y~.,data = SP)
list(RM1)
summary(RM1)
anova(FM1)
plot(stdres(RM1))
FM1<-lm(BP$Y~.,data=BP)
summary(FM1)
plot(stdres(FM1))
(583.369-134.867)/9/(134.867/20)
qf(0.05,9,20,lower.tail=FALSE)
qt(p=0.025,df=20,lower.tail=FALSE)
colinear1<-lm(Z8~Z2+Z3+Z4+Z5+Z6+Z7+Z1+Z9)
summary(colinear1)
1/(1- 0.8128)
colinear2<-lm(Z7~Z2+Z3+Z4+Z5+Z6+Z8+Z1+Z9)
summary(colinear2)
1/(1-0.7162)
colinear3<-lm(Z9~Z2+Z3+Z4+Z5+Z6+Z7+Z1+Z8)
summary(colinear3)
1/(1-0.5627)

BPnew<-data.frame(Y,Z2,Z3,Z4,Z5,Z7,Z8)
plot(BPnew)
RMnew<-lm(TP1$Y~.,data=TPnew)
summary(RMnew)
anova(FM1)
anova(RMnew)
((206.23-180.604)/3)/(180.604/20)
qf(0.05,3,20,lower.tail = FALSE)
BP1<-data.frame(Y,Z1,Z2,Z3,Z4,Z5,Z6)
plot(BP1)
RM1<-lm(BP1$Y~.,data=BP1)
summary(RM1)
anova(RM1)
((145.324-134.867)/3)/(134.867/20)
qf(0.05,3,20,lower.tail = FALSE)
library(MASS)
plot(stdres(RM1))
BP2<-data.frame(Y,Z1,Z2,Z4,Z5,Z6)
plot(BP2)
RM2<-lm(BP2$Y~.,data=BP2)
summary(RM2)
anova(RM2)
plot(stdres(RM2))
Y0hat=29.933+0.095*74+0.226*90+0.13*87+0.176*70.91+0.175*61
l=(62^2+78^2)/(62+78)
Yhat=24.95+0.102*74+0.22*90+0.05*87+0.137*87+0.198*70.91+0.162*61
PDM=cbind(Z1,Z2,Z4,Z5,Z6)
BG=c(74,90,87,70.91,61)
sey0hat<-sqrt(6.326*(1+t(BG)%*%(t(PDM)%*%PDM)^-1%*%BG))
qt(0.025,24,lower.tail = FALSE)
91.768-3.44*2.0639
91.768+3.44*2.0639
PCTL<-cbind(PD1)
PCFM<-prcomp(PD1,scale=TRUE,scores=TRUE)
summary(PCFM)
CRTL<-cor(PD1)
eigen(CRTL)
lj=c(79,75,75,80,74,75,71,96,84,81,81,36,67,71,56,86,88,89,90,95)
LJ=matrix(lj,nrow = 20)
td=c(92,75,35,90,73,25,73,63,73,40,56,67,88,61,85,37,42,60,88,74)
TD=matrix(td,nrow = 20)
mj=c(98,97,75,99,96,74,85,96,85,88,85,47,62,91,56,88,98,57,98,98)
MJ=matrix(mj,nrow=20)
lb=c(97,95,94,94,93,90,89,72,76,79,94,61,77,75,47,64,50,65,97,91)
LB=matrix(lb,nrow=20)
K=PD1
M1=mean(PD1$SSC);
M2=mean(PD1$SSM);
M3=mean(PD1$SST);
M4=mean(PD1$MSC);
M5=mean(PD1$MSM);
M6=mean(PD1$MST);
M7=mean(PD1$FT);
M8=mean(PD1$DL);
M9=mean(PD1$DF);
M10=mean(PD1$BC);
M11=mean(PD1$PA);
M12=mean(PD1$OR);
M13=mean(PD1$DR);
M14=mean(PD1$S);
M15=mean(PD1$B);
M16=mean(PD1$Sp);
M17=mean(PD1$V);
M18=mean(PD1$St);
M19=mean(PD1$RT);
M20=mean(PD1$OD);
M=matrix(c(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16,M17,M18,M19,M20),nrow = 20)
PV1=PD1$SSC;
PV2= PD1$SSM;
PV3= PD1$SST;
PV4= PD1$MSC;
PV5= PD1$MSM;
PV6= PD1$MST;
PV7= PD1$FT;
PV8= PD1$DL;
PV9= PD1$DF;
PV10= PD1$BC;
PV11= PD1$PA;
PV12= PD1$OR;
PV13= PD1$DR;
PV14= PD1$S;
PV15= PD1$B;
PV16= PD1$Sp;
PV17= PD1$V;
PV18= PD1$St;
PV19= PD1$RT;
PV20= PD1$OD;
PV=matrix(c(PV1,PV2,PV3,PV4,PV5,PV6,PV7,PV8,PV9,PV10,PV11,PV12,PV13,PV14,PV15,PV16,PV17,PV18,PV19,PV20),nrow = 30)

for (i in 1:20)
{ 
  LBJ[i]=(LJ[i,1]-M[i,1])/S[i,1]
  
}
for (i in 1:20)
{ 
  TD[i]=(TD[i,1]-M[i,1])/S[i,1]
  
}
for (i in 1:20)
{ 
  MJ[i]=(MJ[i,1]-M[i,1])/S[i,1]
  
}
for (i in 1:20)
{ 
  LB[i]=(LB[i,1]-M[i,1])/S[i,1]
  
}
LBJ<-matrix(c(-1.323539, -0.5231858, 0.2982821, -1.089746, -0.4521834, 0.4525602, -0.6331559, 0.9172888 ,0.5958749, 0.4175822, 0.3294579 ,-0.8817733 ,-0.2691229, 0.1508305, -0.1211373, 0.7366212, 0.8010579, 1.171817, 0.3768984, 1.354328),nrow = 20)
TD<-matrix(c(0.3469472,-0.5231858,-1.6902653,0.223201,-0.5297006,-2.0249593,-0.4513887,-1.54373,-0.5608235,-1.7048399,-1.2443857,0.5904873,1.0350881,-0.7193456,1.3426056,-2.2219393,-1.757877,-0.9215728,0.2176456,-1.5676816),nrow = 20)
MJ
LB
LJ<-read.table("/Applications/regression analysis/final project/ daa3.txt", header = TRUE)

S1=sqrt(sum((PV1-M1)^2)/29)
S2=sqrt(sum((PV2-M2)^2)/29)
S3=sqrt(sum((PV3-M3)^2)/29)
S4=sqrt(sum((PV4-M4)^2)/29)
S5=sqrt(sum((PV5-M5)^2)/29)
S6=sqrt(sum((PV6-M6)^2)/29)
S7=sqrt(sum((PV7-M7)^2)/29)
S8=sqrt(sum((PV8-M8)^2)/29)
S9=sqrt(sum((PV9-M9)^2)/29)
S10=sqrt(sum((PV10-M10)^2)/29)
S11=sqrt(sum((PV11-M11)^2)/29)
S12=sqrt(sum((PV12-M12)^2)/29)
S13=sqrt(sum((PV13-M13)^2)/29)
S14=sqrt(sum((PV14-M14)^2)/29)
S15=sqrt(sum((PV15-M15)^2)/29)
S16=sqrt(sum((PV16-M16)^2)/29)
S17=sqrt(sum((PV17-M17)^2)/29)
S18=sqrt(sum((PV18-M18)^2)/29)
S19=sqrt(sum((PV19-M19)^2)/29)
S20=sqrt(sum((PV20-M20)^2)/29)
S=matrix(c(S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20),nrow = 20)

M1

#no use
LJ<-read.table("/Applications/regression analysis/final project/ daa3.txt", header = TRUE)
eigv<-read.table("/Applications/regression analysis/final project/ ta3.txt", header = TRUE)
L<-cbind(LJ)
e<-cbind(eigv)
P<-t(L)%*%e
t(LBJ)%*%
r%*%e
# so Z1=Z6=Z9=0
#new model is yhat=25.07+0.21Z2+0.11Z3+0.2Z4+0.18Z5+0.003Z7+0.14Z8
#给出球员数据做预测，并计算置信区间
#计算各项均值
#prove to have colinearility
newSP<-data.frame(X1,X2,X3,X4,X5,X6,X7,X8,X9)
CRSP=cor(newSP)
CRSP
eigen(CRSP)
eigen(CRSP)$value
sum(1/(eigen(CRSP)$value))
kappa(CRSP, exact=T)
newTP<-data.frame(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9)
#cr=correlationship
CRTP=cor(newTP)
CRTP
eigen(CRTP)
eigen(CRTP)$value
sum(1/(eigen(CRTP)$value))
kappa(CRTP, exact=T)
library(car)
PCTP<-cbind(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9)
PC2<-prcomp(PCTP,scale=TRUE,scores=TRUE)
summary(PC2)
PCTL<-cbind(PD1)
PCFM<-prcomp(PD1,scale=TRUE,scores=TRUE)
summary(PCFM)
CRTL<-cor(PD1)
eigen(CRTL)
vif(newTP$Z1,newTP,trace = FALSE)
#get C=Xwave*V

#变量共线形
#X1,X2;X2,X3;X3,X4;X4,X5;X5,X6;X6,X7;X7,X8;Z1,Z3;Z5,Z6;Z6,Z7
有共线形

#prove it has colinearility
library(car)
vif(RM1)
vif(RM2)
colinear<-lm(X5~X2+X3+X4+X1+X6+X7+X8+X9)
summary(colinear)
PCSP<-cbind(X1,X2,X3,X4,X5,X6,X7,X8,X9)
PC1<-prcomp(PCSP,scale=TRUE,scores=TRUE)
summary(PC1)
M1=(X1-mean(X1))/sqrt(sum((X1-mean(X1))^2))
M2=(X2-mean(X2))/sqrt(sum((X2-mean(X2))^2))
M3=(X3-mean(X3))/sqrt(sum((X3-mean(X3))^2))
M4=(X4-mean(X4))/sqrt(sum((X4-mean(X4))^2))
M5=(X5-mean(X5))/sqrt(sum((X5-mean(X5))^2))
M6=(X6-mean(X6))/sqrt(sum((X6-mean(X6))^2))
M7=(X7-mean(X7))/sqrt(sum((X7-mean(X7))^2))
M8=(X8-mean(X8))/sqrt(sum((X8-mean(X8))^2))
M9=(X9-mean(X9))/sqrt(sum((X9-mean(X9))^2))
V1=(Y-mean(Y))/sqrt(sum((Y-mean(Y))^2))
N1=(Z1-mean(Z1))/sqrt(sum((Z1-mean(Z1))^2))
N2=(Z2-mean(Z2))/sqrt(sum((Z2-mean(Z2))^2))
N3=(Z3-mean(Z3))/sqrt(sum((Z3-mean(Z3))^2))
N4=(Z4-mean(Z4))/sqrt(sum((Z4-mean(Z4))^2))
N5=(Z5-mean(Z5))/sqrt(sum((Z5-mean(Z5))^2))
N6=(Z6-mean(Z6))/sqrt(sum((Z6-mean(Z6))^2))
N7=(Z7-mean(Z7))/sqrt(sum((Z7-mean(Z7))^2))
N8=(Z8-mean(Z8))/sqrt(sum((Z8-mean(Z8))^2))
N9=(Z9-mean(Z9))/sqrt(sum((Z9-mean(Z9))^2))

M<-data.frame(V1,M1,M2,M3,M4,M5,M6,M7,M8,M9)
fit1<-lm(M$V1~.,data = M)
summary(fit1)
N<-data.frame(V1,N1,N2,N3,N4,N5)
fit2<-lm(N$V1~.,data = N)
M<-cbind(M1,M2,M3,M4,M5)
eigen(M)

newTP<-data.frame(Y,Z2,Z3,Z4,Z5,Z6,Z7,Z8)
RM3<-lm(newTP$Y~.,data = newTP)
summary(RM3)


?vif
#experiment process

colinea<-lm(X9~X1+X2+X3+X4+X5+X6+X7+X8)
summary(colinea)
PCSP<-cbind(X1,X2,X3,X4,X5,X6,X7,X8,X9)
PC1<-prcomp(PCSP,scale=TRUE,scores=TRUE)
summary(PC1)
PCTP<-cbind(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9)
PC2<-prcomp(PCSP,scale=TRUE,scores=TRUE)
summary(PC2)
SPn<-data.frame(Y,X1,X2,X3,X4)
RM4<-lm(SPn$Y~.,data=SPn)
summary(RM4)
TPn<-data.frame(Y,Z1,Z2,Z3,Z4,Z5)
RM3<-lm(TPn$Y~.,data=TPn)
summary(RM3)
