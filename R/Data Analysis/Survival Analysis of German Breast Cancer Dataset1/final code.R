#load library
library(KMsurv)
library(survival)
library(OIsurv)
library(survMisc)
library(readxl)
#library(pec)
#library(rms)
library(MASS)
library(car)

#import data
gbcs <- read_excel("~/Documents/6227/Final Project/gbcs.xls", 
                     sheet = "GBCS")
View(gbcs)
diff1<- difftime(gbcs$deathdate,gbcs$start, units = c("days"))

diff2<- difftime(gbcs$diagdateb,gbcs$start, units = c("days"))

diff3<- difftime(gbcs$recdate,gbcs$start, units = c("days"))
gbcs=data.frame(gbcs,diff1,diff2,diff3)

#model selection, do this firstly to check the vital variable
#total categorial
names(gbcs)
gbcs[,"grade"]=as.factor(gbcs$grade)
gbcs[,"menopause"]=as.factor(gbcs$menopause)
gbcs[,"hormone"] =as.factor(gbcs$hormone)
gbcs$age.stra=recode(gbcs$age, "lo:45=1; 46:55=2; 55:hi=3")
gbcs$size.stra=recode(gbcs$size, "lo:20=1; 21:30=2; 31:hi=3")
gbcs$nodes.stra=recode(gbcs$nodes, "lo:3=1; 4:9=2; 10:hi=3")
gbcs$pro.stra=recode(gbcs$prog_recp, "lo:20=1; 21:90=2 ; 91:hi=3")
gbcs$est.stra=recode(gbcs$estrg_recp, "lo:20=1; 21:90=2 ; 91:hi=3")

surv_fit<-coxph(Surv(survtime,censdead)~age+menopause+hormone+size+
                  grade+nodes+prog_recp+estrg_recp, data=gbcs)
rec_fit<-coxph(Surv(rectime,censrec)~age+menopause+hormone+size+
                 grade+nodes+prog_recp+estrg_recp, data=gbcs)
#summary(rec_fit)
#selection process
selection_surv=stepAIC(surv_fit)
selection_rec=stepAIC(rec_fit)

#the left variables are size grade 2 and 3 nodes and prog_recp
summary(selection_surv)
#the left variables are hormone2 size grade2 and 3 nodes and prog_recp
summary(selection_rec)
propcheck_surv=cox.zph(selection_surv, transform="km", global=TRUE)
propcheck_rec=cox.zph(selection_rec, transform="km", global=TRUE)
par(mfrow=c(2,2))
plot(propcheck_surv)
plot(propcheck_rec)

##then do this to check the interactive
###### Plot the estimated survival for different strata ######
attach(gbcs)
par(mfrow=c(1,2))
surv_fit<-coxph(Surv(survtime,censdead)~age.stra+strata(menopause)+
                  as.factor(grade), data=gbcs)
#relationship between menopause and survival, age bigger than 55
#grade 1
tp1=survfit(surv_fit, newdata=data.frame(grade=1,age.stra=3))
plot(tp1, ylim=c(0.76,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1,7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.865,0)

##on log-log scale
plot(tp1, xlim=c(75,2450),
     fun="cloglog",
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste("log[-log{", hat(S),"(t)}], stratified",sep="")))
#grade 2
tp2=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3))
plot(tp2, ylim=c(0.59,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.74,0)
#axis(side=2,at=seq(0.59,1,0.05))
##on log-log scale
plot(tp2, xlim=c(75,2450),
     fun="cloglog",
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste("log[-log{", hat(S),"(t)}], stratified",sep="")))
#grade 3
tp3=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=3))
plot(tp3,ylim=c(0.33,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.53,0)
#axis(side=2,at=seq(0.33,1,0.05))
##on log-log scale
plot(tp3, xlim=c(75,2400),
     fun="cloglog",
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste("log[-log{", hat(S),"(t)}], stratified",sep="")))

par(mfrow=c(1,2))
rec_fit<-coxph(Surv(rectime,censrec)~age.stra+strata(menopause)+as.factor(grade), data=gbcs)
#grade 1
tp4=survfit(rec_fit, newdata=data.frame(grade=1,age.stra=3))
plot(tp4, ylim=c(0.47,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.645,0)
#axis(side=2,at=seq(0.46,1,0.05))
##on log-log scale
plot(tp4, xlim=c(75,2450),
     fun="cloglog",
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste("log[-log{", hat(S),"(t)}], stratified",sep="")))
#grade 2
tp5=survfit(rec_fit, newdata=data.frame(grade=2,age.stra=3))
plot(tp5, ylim=c(0.33,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.5,0)
#axis(side=2,at=seq(0.59,1,0.05))
##on log-log scale
plot(tp5, xlim=c(75,2450),
     fun="cloglog",
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste("log[-log{", hat(S),"(t)}], stratified",sep="")))

tp6=survfit(rec_fit, newdata=data.frame(grade=3,age.stra=3))
plot(tp6,ylim=c(0.18,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.34,0)
#axis(side=2,at=seq(0.33,1,0.05))
##on log-log scale
plot(tp6, xlim=c(75,2400),
     fun="cloglog",
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste("log[-log{", hat(S),"(t)}], stratified",sep="")))
#so the survival rate for people older than 55 will be very low as the 
#increase of tumor grade, if they are not menopause

#check Progesterone and Estrogen，受体多可以阻碍激素
#与癌细胞接触
par(mfrow=c(1,2))
surv_fit<-coxph(Surv(survtime,censdead)~age.stra+as.factor(grade)+
                  strata(pro.stra), data=gbcs)
tpa=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3))
plot(tpa, ylim=c(0.4,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.42,0)
tpb=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=3))
plot(tpb, ylim=c(0.3,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.3,0)

par(mfrow=c(1,2))
surv_fit<-coxph(Surv(survtime,censdead)~age.stra+as.factor(grade)+
                  strata(est.stra), data=gbcs)
tpc=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3))
plot(tpc, ylim=c(0.43,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.44,0)
tpd=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=3))
plot(tpd, ylim=c(0.22,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.23,0)

#whether accept hormone therapy,即使年轻，grade小，受体多 also bad
##check Progesterone
par(mfrow=c(1,2))
surv_fit<-coxph(Surv(survtime,censdead)~age.stra+as.factor(grade)+pro.stra+
                  strata(hormone), data=gbcs)
tpe=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3,pro.stra=1))
plot(tpe, ylim=c(0.37,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.37,0)
tpf=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3,pro.stra=2))
plot(tpf, ylim=c(0.63,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.64,0)
tpg=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=3,pro.stra=1))
plot(tpg, ylim=c(0.23,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.246,0)
tph=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=1,pro.stra=1))
plot(tph, ylim=c(0.28,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.29,0)
# check Estrogen
par(mfrow=c(1,2))
surv_fit<-coxph(Surv(survtime,censdead)~age.stra+as.factor(grade)+est.stra+
                  strata(hormone), data=gbcs)
tpe1=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3,est.stra=1))
plot(tpe1, ylim=c(0.47,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.48,0)
tpf1=survfit(surv_fit, newdata=data.frame(grade=2,age.stra=3,est.stra=2))
plot(tpf1, ylim=c(0.6,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.61,0)
tpg1=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=3,est.stra=1))
plot(tpg1, ylim=c(0.23,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.26,0)
tph1=survfit(surv_fit, newdata=data.frame(grade=3,age.stra=1,est.stra=1))
plot(tph1, ylim=c(0.34,1),
     col=c("black","red","blue","green","yellow","cyan","brown"),
     lty = c(1, 7),
     main=expression(paste(hat(S), "(t), stratified",sep="")))
abline(0.342,0)

#regression diagnostic
#fit cox model
surv_fit<-coxph(Surv(survtime,censdead)~size+strata(grade)+nodes+prog_recp, data=gbcs)
#summary(surv_fit)
rec_fit<-coxph(Surv(rectime,censrec)~hormone+size+strata(grade)+nodes+prog_recp,
               data=gbcs)
#summary(rec_fit)

#get cox-snell residuals
surv_diff=gbcs$censdead-resid(surv_fit)
surv_haz=survfit(Surv(surv_diff,gbcs$censdead)~1,type="fl")
summary(surv_haz)

rec_diff=gbcs$censrec-resid(rec_fit)
rec_haz=survfit(Surv(rec_diff,gbcs$censrec)~1,type="fl")
summary(rec_haz)

#plot Hhat for cox-snell residuals
plot(surv_haz$time, -log(surv_haz$surv),ylab="H", xlab="r",main = "Stratatified")
abline(0,1)

#plot estimated survival for different strata, 用来对比说明不同荷尔蒙
#受体数，节点数对癌症的影响
#还没画，留白, 4个图


#martingale residuals
attach(gbcs)
names(gbcs)
scatter.smooth(age,resid(surv_fit))
scatter.smooth(size,resid(surv_fit))
scatter.smooth(prog_recp,resid(surv_fit))
scatter.smooth(estrg_recp,resid(surv_fit))
scatter.smooth(nodes,resid(surv_fit))

scatter.smooth(age,resid(rec_fit))
scatter.smooth(size,resid(rec_fit))
scatter.smooth(prog_recp,resid(rec_fit))
scatter.smooth(estrg_recp,resid(rec_fit))
scatter.smooth(nodes,resid(rec_fit))

#schoendfeld residuals

# deviance residual
##deviance
plot(surv_fit$linear.predictors, resid(surv_fit,type='deviance'),
     xlab="Risk Score",ylab="Deviance Residuals")
abline(0,0,lty=2,col='red')

plot(age, resid(surv_fit,type='deviance'),
     xlab="z2",ylab="Deviance Residuals")
abline(0,0,lty=2,col='red')

plot(rec_fit$linear.predictors, resid(rec_fit,type='deviance'),
     xlab="Risk Score",ylab="Deviance Residuals")
abline(0,0,lty=2,col='red')

plot(age, resid(rec_fit,type='deviance'),
     xlab="z2",ylab="Deviance Residuals")
abline(0,0,lty=2,col='red')





