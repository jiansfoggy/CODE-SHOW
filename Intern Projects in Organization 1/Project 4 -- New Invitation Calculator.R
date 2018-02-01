###import needed packages and read dataset
library(readxl)
library(sqldf)
padding <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/st2.xlsx", 
                      sheet = "Sheet2")
test2 <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/test2.xlsx")
Tique=sqldf("select * from padding natural join test2")
Tique=sqldf("select * from Tique order by SubRegion")
#Extraction
###factor needed data
#get variable
AA=as.factor(Tique$AA)
VR=Tique$VR
Inv=Tique$Invited
Inv_deadline=Tique$INVDeadline
Fre=Tique$`F/TC`
SubRegion=Tique$SubRegion
EOD=Tique$`Enter on Duty`

#Pad=data.frame(Sector, AA, Fre, SubRegion, InvAcc, EOD)
Pad=data.frame(SubRegion,Fre,AA,Inv,Inv_deadline, EOD)


### run The function to get equation
wlr=function(Fre,SubRegion,AA,Inv,Inv_deadline,EOD){
  #form data frame
  Padd=data.frame(SubRegion,Fre,AA,Inv_deadline,Inv,EOD)
  
  
  #get row number for different region
  Row=c()
  
  Row1=nrow(Padd[SubRegion=="Africa",])
  Row=append(Row,Row1)
  Row2=nrow(Padd[SubRegion=="Asia",])
  Row=append(Row,Row2)
  Row3=nrow(Padd[SubRegion=="Caribbean",])
  Row=append(Row,Row3)
  Row4=nrow(Padd[SubRegion=="Central America and Mexico",])
  Row=append(Row,Row4)
  Row5=nrow(Padd[SubRegion=="Eastern Europe and Central Asia",])
  Row=append(Row,Row5)
  Row6=nrow(Padd[SubRegion=="North Africa and the Middle East",])
  Row=append(Row,Row6)
  Row7=nrow(Padd[SubRegion=="Pacific Islands",])
  Row=append(Row,Row7)
  Row8=nrow(Padd[SubRegion=="South America",])
  Row=append(Row,Row8)
  
  #get residual for different regression according to differnt region
  Region=c()
  
  Africa=Padd[SubRegion=="Africa",]
  Africa=Africa[,-1]
  Region=append(Region, Africa)
  Asia=Padd[SubRegion=="Asia",]
  Asia=Asia[,-1]
  Region=append(Region, Asia)
  Caribbean=Padd[SubRegion=="Caribbean",]
  Caribbean=Caribbean[,-1]
  Region=append(Region, Caribbean)
  CAM=Padd[SubRegion=="Central America and Mexico",]
  CAM=CAM[,-1]
  Region=append(Region, CAM)
  EECA=Padd[SubRegion=="Eastern Europe and Central Asia",]
  EECA=EECA[,-1]
  Region=append(Region, EECA)
  NAME=Padd[SubRegion=="North Africa and the Middle East",]
  NAME=NAME[,-1]
  Region=append(Region, NAME)
  PI=Padd[SubRegion=="Pacific Islands",]
  PI=PI[,-1]
  Region=append(Region, PI)
  SA=Padd[SubRegion=="South America",]
  SA=SA[,-1]
  Region=append(Region, SA)
  ls=length(Region)
  
  JG_EOD=c()
  JG_INV=c()
  
  i=1
  
  while(i < 9){
    SET=Region[1:(ls/8)]
    hg=lm(SET$EOD~SET$Fre+SET$AA+SET$Inv_deadline,data=SET)
    JG1=summary(hg)$sigma
    JG_EOD=append(JG_EOD,JG1)
    #fc=lm(SET$Inv_deadline~SET$Fre+SET$AA+SET$Inv,data=SET)
    #JG2=summary(fc)$sigma
    #JG_INV=append(JG_INV,JG2)
    Region=Region[-1:-(ls/8)]
    i=i+1
  }
  
  
  #calculate weight for different region
  Weight_EOD=c()
  #Weight_INV=c()
  
  for(i in 1:8){
    #W=rep((Row[i]-1)/JG[i]^2,Row[i])
    W1=rep(JG_EOD[i]^2/(Row[i]-1),Row[i])
    Weight_EOD=append(Weight_EOD,W1)
    #W2=rep(JG_INV[i]^2/(Row[i]-1),Row[i])
    #Weight_INV=append(Weight_INV,W2)
  }
  #do regression
  
  GetPad=lm(Padd$EOD~Padd$Fre+Padd$AA+Padd$Inv_deadline,data=Padd, weights = Weight_EOD)
  SS1=summary(GetPad)
  #Getbad=lm(Padd$Inv_deadline~Padd$Fre+Padd$AA+Padd$Inv,data=Padd, weights = Weight_INV)
  #SS2=summary(Getbad)
  SS2=c()
  sc=list(SS1,SS2)
  
  return(sc)
}

wlr(Fre,SubRegion,AA,Inv,Inv_deadline,EOD)

#####the following is draft#################
