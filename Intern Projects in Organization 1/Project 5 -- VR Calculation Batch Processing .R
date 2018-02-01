###import needed packages and read dataset
library(readxl)
library(sqldf)
library(xlsx)

data_target <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/data1213.xlsx")
coefficient <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/R/Project 4/coefficient.xlsx")
test2 <- read_excel("//dcna_cifs.peacecorps.gov/users/jsun/My Documents/Data/test2.xlsx")

step1=sqldf("select * from data_target natural join coefficient where coefficient.AA=data_target.AA")
step2=sqldf("select * from step1 natural join test2 where step1.POST=test2.POST")
#fdata=sqldf("select Coefficient from coefficient union select * from data_target where coefficient.AA=data_target.AA")

write.xlsx(step2, "//dcna_cifs.peacecorps.gov/users/jsun/My Documents/R/Project 5/Project5.xlsx")
#Inv_deadline = -3.560086 + 1.475775*VR + AA +1.327504*Fre
