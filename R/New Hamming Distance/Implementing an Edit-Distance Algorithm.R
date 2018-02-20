#Implementing an Edit-Distance Algorithm
# write a function firstly
Hamming_Distance=function(word1,word2){
  # install and load packages
  options(repos='http://cran.rstudio.org')
  have.packages <- installed.packages()
  cran.packages <- c('base','utils')
  to.install <- setdiff(cran.packages, have.packages[,1])
  if(length(to.install)>0) 
    install.packages(to.install)
  library(utils)
  library(base)
  
  # split both words
  A=unlist(strsplit(word1, ""))
  B=unlist(strsplit(word2, ""))
  AA=c(A)
  BB=c(B)
  
  # start model based on different conditions
  SZ=c("Z","S")
  sz=c("z","s")
  count=sum(AA!=BB)
  for (i in 1:length(AA)) {
    if(i==1){
      if(AA[i]!=BB[i]){
        if (tolower(AA[i])==tolower(BB[i])){
          count=count-1
        } else if(((AA[i] %in% SZ) && (BB[i] %in% SZ))||
                  ((AA[i] %in% sz) && (BB[i] %in% sz))){
          count=count-1
        } else {count=count-0}
      } else {count=count-0}
    } else if(i>1){
      if(AA[i]!=BB[i]){
        if(tolower(AA[i])==tolower(BB[i])){
          count=count-0.5
        } else if(((AA[i] %in% SZ) && (BB[i] %in% SZ))||
                  ((AA[i] %in% sz) && (BB[i] %in% sz))){
          count=count-1
        } else if(((AA[i] == toupper(AA[i]))&&(BB[i] == toupper(BB[i])))||
                  ((AA[i] == tolower(AA[i]))&&(BB[i] == tolower(BB[i])))){
          count=count-0
        } else {count=count+0.5}
      } else {count=count-0}
    } else {count=count-0}
  }
  
  # manage output word
  Text="Our new hamming distance for two strings is:"
  output=list(Text,count)
  print(output)
}
# running function to solve the following 3 questions
#a)	"data Science" to  "Data Sciency"
first_word="data Science"
second_word="Data Sciency"
Hamming_Distance(first_word,second_word)
# the distance score here is 1.
#b)	"organizing" to "orGanising"
first_word="organizing"
second_word="orGanising"
Hamming_Distance(first_word,second_word)
# the distance score here is 0.5.
#c)	"AGPRklafsdyweIllIIgEnXuTggzF" to "AgpRkliFZdiweIllIIgENXUTygSF"
first_word="AGPRklafsdyweIllIIgEnXuTggzF"
second_word="AgpRkliFZdiweIllIIgENXUTygSF"
Hamming_Distance(first_word,second_word)
# the distance score here is 8.5
