source("coefinvd.R")
source("offdata.R")
library(shiny)
library(shinythemes)
library(stringr)
library(scales)

vars <- reactiveValues(chat=NULL, users=NULL)

# Restore the chat log from the last session.
if (file.exists("chat.Rds")){
  vars$chat <- readRDS("chat.Rds")
} else {
  vars$chat <- "Welcome to Shiny Chat!"
}

#' Get the prefix for the line to be added to the chat window. Usually a newline
#' character unless it's the first line.
linePrefix <- function(){
  if (is.null(isolate(vars$chat))){
    return("")
  }
  return("<br />")
}

shinyServer(function(input, output, session) {
  #eventReactive
  #observeEvent
  TT1=eventReactive(input$goButton1,{
    as.numeric(nameit[which(nameit$Country==input$region),1])
  })
  
  TT2=eventReactive(input$goButton2,{
    as.numeric(nameit[which(nameit$Country==input$region),1])
  })
  
  TT3=eventReactive(input$goButton4,{
    as.numeric(nameit[which(nameit$Country==input$region),1])
  })
  
  CFVR=eventReactive(input$goButton1,{
    as.numeric(xsvr[which(xsvr$AANum==input$AAVR),2])
  })
  
  CFINV=eventReactive(input$goButton2,{
    as.numeric(xsinv[which(xsinv$AANum==input$AAINV),2])
  })
  
  CFINVD=eventReactive(input$goButton4,{
    as.numeric(xsinvd[which(xsinvd$AANum==input$AAVR2),2])
  })
  
  ntext1 <- eventReactive(input$goButton1,{
    #-2.199078+1.043309*Fre+1.59647*EOD
    #equation for "get accepted Number"
    A=c( -2.199078+1.043309*TT1()+CFVR()+1.59647*input$VR)
  })
  
  gt1 <- eventReactive(input$goButton1,{
    "The number needed to accept invitation:"
  })
  output$ms1<- renderText({
    gt1()
  })
  output$acc <- renderText({
    ntext1()
  })
  
  ntext2 <- eventReactive(input$goButton2,{
    #3.276061-3.366614*Fre+1.137882*InvAcc
    #equation for "get invited number"
    B=c(3.276061-3.366614*TT2()+CFINV()+1.137882*input$InvAcc)
  })
  
  gt2 <- eventReactive(input$goButton2,{
    "The number needed to invite:"
  })
  output$ms2<- renderText({
    gt2()
  })
  
  output$inv <- renderText({
    ntext2()
  })
  
  ntext3 <- eventReactive(input$goButton2,{
    #equation for InvAcc/Invited
    C=c(ntext1()/ntext2()) 
  })
  
  gt3 <- eventReactive(input$goButton2,{
    "The Percentage InvAcc/Invited:"
  })
  output$ms3<- renderText({
    gt3()
  })
  
  output$per <- renderText({
    paste(100*ntext3(),"%",sep="")
  })
  
  ntext4 <- eventReactive(input$goButton3,{
    #equation for padding number
    D=c(ntext2()-input$EOD+input$VR-input$EOD-input$BF-input$attrition)
  })
  
  gt4 <- eventReactive(input$goButton3,{
    "The Retrospective Padding Number:"
  })
  output$ms4<- renderText({
    gt4()
  })
  
  output$padnum <- renderText({
    ntext4()
  })
  
  ntext5 <- eventReactive(input$goButton3,{
    #equation for padding rate
    E=c(ntext4()/input$VR)
  })
  
  gt5 <- eventReactive(input$goButton3,{
    "The Retrospective Padding Rate:"
  })
  output$ms5<- renderText({
    gt5()
  })
  
  output$padrate <- renderText({
    percent(ntext5())
  })
  
  
  #page 2
  #this is for EOD prediction
  SC <- eventReactive(input$GoButton,{
    if ((input$SubRgn=="Africa")||(input$SubRgn=="Asia")||(input$SubRgn=="Central America and Mexico")
        ||(input$SubRgn=="North Africa and the Middle East")||(input$SubRgn=="Pacific Islands")){
      if(as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.8957){
        if(as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.80235){
          if(as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.79595){
            if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.71875){
              if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.70295){
                1
              }
              else{
                0
              }
            }
            else{
              1
            }
          }
          else{
            0
          }
        }
        else{
          if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.8304){
            if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.8265){
              1
            }
            else{
              0
            }
          }
          else{
            1
          }
        }
      }
      else{
        if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.90325){
          if(input$GPA<2.966){
            if(input$Age<25.5){
              0
            }
            else{
              1
            }
          }
          else{
            0
          }
        }
        else{
          if((input$SubRgn=="Africa")||(input$SubRgn=="Asia")){
            if(input$Gender=="Male"){
              if(input$Sector=="Education"){
                if(input$Degree=="Bachelors"){
                  0
                }
                else{
                  1
                }
              }
              else{
                1
              }
            }
            else{
              1
            }
          }
          else{
            if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.98215){
              if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.9806){
                1
              }
              else{
                0
              }
            }
            else{
              if(input$GPA<2.7){
                if(input$Age<47.5){
                  0
                }
                else{
                  1
                }
              }
              else{
                if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.9843){
                  if(input$GPA<3.657){
                    1
                  }
                  else{
                    0
                  }
                }
                else{
                  0
                }
              }
            }
          }
        }
      }
    }
    else{
      if ( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.96965){
        if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.8628){
          1
        }
        else{
          if( as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.88675){
            0
          }
          else{
            if(as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.91785){
              1
            }
            else{
              if(input$Degree==2){
                if(input$Age<23.5){
                  1
                }
                else{
                  if((input$MedSort=="Medical Pending")||(input$MedSort=="Nomination Medical Pending")
                     ||(input$MedSort=="Nomination Validation Required")){
                    if(input$GPA<3.586){
                      if(input$ApproveNum<63.5){
                        if(input$GPA<3.4855){
                          0
                        }
                        else{
                          1
                        }
                      }
                      else{
                        1
                      }
                    }
                    else{
                      0
                    }
                  }
                  else{
                    0
                  }
                }
              }
              else{
                0
              }
            }
          }
        }
      }
      else{
        if(input$SubRgn=="Caribbean"){
          if(as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.99675){
            0
          }
          else{
            if(as.numeric(nameit[which(nameit$Country==input$Fre),1])<0.9979){
              1
            }
            else{
              0
            }
          }
        }
        else{
          1
        }
      }
    }
  })
  # generate bins based on input$bins from ui.R
  
  ntb=eventReactive(input$GoButton,{
     if(SC()==1){
       wenzi=c("Yes!")
     }
     
  })
  ntbb=eventReactive(input$GoButton,{
      if(SC()==0){
        wenzi=c("No!")
      }
    
  })
  reactive(ntb())
  output$yes<-renderText({
     ntb()
  })
  output$no<-renderText({
    ntbb()
  })
  #ntb1=eventReactive(input$GoButton,{
  #  "12"
  #  }
    ## Switch active tab to 'Page 1'
  #)
  #output$ACCC=renderText({
  #  ntb1()
  #})
  
  #page 3
  ntext6 <- eventReactive(input$goButton4,{
    
    G=c(-3.560086+1.475775*input$VR2+CFINVD()+1.327504*TT3())
  })
  
  gt6 <- eventReactive(input$goButton4,{
    "The Candidate should be invited at deadline:"
  })
  
  output$ms6<- renderText({
    gt6()
  })
  
  output$DDL <- renderText({
    ntext6()
  })
  
  sessionVars <- reactiveValues(username = "")
  
  # Track whether or not this session has been initialized. We'll use this to
  # assign a username to unininitialized sessions.
  init <- FALSE
  
  # When a session is ended, remove the user and note that they left the room. 
  session$onSessionEnded(function() {
    isolate({
      vars$users <- vars$users[vars$users != sessionVars$username]
      vars$chat <- c(vars$chat, paste0(linePrefix(),
                                       tags$span(class="user-exit",
                                                 sessionVars$username,
                                                 "left the room.")))
    })
  })
  
  # Observer to handle changes to the username
  observe({
    # We want a reactive dependency on this variable, so we'll just list it here.
    input$user
    
    if (!init){
      # Seed initial username
      sessionVars$username <- paste0("User", round(runif(1, 10000, 99999)))
      isolate({
        vars$chat <<- c(vars$chat, paste0(linePrefix(),
                                          tags$span(class="user-enter",
                                                    sessionVars$username,
                                                    "entered the room.")))
      })
      init <<- TRUE
    } else{
      # A previous username was already given
      isolate({
        if (input$user == sessionVars$username || input$user == ""){
          # No change. Just return.
          return()
        }
        
        # Updating username      
        # First, remove the old one
        vars$users <- vars$users[vars$users != sessionVars$username]
        
        # Note the change in the chat log
        vars$chat <<- c(vars$chat, paste0(linePrefix(),
                                          tags$span(class="user-change",
                                                    paste0("\"", sessionVars$username, "\""),
                                                    " -> ",
                                                    paste0("\"", input$user, "\""))))
        
        # Now update with the new one
        sessionVars$username <- input$user
      })
    }
    # Add this user to the global list of users
    isolate(vars$users <- c(vars$users, sessionVars$username))
  })
  
  # Keep the username updated with whatever sanitized/assigned username we have
  observe({
    updateTextInput(session, "user", 
                    value=sessionVars$username)    
  })
  
  # Keep the list of connected users updated
  output$userList <- renderUI({
    tagList(tags$ul( lapply(vars$users, function(user){
      return(tags$li(user))
    })))
  })
  
  # Listen for input$send changes (i.e. when the button is clicked)
  observe({
    if(input$send < 1){
      # The code must be initializing, b/c the button hasn't been clicked yet.
      return()
    }
    isolate({
      # Add the current entry to the chat log.
      vars$chat <<- c(vars$chat, 
                      paste0(linePrefix(),
                             tags$span(class="username",
                                       tags$abbr(title=Sys.time(), sessionVars$username)
                             ),
                             ": ",
                             tagList(input$entry)))
    })
    # Clear out the text entry field.
    updateTextInput(session, "entry", value="")
  })
  
  # Dynamically create the UI for the chat window.
  output$chat <- renderUI({
    if (length(vars$chat) > 500){
      # Too long, use only the most recent 500 lines
      vars$chat <- vars$chat[(length(vars$chat)-500):(length(vars$chat))]
    }
    # Save the chat object so we can restore it later if needed.
    saveRDS(vars$chat, "chat.Rds")
    
    # Pass the chat log through as HTML
    HTML(vars$chat)
  })
})

