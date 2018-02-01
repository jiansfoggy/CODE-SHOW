#load all packages and needed files
# this one is for Med Sort Friendliness data
source("offdata.R")
#the coefficent of Assignment Area
source("coefvr.R")
source("coefinv.R")
source("coefinvd.R")
library(shiny)
library(shinythemes)
library(scales)

shinyUI(
  tagList(shinythemes::themeSelector(),
  navbarPage(
             #theme = shinytheme("flatly"), 
             #includeCSS("gys.css"),
             #inverse=TRUE,
             #color:White;
             # a title
             strong("Go Go Peace Corps!", style="font-size:20px;"),
             #define the appearance of padding calculator
             tabPanel(strong("Padding Calculator", style="font-size:20px;"),
                      titlePanel(h1("Padding Calculator",align="center")),
                      br(),
                      sidebarLayout(
                        selectInput("region", "Post:", choices=nameit[,2]),
                        #divide the whole process into three parts
                        #part1
                        tabsetPanel(
                           tabPanel("Get Accepted Number",
                                   sidebarPanel(
                                     #define several buttons
                                     selectInput("AAVR", "Assignment Area:", choices=xsvr[,1]),
                                     numericInput("VR","Volunteer Request:" ,min=0, max=1000, value=60),
                                     br(),
                                     actionButton('goButton1', img(src="pclogo.png", height = 50, width = 95)),
                                     p("Click the button to update the value displayed in the main panel.")
                                   ),
                                   # mainPanel controls what will come as results
                                   mainPanel(
                                     h3(textOutput("ms1"),style="font-weight:bold;font-size=25px;text-align:center;background-color:Cornsilk ;color:SteelBlue ;"),
                                     #strong("The number need to accept invitation:"),
                                     div(verbatimTextOutput("acc"),style="font-weight:bold;font-size:16px;")
                                   )
                         ),
                         #part2
                         tabPanel("Get Invited Number",
                                 sidebarPanel(
                                   #define several buttons
                                   selectInput("AAINV", "Assignment Area:", choices=xsinv[,1]),
                                   numericInput("InvAcc", "Invitation Accepted:",min=0, max=1000, value=70),
                                   br(),
                                   actionButton('goButton2', img(src="pclogo.png", height = 50, width = 95)),
                                   p("Click the button to update the value displayed in the main panel.")
                                 ),
                                 # mainPanel controls what will come as results
                                 mainPanel(
                                   h3(textOutput("ms2"),style="font-weight:bold;font-size=25px;text-align:center;background-color:Cornsilk ;color:SteelBlue ;"),
                                   div(verbatimTextOutput("inv"),style="font-weight:bold;font-size:16px;"),
                                   h3(textOutput("ms3"),style="font-weight:bold;font-size=25px;text-align:center;background-color:Cornsilk ;color:SteelBlue ;"),
                                   div(verbatimTextOutput("per"),style="font-weight:bold;font-size:16px;")
                                 )
                         ),
                         #part3
                         tabPanel("Get Padding Rate",
                                 sidebarPanel(
                                   #define several buttons
                                   numericInput("EOD", "Actual EOD Number:",min=0, max=1000, value=1),
                                   numericInput("BF", "BackFill:",min=-100, max=100, value=0),
                                   numericInput("attrition", "PC Attrition:",min=0, max=100, value=0),
                                   br(),
                                   actionButton('goButton3', img(src="pclogo.png", height = 50, width = 95)),
                                   p("Click the button to update the value displayed in the main panel.")
                                 ),
                                 # mainPanel controls what will come as results
                                 mainPanel(
                                   h3(textOutput("ms4"),style="font-weight:bold;font-size=25px;text-align:center;background-color:Cornsilk ;color:SteelBlue ;"),
                                   div(verbatimTextOutput("padnum"),style="font-weight:bold;font-size:16px;"),
                                   h3(textOutput("ms5"),style="font-weight:bold;font-size=25px;text-align:center;background-color:Cornsilk ;color:SteelBlue ;"),
                                   div(verbatimTextOutput("padrate"),style="font-weight:bold;font-size:16px;")
                                 )
                        )
                      )
                    )
             ),
             #define the appearance of EOD Prediction
             tabPanel(strong("EOD Prediction", style="font-size:20px;"),
                      headerPanel(h1("EOD Prediction",align="center")),
                      sidebarLayout(
                        #define several buttons
                        sidebarPanel(
                          selectInput("Gender", "Gender:",choices = c("Male", "Female")),
                          selectInput("Race", "Race:",choices = c("Asian or Pacific Islanders","Black or African - American","Hispanic or Latino",
                                                                  "Indian or Native American or Alaskan Native","Not Specified","Two or More Races","White")),
                          numericInput("Age", "Age:",min=0, max=100, value=25),
                          selectInput("Degree", "Degree:",choices = c("Associates","Bachelors","Doctorate and JD", "Master and MBA","Other")),
                          numericInput("GPA", "GPA:" ,min=0, max=4, value=3.5),
                          selectInput("SubRgn", "Sub Region:", choices = c("Africa","Asia","Caribbean","Central America and Mexico",
                                                                           "Eastern Europe and Central Asia","North Africa and the Middle East",
                                                                           "Pacific Islands","South America")),
                          selectInput("Sector", "Sector:",choices=c("Agriculture","Business","Education","Environment","Health","Youth")),
                          selectInput("Fre", "Medical Friendliness:",choices=nameit[,2]),
                          selectInput("MedSort", "Med Sort:",choice=c("Additional validation required, support required",
                                                                     "Medical Pending",
                                                                     "No additional validation required, cleared for all countries",
                                                                     "Nomination Cleared","Nomination Medical Pending",
                                                                     "Nomination Validation Required","Null","Triage")),
                          numericInput("ApproveNum", "The number of countries on Med Sort list:",min=0, max=70, value=55),
                          br(),
                          actionButton('GoButton', img(src="pclogo.png", height = 50, width = 95)),
                          p("Click the button to update the value displayed in the main panel.")
                        ),
                        mainPanel(
                          # mainPanel controls what will come as results
                          includeHTML("include.html"),
                          br(),
                          br(),
                          br(),
                          br(),
                          div(textOutput("yes"), style="color:Green;font-size:500%;text-align:center;"),
                          div(textOutput("no"), style="color:Red;font-size:500%;text-align:center;"),
                          br(),
                          br()
                          #verbatimT textOutput("ACCC")
                        )
                      )
             ),
             tabPanel(strong("New VR Calculator", style="font-size:20px;"),
                      titlePanel(h1("New VR Calculator",align="center")),
                      br(),
                      sidebarLayout(
                        sidebarPanel(
                           selectInput("region", "Post:", choices=nameit[,2]),
                           #divide the whole process into three parts
                           selectInput("AAVR2", "Assignment Area:", choices=xsinvd[,1]),
                           numericInput("VR2","Volunteer Request:" ,min=0, max=1000, value=60),
                           br(),
                           actionButton('goButton4', img(src="pclogo.png", height = 50, width = 95)),
                           p("Click the button to update the value displayed in the main panel.")
                        ),
                        mainPanel(
                           h3(textOutput("ms6"),style="font-weight:bold;font-size=25px;text-align:center;background-color:Cornsilk ;color:SteelBlue ;"),
                           #strong("The number need to accept invitation:"),
                           div(verbatimTextOutput("DDL"),style="font-weight:bold;font-size:16px;")
                        )
                      )
             )
  ),
             bootstrapPage(
               # We'll add some custom CSS styling -- totally optional
               includeCSS("shinychat.css"),
               
               # And custom JavaScript -- just to send a message when a user hits "enter"
               # and automatically scroll the chat window for us. Totally optional.
               includeScript("sendOnEnter.js"),
               
               div(
                 # Setup custom Bootstrap elements here to define a new layout
                 class = "container-fluid", 
                 div(class = "row-fluid",
                     # Set the page title
                     tags$head(tags$title("Chat Room")),
                     
                     # Create the header
                     div(class="span6", style="padding: 10px 0px;",
                         h1("PC Chat Room"), 
                         h4("Feel Free to contact with developer here...")
                     ), div(class="span6", id="play-nice",
                            "IP Addresses are logged... be a decent human being."
                     )
                     
                 ),
                 # The main panel
                 div(
                   class = "row-fluid", 
                   mainPanel(
                     # Create a spot for a dynamic UI containing the chat contents.
                     uiOutput("chat"),
                     
                     # Create the bottom bar to allow users to chat.
                     fluidRow(
                       span(class="span10",
                           textInput("entry", "")
                       ),
                       span(class="span2 center",
                            actionButton("send", "Send")
                       )
                     )
                   ),
                   # The right sidebar
                   sidebarPanel(
                     # Let the user define his/her own ID
                     textInput("user", "Your User ID:", value=""),
                     tags$hr(),
                     h5("Connected Users"),
                     # Create a spot for a dynamic UI containing the list of users.
                     uiOutput("userList"),
                     tags$hr()
                    #helpText(HTML("<p>Built using R & <a href = \"http://rstudio.com/shiny/\">Shiny</a>.<p>Source code available <a href =\"https://github.com/trestletech/ShinyChat\">on GitHub</a>."))
                   )
                 )
               ),
               #define footer
               includeHTML("foot.html")
             )
  )
)

