library(shinydashboard)
library(shinyjs)

dashboardPage(
  
  dashboardHeader( title = "Questions similarities"),
  dashboardSidebar( selectInput("selection", "Choose a data set:",choices = books),
                    sidebarMenu(
                      menuItem("Words Analysis", tabName = "Word_analysis",startExpanded = TRUE),
                      menuItem("Feature Engineering" , tabName = "Feature_engineering",startExpanded = TRUE,
                               menuSubItem("How:",tabName="FEAT_ENG_HOW"),
                               menuSubItem("Show:",tabName="FEAT_ENG_SHOW")),
                      menuItem("Machine Learning" , tabName = "Machine_learning",startExpanded = TRUE,
                                menuSubItem("Classical Machines",tabName = "classic_machines"),
                                menuSubItem("Deep Learning",tabName = "deep_learning"))   )
                    ),
  dashboardBody(
    useShinyjs(),
    fluidRow(
      infoBoxOutput("MemoryUsage")
    ),
    fluidRow(
     
      box(title = "Samples selection", width = "100%",
          plotOutput("plotLabelProp",height = 250) ,
          sliderInput("numberex","Number of Examples:", min = 1,  max = 404290,  value = 1000,step = 1),
          sliderInput("labelproportion","Label proportion:", min = 0,  max = 1.,  value = 0.5,step = 0.01) )),

      box(title = 'Table of Selected Data Samples',width = 300, DT::dataTableOutput('dataset1'),collapsed = TRUE,collapsible = TRUE),
      tabItems(
      tabItem(tabName = "Feature_engineering" ),
      
      tabItem(tabName = "FEAT_ENG_HOW",
               box(width = "100%",
                   fluidRow(
                   box(title = "Words Vectorization GLOVE",
                       radioButtons("OPTIONS_WORD_VEC_GLOVE", label = h3("Select Glove Parameters:"),
                                    choices = list("Pretrained" = '1', "Retrain" = '2'), selected = 1),
                       sliderInput("GloveVectorsDimention","Vectors Dimention:", min = 10,  max = 300,  value = 100,step = 1),
                       sliderInput("GloveWindowLength","Window Length:", min = 1,  max = 30,  value = 5,step = 1),
                       sliderInput("x_max","x_max:", min = 1,  max = 30,  value = 10,step = 1),
                       sliderInput("n_iter","n_iter:", min = 1,  max = 30,  value = 5,step = 1)
                       ),
                   box(title = "Similariy Matrix",plotOutput("SIMILARITY_MATRIX_PLOT")),
                   box(title = "Questions Word Vectors",plotOutput("QUESTION_WORD_VEC_PLOT"),
                       radioButtons("OPTIONS_QUESTIONS_WORD_VEC", label = h3("Select Option"),
                                    choices = list("Question1 - Question2" = '1', "Question2 - Question1" = '2'), selected = 1) ),
                   box(title = "Word Vectors T-SNE 2D",
                       plotOutput("WORD_VEC_D2_TSNE"),
                       textInput("select_WORD", label = "Enter a Word"),
                       sliderInput("numberOfWordNeigbours","Number of Neighbours:", min = 10,  max = 200,  value = 15,step = 1) ),
                       
                   box(title = "Output Tests",textOutput("text_print_show") )
                   
                   ),
                   box( title = "Features Calculation",width = "100%",
                        box( title = "Words Statistics"            ,DT::dataTableOutput('FEAT1_TABLE',width = "700")),
                        box( title = "Gramatical Entity Statistics",DT::dataTableOutput('FEAT2_TABLE',width = "700"))
                       # box( title = "Words Statistics",textOutput("FEAT1_TABLE") ),
                       # box( title = "Gramatical Entity Statistics",textOutput("FEAT8_TABLE") )
                       
                        # DT::dataTableOutput('FEAT3_TABLE'),
                        # DT::dataTableOutput('FEAT4_TABLE'),
                        # DT::dataTableOutput('FEAT5_TABLE'),
                        # DT::dataTableOutput('FEAT6_TABLE'),
                        # DT::dataTableOutput('FEAT7_TABLE')
                       
                        
                   )
               )
              ),
      
      tabItem(tabName = "FEAT_ENG_SHOW",
           box( width = "100%",
              box(title = "Correlation Between Features:",
                  actionButton("Select_All_SHOW"   ,label = "Select All"),
                  actionButton("Unselect_All_SHOW" ,label = "Unselect All"),
                  selectInput(inputId = "Feat_select_SHOW",choices = NULL, label = "Select at lest 2 Features",multiple = TRUE),
                  plotOutput("FEATURES_CORRELATION")
                  ),
              
              box( title = "Densities:", selectInput(inputId = "Feat_select_Density",choices = NULL, label = "Select Features:",multiple = TRUE),
                   plotOutput("FEATURES_DENSITIES",dblclick = "ZOOM_FEATURES_DENSITIES",brush = brushOpts(id= "brush_FEATURES_DENSITIES", resetOnNew = TRUE))),
              box( title = "Scatter:",
                   selectInput(inputId = "Feat_select_Scatter_x",choices = NULL, label = "X",multiple = FALSE),
                   selectInput(inputId = "Feat_select_Scatter_y",choices = NULL, label = "y",multiple = FALSE),
                   plotOutput("FEATURES_SCATTER",dblclick = "ZOOM_FEATURES_SCATTER",
                              brush = brushOpts(
                                id = "brush_FEATURES_SCATTER",
                                resetOnNew = TRUE
                              ) )),
              
              box(title = "Scatter 3D:",
                  selectInput(inputId = "Feat_select_Scatter_3d_x",choices = NULL, label = "X",multiple = FALSE),
                  selectInput(inputId = "Feat_select_Scatter_3d_y",choices = NULL, label = "y",multiple = FALSE),
                  selectInput(inputId = "Feat_select_Scatter_3d_z",choices = NULL, label = "z",multiple = FALSE),
                  rglwidgetOutput("plot_scatter_3D_features",  width = 800, height = 600))
              
              
           )
           
      ),
      
      
    
      tabItem( tabName = "Word_analysis",
               
              fluidRow( 
                 box( title ="Commoun Words Pyrmaid",plotOutput("plotPyramidWords",dblclick = "ZOOM_PYRAMID_WORDS", brush = brushOpts(
                   id = "brush_PYRAMID_WORDS",
                   resetOnNew = TRUE
                 )),
                      sliderInput("commwords_wind","Window size:",  min = 1,  max = 1000,  value = c(1,10) ) ),
                 box(title = "World Cloud" , plotOutput("plot"),
                     sliderInput("freq","Minimum Frequency:",min = 1,  max = 50, value = 15),
                     sliderInput("max","Maximum Number of Words:",min = 1,  max = 1000,  value = 100)) ),

              fluidRow( 
                 box( title ="Comparaison Word Cloud",plotOutput("plotComparaisonCloud"),
                      sliderInput("maxcloudcomp","Maximum of Words:",min = 1,  max = 1000, value = 15)),
                 box( title= "Comoun Word Cloud"     ,plotOutput("plotCommounCloud"),
                      sliderInput("maxcloudcomm","Maximum of Words:",min = 1,  max = 1000, value = 15)))
               ),
      
      tabItem( tabName ="classic_machines",
               box(title = "Calculated Features:",DT::dataTableOutput('dataset_features',width = 2000),width=300,collapsed = TRUE,collapsible = TRUE),
               fluidRow(
                 box( title = "Machine Parameters:",
                      
                      box(title = "Data Partition:",plotOutput("plotTestProp"), sliderInput( "Test_Prop","Training/Test Percentage:",min = .01,max=1.,value=0.2)),
                      
                      box(title = "Features Selection:", 
                          radioButtons("MODE_MODEL_CLASSIC",label = "Choose Mode",choices = c('Pretrained','Retrain')),
                          actionButton("Select_All"   ,label = "Select All"),
                          actionButton("Unselect_All" ,label = "Unselect All"),
                          selectInput(inputId = "Feat_select",choices = NULL, label = "Select Features",multiple = TRUE),
                          
                          selectInput(inputId = "Select_Model", label = "Select Model",choices = Models),
                          selectInput(inputId = "Select_Model_pretrained", label = "Select Model",choices = Models),
                          actionButton("TRAIN_MODEL" ,label = "Train"),
                          selectInput(inputId = "glm", label = "Family",choices = c("binomial","bernoulli","gaussian","Gamma","inverse.gaussian","poisson","quasi","quasibinomial","quasipoisson") ),
                          sliderInput( "nnet","Hiden Layer Unites",min = 1,  max = 21, value = 5),
                          sliderInput("rpart","Number of branches",min = 1,  max = 1000, value = 5),
                          sliderInput( "randomForest1","mtry",min = 1,  max = 40, value = 5),
                          sliderInput("randomForest2","ntree",min = 1,  max = 1000, value = 5) ,
                          
                          selectInput( "svm1",choices = c("linear","radial","sigmoid","polynomial"), label = "Kernal"),
                          sliderInput( "svm2","cost" ,min = 0,  max = 20, value = 0),
                          sliderInput( "svm3","gamma" ,min = 0,  max = 20, value = 0),
                          
                          sliderInput( "xgboost1","eta" ,min = 0.01,  max = 3, value = 0,step = 0.01),
                          sliderInput( "xgboost2","max depth" ,min = 1,  max = 100, value = 0)
                        )
                      ),
                 
                 box(title= "Model Performance:", plotOutput("plotModelMetrics",height = 250), 
                     plotOutput("plotConfusionMatrix", height = 250 ) ),
                 
                 box(title = "Output Tests",textOutput("text_print") )

                 )
               ),
      tabItem( tabName ="deep_learning", 
                       box(title = "Models:", width = "100%",
                         radioButtons("MODE_MODEL",label = "Choose Mode",choices = c('Pretrained','Retrain')),
                         selectInput( "deep_learning_model",choices = c("model1","model2","model3","model4","model5","model6","model7","model8","model9","model10"),
                                      label = "Select Pretrained Model"),
                         sliderInput("DEEP_LEARN_EPOCHS",label = "Number of Epochs:",value = 1,min = 1,max = 20,step=1),
                         sliderInput("DEEP_LEARN_BATCH",label = "Batch Size:",min = 1,max = 300,value = 100,step = 1),
                         sliderInput("DEEP_LEARN_TSTRATIO",label = "Test/Train Ratio:",min = 0,max = 1,value = 0.1,step = 0.01),
                         
                        box(title = "Model ROC Curve",
                           plotOutput("DeepLearningModelPerf")
                       ),
                       box(title = "Accuracy and Error Curves",
                           plotOutput("DeepLearningModelPerfAccErr")
                       ),
                       box(title = "Model Topology",
                           textOutput( "DEEP_LEARN_MODEL_TOPO")
                       )
                       )
                       
                    ) 
               
          )
      )
    
)


