# Text of the books downloaded from:
# A Mid Summer Night's Dream:
#  http://www.gutenberg.org/cache/epub/2242/pg2242.txt
# The Merchant of Venice:
#  http://www.gutenberg.org/cache/epub/2243/pg2243.txt
# Romeo and Juliet:
#  http://www.gutenberg.org/cache/epub/1112/pg1112.txt


function(input, output, session) {
  # Define a reactive expression for the document term matrix
  
  data_set <- reactive({
    input$numberex
    input$labelproportion
    isolate({
      withProgress({
        setProgress(message = "Processing data set ...")
        set.seed(500000)
        # ifelse("../df.RData" %>% file.exists() ,load("../df.RData") ,df<- readQuoraDataSet())
        df %>% get_data_with_label_proprtion(input$labelproportion,input$numberex )
      })
    })

  })


  

labels_numbers <- reactive({
  list( duplicate_1 =  which(data_set()$is_duplicate == 1 ) %>% length(),duplicate_0 =  which(data_set()$is_duplicate == 0 )%>% length())
})
  
 
vocab_question1 <- reactive({
   isolate({
     data_set() %>% get_number_words_in_question1( )
   })
 }) 
 
vocab_question2 <- reactive({
  isolate({
    data_set() %>% get_number_words_in_question2( )
  })
}) 

comoun_vocab_questions <- reactive({
  isolate({
     data_set() %>% get_pyrmaid_words_data( )
     # vocab_question1() %>% full_join(vocab_question2() ,by = "term" ) %>% arrange(Words_in_question1 %>% desc() , Words_in_question2 %>% desc())
  })
})


pyrmaid_words_data_reactive <- reactive({
  input$commwords_wind 
  isolate({

    vocab_question1() %>% data_inner_join(vocab_question2()) %>% 
      gather_term_counts_of_questions( ) %>% 
      subset_data(input$commwords_wind )

  })
})




 
 get_number_words_in_questions_reactive <- reactive({
   data_set() %>%  get_number_words_in_questions()
   
 })
 
 
 features_data <- reactive({
   input$numberex
   isolate({
       Ids <- data_set()$id + 1
       features_data <- build_table_of_features(F,F,F,F,F,F) %>% as.data.frame()
       features_data[Ids,]
   })
   
   
 })
 
 autoTicks_2secondes <- reactiveTimer(2000)
 
 Resources_usage <- reactive({
     autoTicks_2secondes() 
    
     isolate({
      get_free_ram()
      
    })
  })
 
 get_selected_features_idx <- reactive({
       data_1 <- features_data()
       columns_feat <- data_1 %>% colnames() 
       if(input$Feat_select %>% length() == 0 ){
         columns_selected = columns_feat
       } else{
         columns_selected = input$Feat_select
       }
       which( columns_feat  %in% columns_selected )
 })
 

 
################   Features Engeering

# HOW
 
 
 
# SHOW
 
 Col_number_of_label <- reactive({
   features_data() %>% colnames() %>% length()
 })
 
 

 
 output$FEATURES_CORRELATION <- renderPlot({

   if( input$Feat_select_SHOW %>% length() > 1 ) {
     columns_    <- features_data() %>% colnames()
     columns_idx <- which( columns_  %in% input$Feat_select_SHOW  )
     
     features_data()[,columns_idx ] %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
       corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)
   }  
   

 })
 

 ######### CLASSIC Machine Learning
 train_test_data_subset <- reactive({
     input$Test_Prop
     input$numberex
     input$TRAIN_MODEL
     isolate({
        data_1 <- features_data()
        data_1[,get_selected_features_idx()] %>% create_data_partition( (1-input$Test_Prop) )
     })
   })
 
 
 load_pretrained_model_classic <- reactive({
   switch(input$Select_Model_pretrained,
          'glm' = readRDS('data/roc_data_glm.rds') ,
          'lda' = readRDS('data/roc_data_lda.rds'),
          'qda' = readRDS('data/roc_data_qda.rds'),
          'naiveBayes' = readRDS('data/roc_data_naivb.rds'),
          'nnet' = readRDS('data/roc_data_nnet.rds'),
          'rpart' = readRDS('data/roc_data_rpart.rds'),
          'randomForest' = readRDS('data/roc_data_rfrst.rds'),
          'default' = readRDS('data/roc_data_glm.rds') )

 })
 
 
 
 
 train_model_test_metrics <- reactive({
   input$Test_Prop
   input$numberex
   input$TRAIN_MODEL
   input$Select_Model
   input$Select_Model_pretrained
   input$labelproportion
   input$glm
   input$nnet
   input$rpart
   input$randomForest1
   input$randomForest2
   input$svm1
   input$svm2
   input$svm3
   input$xgboost1
   input$xgboost2
   isolate({
     
     switch(input$MODE_MODEL_CLASSIC,
                  'Retrain'    = {
              withProgress({
                setProgress(message = "Train the model ...")
                machine_parameters <- list( 
                  glm = input$glm,
                  rpart = input$rpart,
                  nnet = input$nnet,
                  ntree         = input$randomForest1,
                  mtry          = input$randomForest2,
                  kernal        = input$svm1,
                  cost          = input$svm2,
                  gamma         = input$svm3,
                  eta           = input$xgboost1,
                  max_depth     = input$xgboost2)
                train_test_data_subset() %>% get_model_train.predict_metrics(Model_Alg = input$Select_Model,machine_parameters)
              })
              
            },
            'Pretrained' = load_pretrained_model_classic(),
            'default'    = load_pretrained_model_classic()
     )
     
     
     

     
   })
 })
 
 


  output$plotLabelProp <- renderPlot({
     lablel_ratio <- data.frame( NumberOfExamples =  c(labels_numbers( )$duplicate_1,labels_numbers( )$duplicate_0), Category = c("Duplicated","Non-duplicated")  )
    

    lablel_ratio %>% ggplot(aes(fill=Category, y=NumberOfExamples, x=Category)) + 
      geom_bar(stat = "identity", position = "dodge") +
      ylab("Number of Examples:") +
      coord_flip()
    
  })
  
  output$dataset1 <- DT::renderDataTable({
    data_set( )%>% DT::datatable( options = list(pageLength = 25),selection = 'single')
  })
  
   
  output$dataset_features <- DT::renderDataTable({
    features_data( )%>% DT::datatable( options = list(pageLength = 25))
  })

  output$FEAT8_TABLE <- renderText({
   req(input$dataset1_rows_selected)
   v <- data_set( )[input$dataset1_rows_selected,] %>% get_features2()
   v[1,]
  }) 
  
  output$FEAT1_TABLE <- renderText({
    req(input$dataset1_rows_selected)
    data_set( )[input$dataset1_rows_selected,] %>% get_features1()[1,]
  }) 
      
  # output$FEAT1_TABLE <- DT::renderDataTable({
  #   # data_set( )[input$dataset1_rows_selected,] %>% get_features1() %>% DT::datatable( options = list(pageLength = 1,searching = FALSE,paging = FALSE))
  #   data_set( )[input$dataset1_rows_selected,] %>% get_features1()%>%   summary() %>% str()
  #   
  #   
  # })
  
  
  #   output$FEAT2_TABLE <- DT::renderDataTable({
  #       data_set( )[input$dataset1_rows_selected,] %>% get_features2()%>% DT::datatable( options = list(pageLength = 1,searching = FALSE,paging = FALSE))
  #   
  #   
  # })
  #   output$FEAT3_TABLE <- DT::renderDataTable({
  #   
  #   
  # })
  #   output$FEAT4_TABLE <- DT::renderDataTable({
  #   
  #   
  # })
  # output$FEAT5_TABLE <- DT::renderDataTable({
  #   
  #   
  # })
  #   output$FEAT6_TABLE <- DT::renderDataTable({
  #   
  #   
  # })
  # 
  #   output$FEAT6_TABLE <- DT::renderDataTable({
  #     
  #     
  #   })    
    
    
  
  
  load_word_vectors <- reactive({
    load_word_vectors_glove()
  })
  
  
  output$SIMILARITY_MATRIX_PLOT <- renderPlot({
    
    data_set( )[input$dataset1_rows_selected,] %>% plot_conv_matrix_q1q2(word_vectors = load_word_vectors() )
    
   
  })
  
  
  wordcloud_rep <- repeatable(wordcloud)
  
  output$plot <- renderPlot({
    # v <- terms()
    
    v <- comoun_vocab_questions() 
    wordcloud_rep(words = v$term,freq =  v$term_count , scale=c(4,0.5),
                  min.freq = input$freq, max.words=input$max,
                  colors=brewer.pal(8, "Dark2"))
  })
  
  ranges_ZOOM_FEATURES_SCATTER   <- reactiveValues(x = NULL, y = NULL)
  ranges_ZOOM_FEATURES_DENSITIES <- reactiveValues(x = NULL, y = NULL)
  
  
  output$FEATURES_SCATTER <- renderPlot({
    req(input$Feat_select_Scatter_x )
    req(input$Feat_select_Scatter_y )
    

    features_data( ) %>% ggplot(aes(input$Feat_select_Scatter_x %>% get(), input$Feat_select_Scatter_y %>% get(),colour= is_duplicate))+
      geom_point() +
      xlab(input$Feat_select_Scatter_x) + ylab(input$Feat_select_Scatter_y)+
      coord_cartesian(xlim = ranges_ZOOM_FEATURES_SCATTER$x, ylim = ranges_ZOOM_FEATURES_SCATTER$y, expand = FALSE)
    
  })
  
  observeEvent(input$ZOOM_FEATURES_SCATTER, {
    brush <- input$brush_FEATURES_SCATTER
    if (!is.null(brush)) {
      ranges_ZOOM_FEATURES_SCATTER$x <- c(brush$xmin, brush$xmax)
      ranges_ZOOM_FEATURES_SCATTER$y <- c(brush$ymin, brush$ymax)
      
    } else {
      ranges_ZOOM_FEATURES_SCATTER$x <- NULL
      ranges_ZOOM_FEATURES_SCATTER$y <- NULL
    }
  })
  
  output$FEATURES_DENSITIES <- renderPlot({
    if( input$Feat_select_Density %>% length() ==1 ) {
      columns_    <- features_data() %>% colnames()
      columns_idx <- which(columns_ ==input$Feat_select_Density[1] | columns_ == "is_duplicate")
      features_ <- features_data()[,columns_idx] 
      features_ %>% ggplot( aes(get(input$Feat_select_Density[1]))) + 
        geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)+
        coord_cartesian(xlim = ranges_ZOOM_FEATURES_DENSITIES$x, ylim = ranges_ZOOM_FEATURES_DENSITIES$y, expand = FALSE)
    }  else if( input$Feat_select_Density %>% length() !=0 ){
      columns_    <- features_data() %>% colnames()
      columns_idx <- which(columns_ %in% input$Feat_select_Density & columns_ != "is_duplicate")
      features_ <- features_data()[,columns_idx] %>% gather( key = variable_name, value= distance_value  )
      features_ <- cbind(features_,is_duplicate = features_data()$is_duplicate)
      features_ %>% ggplot( aes(distance_value,  colour=variable_name)) + 
        geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)+
        coord_cartesian(xlim = ranges_ZOOM_FEATURES_DENSITIES$x, ylim = ranges_ZOOM_FEATURES_DENSITIES$y, expand = FALSE)
    }
  })
  
  observeEvent(input$ZOOM_FEATURES_DENSITIES, {
    brush <- input$brush_FEATURES_DENSITIES
    if (!is.null(brush)) {
      ranges_ZOOM_FEATURES_DENSITIES$x <- c(brush$xmin, brush$xmax)
      ranges_ZOOM_FEATURES_DENSITIES$y <- c(brush$ymin, brush$ymax)
      
    } else {
      ranges_ZOOM_FEATURES_DENSITIES$x <- NULL
      ranges_ZOOM_FEATURES_DENSITIES$y <- NULL
    }
  })

  
  ranges_ZOOM_PYRAMID_WORDS   <- reactiveValues(x = NULL, y = NULL)

  observeEvent(input$ZOOM_PYRAMID_WORDS, {
    brush <- input$brush_PYRAMID_WORDS
    if (!is.null(brush)) {
      ranges_ZOOM_PYRAMID_WORDS$x <- c(brush$xmin, brush$xmax)
      ranges_ZOOM_PYRAMID_WORDS$y <- c(brush$ymin, brush$ymax)
      
    } else {
      ranges_ZOOM_PYRAMID_WORDS$x <- NULL
      ranges_ZOOM_PYRAMID_WORDS$y <- NULL
    }
  })
  
  
  output$plotPyramidWords <- renderPlot({
    v = pyrmaid_words_data_reactive()
    
    ggplot(data = v, 
           mapping = aes(x = term, fill = WordsCount, 
                         y = ifelse(WordsCount == "Words_in_question1", yes = -term_count, no = term_count))) +
      geom_bar(stat = "identity") +
      scale_y_continuous(limits = max(v$term_count) * c(-1,1)) +
      coord_cartesian(xlim = ranges_ZOOM_PYRAMID_WORDS$x, ylim = ranges_ZOOM_PYRAMID_WORDS$y, expand = FALSE) +
      labs(y = "Words Count", x = "Words") +
      coord_flip() 

  })
  
  output$plotCommounCloud <- renderPlot({
    comoun_vocab_1_2 <- get_number_words_in_questions_reactive()
    
    row.names(comoun_vocab_1_2) = comoun_vocab_1_2$term
    comoun_vocab_1_2 <- comoun_vocab_1_2 %>% dplyr::select(Words_in_question1,Words_in_question2)
    colnames(comoun_vocab_1_2)  = c("question1", "question2")
    display.brewer.all()
    pal <- brewer.pal(8,"Purples")
    pal <- pal[-(1:4)]
    
    comoun_vocab_1_2 <- comoun_vocab_1_2[1:input$maxcloudcomm,]
    comoun_vocab_1_2 %>% commonality.cloud(max.words = input$maxcloudcomm, random.order = FALSE,  colors=brewer.pal(8, "Dark2"))
    
  })
  

  output$plotComparaisonCloud <- renderPlot({
    comoun_vocab_1_2 <- get_number_words_in_questions_reactive()
    
    row.names(comoun_vocab_1_2) = comoun_vocab_1_2$term
    comoun_vocab_1_2 <- comoun_vocab_1_2 %>% dplyr::select(Words_in_question1,Words_in_question2)
    colnames(comoun_vocab_1_2)  = c("question1", "question2")
    display.brewer.all()
    pal <- brewer.pal(8,"Purples")
    pal <- pal[-(1:4)]
    
    comoun_vocab_1_2 <- comoun_vocab_1_2[1:input$maxcloudcomp,]
    comoun_vocab_1_2 %>% comparison.cloud( max.words = input$maxcloudcomp, random.order = FALSE, title.size = 1.0,  colors=brewer.pal(8, "Dark2"))
  })
  
  
  output$plotTestProp <- renderPlot({
    
    Max_ = 100
    data_set_raws <- data_set() %>% nrow()
    
    Partition <- c("Train","Test")
    Percentage <- c((1-input$Test_Prop )*Max_ , input$Test_Prop *Max_)
    ## create data frame
    colour.df <- data.frame(Partition, Percentage)

    ## calculate percentage 
    colour.df$percentage = Max_ * input$Test_Prop %>%  c((1-.),.)
    colour.df = colour.df$percentage %>% order() %>% rev( ) %>% colour.df[. , ]
    colour.df$ymax = colour.df$percentage %>% cumsum()
    colour.df$ymin = colour.df$ymax %>% head(n = -1) %>%  c(0, .)
    
    colour.df$label_data <-   c( data_set_raws * (1-input$Test_Prop ) , data_set_raws * input$Test_Prop )
    
    ggplot(colour.df, aes(fill = Partition, ymax = ymax, ymin = ymin, xmax = Max_, xmin = 80)) +
      geom_rect(colour = "black") +
      coord_polar(theta = "y") + 
      xlim(c(0, 100)) +
      # geom_label_repel(aes(label = percentage %>% round(2) %>% paste("%"), x = Max_, y = (ymin + ymax)/2),inherit.aes = F, show.legend = F, size = 5)+
      geom_label_repel(aes(label = label_data , x = Max_, y = (ymin + ymax)/2),inherit.aes = F, show.legend = F, size = 5)+
      theme(legend.title = element_text(colour = "black", size = 16, face = "bold"), 
            legend.text = element_text(colour = "black", size = 15), 
             panel.grid = element_blank(),
             axis.text = element_blank(),
             axis.title = element_blank(),
             axis.ticks = element_blank() )  +
      ggplot2::annotate("text", x = 0, y = 0, size = 5, label = "Test/Train" )
 
    
    
  })
  
  output$plot_scatter_3D_features <- renderRglwidget({
    req(input$Feat_select_Scatter_3d_x)
    req(input$Feat_select_Scatter_3d_y )
    req(input$Feat_select_Scatter_3d_z)
    
    colname_ <- features_data() %>% colnames() 
    x_idx = which(colname_ == input$Feat_select_Scatter_3d_x)
    y_idx = which(colname_ == input$Feat_select_Scatter_3d_y)
    z_idx = which(colname_ == input$Feat_select_Scatter_3d_z)
    

      rgl.open(useNULL=T)
      scatter3d(x=features_data()[,x_idx], y=features_data()[,y_idx], z=features_data()[,z_idx], 
                surface=FALSE, 
                point.col = features_data()$is_duplicate %>% as.numeric(),
                xlab = input$Feat_select_Scatter_3d_x,
                ylab = input$Feat_select_Scatter_3d_y,
                zlab = input$Feat_select_Scatter_3d_z)
      
      rglwidget()
      
    
    
    
  })
  
  output$plotModelMetrics <- renderPlot({
    
    train_model_test_metrics()$roc_data %>% ggplot(aes(x = fpr, y = tpr)) +
      geom_line(aes(color = "red")) + geom_abline(intercept=0, slope=1, lty=3) + 
      ylab('True Positive Rate') + xlab('False Positive Rate') +  ggtitle("ROC Curve")

    
  })
  
  output$plotConfusionMatrix <- renderPlot({
    
    train_model_test_metrics()$conf_matr %>%  ggplotConfusionMatrix(input$Select_Model)
    
  })
  
  output$MemoryUsage <- renderInfoBox({
    Resources_usage() %>% as.character() %>% paste0(., " ", "Giga Bytes") %>% infoBox(" Memory Usage:", ., icon = icon("list"),
                               color = "purple")
  })
  

  output$CPU <- renderInfoBox({
    
    
  })
  
  output$text_print <- renderText({
    features_data() %>% colnames() 
  })
  
  
  output$text_print_show <- renderText({
    input$dataset1_rows_selected

    # x_idx = which(colname_ == input$Feat_select_Scatter_3d_x)

    
  })

  
  
  
 observe({
   input$selection
   input$numberex
   comoun_vocab_1_2 <- data_set() %>% get_number_words_in_questions()
   updateSliderInput(session, "commwords_wind", max = comoun_vocab_1_2 %>% nrow()) 
   
 }) 
 
 get_feature_names <- reactive({
    feat <- features_data() %>% colnames()
    
    feat[- which(feat == "is_duplicate") ]
 })
 
 observeEvent(input$Unselect_All,{
   features_data() %>% colnames() %>% updateSelectInput(session ,inputId = "Feat_select",selected = c( ), choices = .)
 })
 
 observeEvent(input$Select_All,{
   cbind(get_feature_names(),"is_duplicate") %>% updateSelectInput(session ,inputId = "Feat_select",selected = ., choices = . )
 })
 

 observeEvent(input$Unselect_All_SHOW,{
   features_data() %>% colnames() %>% updateSelectInput(session ,inputId = "Feat_select_SHOW",selected = c( ), choices = .)
 })
 
 observeEvent(input$Select_All_SHOW,{
   get_feature_names() %>% updateSelectInput(session ,inputId = "Feat_select_SHOW",selected = ., choices = . )
 })
 

 
 observe({
   input$Feat_select_SHOW
   updateSelectInput(session ,inputId = "Feat_select_SHOW",choices = get_feature_names() ,selected = input$Feat_select_SHOW )
   
 })
 
 
 observe({
   input$Feat_select
   updateSelectInput(session ,inputId = "Feat_select",choices = features_data() %>% colnames() ,selected = input$Feat_select )
   
 })
 
 
 observe({
   input$Feat_select_SHOW
   updateSelectInput(session ,inputId = "Feat_select_Density",choices = features_data() %>% colnames() ,selected = input$Feat_select_Density )
   
 })
 
 
 observe({
   input$Feat_select_Scatter
   updateSelectInput(session ,inputId = "Feat_select_Scatter",choices = features_data() %>% colnames() ,selected = input$Feat_select_Scatter )
   
 })
 
 observe({
   input$Feat_select_Scatter_x
   updateSelectInput(session ,inputId = "Feat_select_Scatter_x",choices = features_data() %>% colnames() ,selected = input$Feat_select_Scatter_x )
   
 })
 
 observe({
   input$Feat_select_Scatter_y
   updateSelectInput(session ,inputId = "Feat_select_Scatter_y",choices = features_data() %>% colnames() ,selected = input$Feat_select_Scatter_y )
   
 })
 
 
 observe({
   input$Feat_select_Scatter_3d_x
   updateSelectInput(session ,inputId = "Feat_select_Scatter_3d_x",choices = features_data() %>% colnames() ,selected = input$Feat_select_Scatter_3d_x )
   
 })
 
 observe({
   input$Feat_select_Scatter_3d_y
   updateSelectInput(session ,inputId = "Feat_select_Scatter_3d_y",choices = features_data() %>% colnames() ,selected = input$Feat_select_Scatter_3d_y )
   
 })
 
 observe({
   input$Feat_select_Scatter_3d_z
   updateSelectInput(session ,inputId = "Feat_select_Scatter_3d_z",choices = features_data() %>% colnames() ,selected = input$Feat_select_Scatter_3d_z )
   
 })
 
 
 
 observeEvent(input$MODE_MODEL_CLASSIC,{

   shinyjs::hide(id = "Select_Model_pretrained")
   
   shinyjs::hide(id = "Select_Model") 
   shinyjs::hide(id = "TRAIN_MODEL")

   shinyjs::hide(id = "Unselect_All")
   shinyjs::hide(id = "Feat_select") 
   shinyjs::hide(id = "Select_All")
   
   switch(input$MODE_MODEL_CLASSIC,
          "Retrain" = {shinyjs::show(id = "Select_Model") 
                       shinyjs::show(id = "TRAIN_MODEL")

                       shinyjs::show(id = "Select_All")
                       shinyjs::show(id = "Unselect_All")
                       shinyjs::show(id = "Feat_select") 
                       
                       },
          "Pretrained" = {
            shinyjs::show(id = "Select_Model_pretrained")

            },
          "default"= shinyjs::show(id = "Select_Model_pretrained") )
   
 }) 
 
 
 observeEvent(input$Select_Model,{
    shinyjs::hide(id = "glm")
    shinyjs::hide(id = "rpart") 
    shinyjs::hide(id = "nnet")
    shinyjs::hide(id = "randomForest1")
    shinyjs::hide(id = "randomForest2")
    shinyjs::hide(id = "svm1")
    shinyjs::hide(id = "svm2")
    shinyjs::hide(id = "svm3")
    shinyjs::hide(id = "xgboost1")
    shinyjs::hide(id = "xgboost2")
    
    
    if(input$MODE_MODEL_CLASSIC == "Retrain" )
    {
    switch( 
      input$Select_Model,
      "glm"     = shinyjs::show(id = "glm", anim = TRUE),
      "nnet"    = shinyjs::show(id = "nnet", anim = TRUE),
      "rpart"   = shinyjs::show(id = "rpart", anim = TRUE),
      "randomForest" ={ shinyjs::show(id = "randomForest1", anim = TRUE)
        shinyjs::show(id = "randomForest2", anim = TRUE) },
      "svm"     = {
        shinyjs::show(id = "svm1", anim = TRUE)
        shinyjs::show(id = "svm2", anim = TRUE)
        shinyjs::show(id = "svm3", anim = TRUE)},
      "xgboost" ={ 
        shinyjs::show(id = "xgboost1", anim = TRUE)
        shinyjs::show(id = "xgboost2", anim = TRUE)
      },
      
      "default" = shinyjs::show(id = "glm", anim = TRUE)
      )
    }
    

    
    
    
    
    

  })
 

## Retrain deep learning models
 
 deep_learn_model <- reactive({
   switch(input$deep_learning_model,
                             'model1' = { FLAGS %>% model1() },
                             'model2' = { FLAGS %>% model2() },
                             'model3' = { FLAGS %>% model3() },
                             'model4' = { FLAGS %>% model4() },
                             'model5' = { FLAGS %>% model5() },
                             'model6' = { FLAGS %>% model6(keras_load_data()$word_vectors_glove) },
                             'model7' = { FLAGS %>% model7(keras_load_data()$word_vectors_glove) },
                             'model8' = { FLAGS %>% model8(keras_load_data()$word_vectors_glove) },
                             'model9' = { FLAGS %>% model9(keras_load_data()$word_vectors_glove) },
                             'model10' = { FLAGS %>% model10(keras_load_data()$word_vectors_glove) },
                             default  = { FLAGS %>% model1() } )
 })
 
 
 keras_load_data <- reactive({
   load_keras_question_data()
 })
 
 
 keras_questions <- reactive( {
    input$numberex
    input$labelproportion
    isolate({

        Ids <- data_set()$id + 1
     
        list(question1_ =  keras_load_data()$question1[Ids,],
             question2_ =  keras_load_data()$question2[Ids,],
             word_vector = keras_load_data()$word_vectors_glove,
             Labels      = df$is_duplicate[Ids] %>% as.numeric())
     
   })
   
 })
 
 reatrain_deep_learning_model <- reactive({
   input$deep_learning_model_retrain
   input$deep_learning_model
   input$MODE_MODEL
   input$DEEP_LEARN_TSTRATIO
   input$DEEP_LEARN_BATCH
   input$DEEP_LEARN_EPOCHS
   isolate({
     
     switch(input$MODE_MODEL,
            'Retrain'    =  {
              
              deep_learn_model() %>% train_model(keras_questions()$question1_,keras_questions()$question2_,keras_questions()$Labels,
                                                 input$DEEP_LEARN_TSTRATIO,
                                                 input$DEEP_LEARN_BATCH,
                                                 input$DEEP_LEARN_EPOCHS)
              
            },
            'Pretrained' = {
              history <- switch( input$deep_learning_model,
                                        "model1" = {
                                          load('data/validation1.RData')
                                          list(validation = validation_data1,history =  readRDS('data/histmodel1.rds'),model = deep_learn_model() )
                                        },
                                        "model2" = {
                                          load('data/validation2.RData')
                                          list(validation = validation_data2,history =  readRDS('data/histmodel2.rds'),model = deep_learn_model() )
                                        },
                                        "model3" = {
                                          load('data/validation3.RData')
                                          list(validation = validation_data3,history =  readRDS('data/histmodel3.rds'),model = deep_learn_model())
                                        },
                                        "model4" = {
                                          load('data/validation4.RData')
                                          list(validation = validation_data4,history =  readRDS('data/histmodel4.rds'),model = deep_learn_model() )
                                        },
                                        "model5" = {
                                          load('data/validation5.RData')
                                          list(validation = validation_data5,history =  readRDS('data/histmodel5.rds'),model = deep_learn_model() )
                                        },
                                        "model6" = {
                                          load('data/validation6.RData')
                                          list(validation = validation_data6,history =  readRDS('data/histmodel6.rds'),model = deep_learn_model()) 
                                        },
                                        "model7" = {
                                          load('data/validation7.RData')
                                          list(validation = validation_data7,history =  readRDS('data/histmodel7.rds'),model = deep_learn_model() )
                                        },
                                        "model8" = {
                                          load('data/validation8.RData')
                                          list(validation = validation_data8,history =  readRDS('data/histmodel8.rds'),model = deep_learn_model() )
                                        },
                                        "model9" = {
                                          load('data/validation9.RData')
                                          list(validation = validation_data9,history =  readRDS('data/histmodel9.rds'),model = deep_learn_model() )
                                        },
                                        "model10" = {
                                          load('data/validation10.RData')
                                          list(validation = validation10,history =  readRDS('data/histmodel10.rds'),model = deep_learn_model() )
                                        },
                                        default = {
                                          load('data/validation1.RData')
                                          list(validation = validation_data1,history =  readRDS('data/histmodel1.rds'),model = deep_learn_model() )
                                        } )
            }  )
     
   })
   
   
 })
 
 
 output$DEEP_LEARN_MODEL_TOPO <- renderPrint( {

   deep_learn_model() %>% print()
 })
 

 output$DeepLearningModelPerf <- renderPlot({
   
   roc_data   <-   reatrain_deep_learning_model()$validation %>%  
                                get_roc_data_for_keras_model( model_name = input$deep_learning_model)
   
   ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
     geom_line(aes(color='blue')) + geom_abline(intercept=0, slope=1, lty=3) +
     ylab('True Positive Rate') + xlab('False Positive Rate') +  ggtitle("ROC Curve")
   
 })
 
 output$DeepLearningModelPerfAccErr <- renderPlot({
   reatrain_deep_learning_model()$history %>% plot()    
 })
 
 

 observeEvent(input$MODE_MODEL,{
   switch( 
     input$MODE_MODEL,
     "Pretrained" = {
       shinyjs::hide(id = "DEEP_LEARN_EPOCHS")
       shinyjs::hide(id = "DEEP_LEARN_BATCH") 
       shinyjs::hide(id = "DEEP_LEARN_TSTRATIO")
     },
     "Retrain"     = {
       shinyjs::show(id = "DEEP_LEARN_EPOCHS", anim = TRUE)
       shinyjs::show(id = "DEEP_LEARN_BATCH", anim = TRUE)
       shinyjs::show(id = "DEEP_LEARN_TSTRATIO", anim = TRUE)
       },
     
     "default" = {
       shinyjs::hide(id = "DEEP_LEARN_EPOCHS")
       shinyjs::hide(id = "DEEP_LEARN_BATCH") 
       shinyjs::hide(id = "DEEP_LEARN_TSTRATIO")
     }
   )
 })
 
 
}
