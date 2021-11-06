library(RColorBrewer)
library(ggrepel)
require(class)
require(MASS)
require(randomForest)
require(e1071)
require(nnet)
require(ROCR)
require(xgboost)
require(neuralnet)
library(tm)
library(wordcloud)
library(keras)
library(dplyr)
library(ggplot2)
library(scales)

library(readr)
library(purrr)
library(tidytext)
library(tidyr)
library(caret)

library(corrplot)
library(pryr)
library(rgl)
library(car)

# library(data.table)
# library(qdapRegex)
# library(stringr)
# library(qdap)
# library(stringi)

# library(lsa)
# library(Rtsne)
# library(tokenizers)
# library(TraMineR)
# library(plot3D)
# library(textstem)
# library(textclean)
# library(text2vec)

# library(SnowballC)
# library(grid)


# require(NLP)
# require(openNLP)
# require(openNLPdata)

# library(openNLPmodels.en)

# The list of valid books
books <- list("Quora" = "quora",
              "Data set2" = "merchant",
              "data set3" = "romeo")

Models <- list( "Global Linear Regression" = "glm" ,
                "Tree" = "rpart" ,
                "Random Forest" = "randomForest" ,
               "Naive Bayes" = "naiveBayes" ,
               "qda" = "qda",
               "nnet" = "nnet",
               "svm"  = "svm",
               "xgboost" = "xgboost"
               # "knn" = "knn",
               )



readQuoraDataSet <- function()
{
  get_file(
    "quora_duplicate_questions.tsv",
    "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv" )
  "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv" %>% get_file("quora_duplicate_questions.tsv", . ) %>% read_tsv()
  
}


ifelse("data/df.RData" %>% file.exists() ,load("data/df.RData") ,df<- readQuoraDataSet())

number_samples <- df %>% nrow()


get_data_with_label_proprtion <- function(df,label_proprtion,number_of_entries)
{

  idx_label_1 = which(df$is_duplicate == 1 ) 
  idx_label_0 = which(df$is_duplicate == 0 ) 
  
  label0_number <- idx_label_0 %>% length()
  label1_number <- idx_label_1 %>% length()

  number_of_0_rows = label_proprtion * number_of_entries
  number_of_0_rows <- number_of_0_rows %>% floor()
  number_of_1_rows = number_of_entries - number_of_0_rows
  
  number_of_0_rows <- ifelse(label0_number > number_of_0_rows , number_of_0_rows, label0_number )
  number_of_1_rows <- ifelse(label1_number > number_of_1_rows , number_of_1_rows, label1_number )
    
  idx_label_0 <- sample(idx_label_0,size = number_of_0_rows)  %>% as.integer()
  idx_label_1 <- sample(idx_label_1,size = number_of_1_rows)  %>% as.integer()
  
  Idxs <- c(idx_label_1, idx_label_0  ) 
  
  df1 <- df[Idxs,]

  df1
}




# Using "memoise" to automatically cache the results
getTermMatrix <- function(df,book) {
  # Careful not to let just any name slip in here; a
  # malicious user could manipulate this value.
  if (!(book %in% books))
    stop("Unknown data set")
    if(book == "quora") {

      rbind(df$question1,df$question2) %>% VectorSource( ) %>% Corpus( ) %>%
      # tm_map( content_transformer(tolower)) %>%
      # tm_map(removePunctuation) %>%
      # tm_map(removeNumbers) %>%
      tm_map( removeWords, c(stopwords(language = 'en'), "thy", "thou", "thee", "the", "and", "but")) %>%
      TermDocumentMatrix( control = list(minWordLength = 1)) %>%
      as.matrix() %>%
      rowSums() %>%
      sort( decreasing = TRUE)

     } else{
      book %>% sprintf("%s.txt.gz", .) %>% readLines(encoding="UTF-8") %>%
      VectorSource( ) %>% Corpus( ) %>%
      tm_map( content_transformer(tolower)) %>%
      tm_map(removePunctuation) %>%
      tm_map(removeNumbers) %>%
      tm_map( removeWords, c(stopwords("SMART"), "thy", "thou", "thee", "the", "and", "but")) %>%
      TermDocumentMatrix( control = list(minWordLength = 1)) %>%
      as.matrix() %>%
      rowSums() %>%
      sort( decreasing = TRUE)
  }
}

get_number_words_in_question1 <- function(dfQ)
{
  vocab1 <- dfQ$question1 %>% space_tokenizer( ) %>% itoken( progressbar = FALSE ) %>%  create_vocabulary( ) %>% 
    as.data.frame() %>% arrange( term %>% desc()) %>% unique()

  colnames(vocab1) = c("term","Words_in_question1","doc_count1" )
  vocab1

}

get_number_words_in_question2 <- function(dfQ)
{
  vocab2 <- dfQ$question2 %>% space_tokenizer( ) %>% itoken( progressbar = FALSE)  %>%  create_vocabulary( )  %>% 
    as.data.frame() %>% arrange( term %>% desc()) %>% unique()
  
  colnames(vocab2) = c("term","Words_in_question2","doc_count2" )

  vocab2
}


subset_data <- function(dfQ, Interval)
{
  dfQ[Interval[1]:Interval[2],]
}

data_inner_join <- function(dfQ1,dfQ2)
{
  dfQ1 %>% inner_join(dfQ2,by = "term" ) %>% 
    mutate_all(funs(ifelse(is.na(.),0,.))) %>% 
    arrange(Words_in_question1 %>% desc() , Words_in_question2 %>% desc())
}




get_number_words_in_questions <- function(dfQ)
{
  vocab1 <- dfQ$question1 %>% space_tokenizer( ) %>% itoken( progressbar = FALSE ) %>%  
    create_vocabulary( )  


  vocab2 <- dfQ$question2 %>% space_tokenizer( ) %>% itoken( progressbar = FALSE)  %>%  
    create_vocabulary( )  
  
  
  colnames(vocab2) = c("term","Words_in_question2","doc_count2" )
  colnames(vocab1) = c("term","Words_in_question1","doc_count1" )
  
  vocab1 %>% full_join(vocab2,by = "term" ) %>% 
    mutate_all(funs(ifelse(is.na(.),0,.))) %>% 
    arrange(Words_in_question1 %>% desc() , Words_in_question2 %>% desc())  
}


get_pyrmaid_words_data <- function(dfQ)
{
  dfQ %>% 
    get_number_words_in_questions() %>% as.data.frame() %>% 
    gather(key = WordsCount ,value = term_count,-term,-doc_count1,-doc_count2) %>% 
    mutate( WordsCount = as.factor(WordsCount) ) %>% 
    arrange(term_count %>% desc() ,term %>% desc() )

}

gather_term_counts_of_questions <- function(dfQ)
{
  dfQ %>% 
  gather(key = WordsCount ,value = term_count,-term,-doc_count1,-doc_count2) %>% 
    mutate( WordsCount = as.factor(WordsCount) ) %>% 
    arrange(term_count %>% desc() ,term %>% desc() )
  
}


 # v <- df %>% get_number_words_in_questions() %>% gather_term_counts_of_questions
 # v %>% View()

get_pyrmaid_words_data_by_inval <- function(dfQ,Interval)
{
    dfQ %>% get_pyrmaid_words_data()[Interval[1]:Interval[2],]
  
}



# Build table of all extracted features -----

 build_table_of_features <- function( no_cosine_distance_word_vec, no_cosine_distance_by_dtm,no_traminr_seq,no_gramatic,no_word_numbers,no_line_col_word_prop)
 {

  load('data/df.RData')

  if (!no_cosine_distance_word_vec){
     load('data/question_similarity_distances.RData')
    df_more_features1 = data.frame(q2_q1_cos_dist_max_l = question_similarity_distances$q2_q1_cos_dist_max_l,
                                   q1_q2_cos_dist_max_c = question_similarity_distances$q1_q2_cos_dist_max_c,
                                   q1_q2_cos_dist_max_l = question_similarity_distances$q1_q2_cos_dist_max_l,
                                   q2_q1_cos_dist_max_c = question_similarity_distances$q2_q1_cos_dist_max_c,
                                   q2_q1_cos_dist_min_l = question_similarity_distances$q2_q1_cos_dist_min_l,
                                   q1_q2_cos_dist_min_c = question_similarity_distances$q1_q2_cos_dist_min_c,
                                   q1_q2_cos_dist_min_l = question_similarity_distances$q1_q2_cos_dist_min_l,
                                   q2_q1_cos_dist_min_c = question_similarity_distances$q2_q1_cos_dist_min_c
    )

    df_more_features= df_more_features1

  }
  if(!no_cosine_distance_by_dtm ) {
    load('data/df_simelarities.RData')
    load('data/Relaxed_Word_Mover_dist.RData')
    df_more_features1 = data.frame(tfidf_lsa_cos_sim   = df_simelarities$tfidf_lsa_cos_sim,
                                   tfidf_cos_sim       = df_simelarities$tfidf_cos_sim,
                                   tfidf_lsa_dist_m2   = df_simelarities$tfidf_lsa_dist_m2,
                                   tfidf_lsa_dist_m3   = df_simelarities$tfidf_lsa_dist_m3,

                                   d1_d2_cosine_sim    = df_simelarities$d1_d2_cosine_sim,
                                   rwmd_dist_colmean = Relaxed_Word_Mover_dist$rwmd_dist_colmean,
                                   rwmd_dist_rowmean = Relaxed_Word_Mover_dist$rwmd_dist_rowmean,
                                   rwmd_dist_diag    = Relaxed_Word_Mover_dist$rwmd_dist_diag
    )


    if (no_cosine_distance_word_vec ) {df_more_features = df_more_features1
    } else { df_more_features = cbind(df_more_features,df_more_features1) }

  }
  if(!no_traminr_seq)
  {
    load('data/questions_Tag_sequences_distances.RData')
    load('data/questions_word_sequence_distances.RData')
    df_more_features1 = data.frame(
      # traminr_seqLLCS = questions_Tag_sequences_distances$traminr_seqLLCS,
      traminr_seqLLCP = questions_Tag_sequences_distances$traminr_seqLLCP,
      # traminr_seqLLCS_ws = questions_word_sequence_distances$traminr_seqLLCS,
      traminr_seqLLCP_ws = questions_word_sequence_distances$traminr_seqLLCP
      # traminr_seqmpos_ws = questions_word_sequence_distances$traminr_seqmpos
      # traminr_seqmpos = questions_Tag_sequences_distances$traminr_seqmpos
    )
    if (no_cosine_distance_word_vec | no_cosine_distance_by_dtm ) {
      df_more_features = df_more_features1
    } else { df_more_features = cbind(df_more_features,df_more_features1) }

  }

  if(! no_gramatic )
  {
    load('data/question_gramatical_entities_pca.RData')
    load('data/question_gramatical_entities_cosine.RData')
    question_gramatical_entities_pca = question_gramatical_entities_pca %>% as.data.frame()

    df_more_features1 = data.frame(
      gramatical_entities_cosine  =  question_gramatical_entities_cosine$gramatical_entities_cosine,
      gramatical_entities_q1_PCA1 = question_gramatical_entities_pca$question1_PCA1,
      gramatical_entities_q1_PCA2 = question_gramatical_entities_pca$question1_PCA2,
      gramatical_entities_q1_PCA3 = question_gramatical_entities_pca$question1_PCA3,
      gramatical_entities_q1_PCA4 = question_gramatical_entities_pca$question1_PCA4,
      gramatical_entities_q2_PCA1 = question_gramatical_entities_pca$question2_PCA1,
      gramatical_entities_q2_PCA2 = question_gramatical_entities_pca$question2_PCA2,
      gramatical_entities_q2_PCA3 = question_gramatical_entities_pca$question2_PCA3,
      gramatical_entities_q2_PCA4 = question_gramatical_entities_pca$question2_PCA4
    )

    if (no_cosine_distance_word_vec | no_cosine_distance_by_dtm | no_traminr_seq) {
      df_more_features = df_more_features1
    } else { df_more_features = cbind(df_more_features,df_more_features1) }
  }

  if(! no_word_numbers)
  {
    load('data/question1_number_of_words.RData')
    load('data/question2_number_of_words.RData')
    load('data/question_delta_words.RData')
    load('data/question_no_stop_words.RData')

    df_more_features1 = data.frame(
      question1_no_words = question1_number_of_words$question1_no_words,
      question2_no_words = question2_number_of_words$question2_no_words,
      # delta_q1_q2 = question_delta_words$delta_q1_q2,
      # delta_q2_q1 = question_delta_words$delta_q2_q1,
      question1_no_stop_words = question_no_stop_words$question1_no_stop_words,
      question2_no_stop_words = question_no_stop_words$question2_no_stop_words
    )

    if (no_cosine_distance_word_vec | no_cosine_distance_by_dtm | no_traminr_seq | no_word_numbers) {
      df_more_features = df_more_features1
    } else { df_more_features = cbind(df_more_features,df_more_features1) }

  }

  if(! no_line_col_word_prop)
  {
    load('data/similar_words_proprtion.RData')
    df_more_features1 = data.frame(
        sim_words_on_line_proportion_10   = similar_words_proprtion$sim_words_on_line_proportion_10,
        sim_words_on_column_proportion10  = similar_words_proprtion$sim_words_on_column_proportion10,
        sim_words_on_line_proportion_9    = similar_words_proprtion$sim_words_on_line_proportion_9,
        sim_words_on_column_proportion9   = similar_words_proprtion$sim_words_on_column_proportion9,
        sim_words_on_line_proportion_8    = similar_words_proprtion$sim_words_on_line_proportion_8,
        sim_words_on_column_proportion8   = similar_words_proprtion$sim_words_on_column_proportion8,
        sim_words_on_line_proportion_7    = similar_words_proprtion$sim_words_on_line_proportion_7,
        sim_words_on_column_proportion7   = similar_words_proprtion$sim_words_on_column_proportion7,
        sim_words_on_line_proportion_6    = similar_words_proprtion$sim_words_on_line_proportion_6,
        sim_words_on_column_proportion6   = similar_words_proprtion$sim_words_on_column_proportion6,
        sim_words_on_line_proportion_5    = similar_words_proprtion$sim_words_on_line_proportion_5,
        sim_words_on_column_proportion5   = similar_words_proprtion$sim_words_on_column_proportion5,
        sim_words_on_line_proportion_4    = similar_words_proprtion$sim_words_on_line_proportion_4,
        sim_words_on_column_proportion4   = similar_words_proprtion$sim_words_on_column_proportion4
    )

    if (no_cosine_distance_word_vec | no_cosine_distance_by_dtm | no_traminr_seq | no_word_numbers |no_line_col_word_prop ) {
      df_more_features = df_more_features1
    } else { df_more_features = cbind(df_more_features,df_more_features1) }


  }


  df_more_features = cbind(df_more_features, is_duplicate = df$is_duplicate %>% as.factor( ) )
  df_more_features
 }
  # df_more_features = build_table_of_features(F,F,F,F,F,F) %>% as.data.frame()
  # 
  #  v = df_more_features %>% colnames() %>% [-which(df_more_features %>% colnames() == "is_duplicate")]
  # v["is_duplicate"]
   
   # 

create_data_partition <- function(Data,Percentage)
{
  set.seed(504023)  
  trainIndex <- createDataPartition(Data$is_duplicate, p = Percentage, list = FALSE, times = 1)
  df_more_features_train_ <- Data[ trainIndex,]
  df_more_features_test_  <- Data[-trainIndex,]  
  list( train = df_more_features_train_, test = df_more_features_test_)
}

  # data.partition = df_more_features[1:3000,] %>% create_data_partition(0.8)
 # 
 # data.partition %>% get_model_train.predict_metrics(Model_Alg = "xgboost", machine_parameters = 1)
 # Model_Alg = "xgboost"
get_model_train.predict_metrics <- function(data.partition,Model_Alg,machine_parameters)
{

  number_label_column = which( data.partition$train %>% colnames() == "is_duplicate")
  
  
  if (Model_Alg == 'glm')
  {
    model.fits <-   switch( machine_parameters$glm ,
             "binomial"  = glm(is_duplicate~. , data = data.partition$train ,family = binomial(link = 'logit')  ),
             "bernoulli" = glm(is_duplicate~. , data = data.partition$train ,family = bernoulli  ),
             "gaussian"  = glm(is_duplicate~. , data = data.partition$train ,family = gaussian(link = 'identity')  ),
             "Gamma"     = glm(is_duplicate~. , data = data.partition$train ,family = Gamma(link = "inverse")  ),
             "inverse.gaussian" = glm(is_duplicate~. , data = data.partition$train ,family = inverse.gaussian(link = "1/mu^2")  ),
             "poisson"   = glm(is_duplicate~. , data = data.partition$train ,family = poisson(link = "log") ),
             "quasi"     = glm(is_duplicate~. , data = data.partition$train ,family = quasi(link = "identity", variance = "constant")  ),
             "quasibinomial" = glm(is_duplicate~. , data = data.partition$train ,family = quasibinomial(link = "logit")  ),
             "quasipoisson"  = glm(is_duplicate~. , data = data.partition$train ,family = quasipoisson(link = "log")  ),
              default     = glm(is_duplicate~. , data = data.partition$train ,family = binomial(link = 'logit')  ) 
            )
  
    
  } else if(Model_Alg == 'lda') {
    model.fits = lda(is_duplicate~. , data = data.partition$train  )
    
  } else if(Model_Alg == 'qda') {
    data.partition$train$is_duplicate = data.partition$train$is_duplicate %>% as.character() %>% as.numeric() 
    model.fits = qda(is_duplicate~. , data = data.partition$train  ) 
  } else if(Model_Alg == 'knn') {
    model.fits = knn( train = data.partition$train , test = data.partition$test ,
                      cl = data.partition$train$is_duplicate, k=20)     
    
    
  } else if(Model_Alg == 'rpart')
  {
    model.fits = rpart(is_duplicate ~.  , data = data.partition$train )
    
  } else if(Model_Alg == 'randomForest')
  {
    model.fits = randomForest(is_duplicate ~. , data = data.partition$train, ntree = machine_parameters$ntree, mtry = machine_parameters$mtry)
  } else if(Model_Alg == 'naiveBayes'){
    model.fits <- naiveBayes(is_duplicate ~. , data = data.partition$train)
    
  } else if(Model_Alg == 'nnet') {

    model.fits <- nnet(is_duplicate ~., data = data.partition$train, size = machine_parameters$nnet)
  } else if(Model_Alg == 'svm' )
  {

    if( machine_parameters$gamma == 0 | machine_parameters$cost == 0 ) {
      model.fits <- svm(is_duplicate ~. , data = data.partition$train,kernel=machine_parameters$kernal, probability = T)      
    } else if( machine_parameters$gamma == 0  ) {
      model.fits <- svm(is_duplicate ~. , data = data.partition$train,kernel=machine_parameters$kernal, probability = T, cost = machine_parameters$cost)      
    } else if( machine_parameters$cost == 0) {
      model.fits <- svm(is_duplicate ~. , data = data.partition$train,kernel=machine_parameters$kernal, probability = T,gamma = machine_parameters$gamma)      
    } else {
      model.fits <- svm(is_duplicate ~. , data = data.partition$train,kernel=machine_parameters$kernal, probability = T,gamma = machine_parameters$gamma, cost = machine_parameters$cost)
    }

    
  } else if(Model_Alg == 'neuralnet'){
 

    maxs <- apply(data.partition$train[,-collabel] , 2, max) 
    mins <- apply(data.partition$train[,-collabel] , 2, min) 
    train_data <- data.partition$train[,-collabel] %>% scale(center = mins, scale = maxs-mins) %>% as.data.frame()
    train_data = cbind( train_data, is_duplicate = data.partition$train[,collabel] %>% as.numeric)
    
    maxs <- apply(data.partition$test[,-collabel] , 2, max) 
    mins <- apply(data.partition$test[,-collabel] , 2, min) 
    test_data <- data.partition$test[,-collabel] %>% scale(center = mins, scale = maxs-mins) %>% as.data.frame()
    test_data = cbind( test_data , is_duplicate = data.partition$test[,collabel] %>% as.numeric() )

    formula1 = test_data[,-number_label_column] %>% colnames %>%  paste0(collapse = ' + ') 
    formula1 = paste0('is_duplicate ~ ' ,formula1) %>%  as.formula()
    
    model.deepnn <- neuralnet(formula1 , data=train_data , hidden=c(50,50,50,50),act.fct = "logistic",
                              linear.output = FALSE)
  } else if( Model_Alg == 'xgboost'){

    train_data_xgboost  = xgb.DMatrix(data = data.partition$train[,-number_label_column] %>% as.matrix(),
                                     label = data.partition$train[,number_label_column] %>% as.character() %>% as.numeric() )

    model.fits  <- xgboost(data = train_data_xgboost , max_depth = machine_parameters$max_depth, eta = machine_parameters$eta, nthread = 2, nrounds = 2, objective = "binary:logistic")
    
  }
  
  
  if(Model_Alg == 'rpart'  | Model_Alg == 'randomForest' ){
    model.probs = predict(model.fits,data.partition$test, type = "prob")[,2] %>% as.numeric()
  } else if(Model_Alg == 'lda' | Model_Alg == 'qda' )
  {
    model.probs = predict(model.fits,data.partition$test, type="response")$posterior[,2]%>% as.numeric() 
  } else if(Model_Alg == 'naiveBayes') {
    model.probs <- predict(model.fits, newdata = data.partition$test,type ="raw")[,2] %>% as.numeric()
    
  } else if(Model_Alg == 'nnet'){
    
    
    model.probs <- predict(model.fits, newdata = data.partition$test, type = "raw")%>% as.numeric()
    
  } else if(Model_Alg == 'glm')
  {
    model.probs = predict(model.fits,data.partition$test, type="response") %>% as.numeric()
  } else if(Model_Alg == 'svm')
  {
    model.probs = predict(model.fits,data.partition$test, type = 'prob') %>% as.numeric()
    model.probs = ifelse(model.probs == 2 ,1,0)
  } else if(Model_Alg == 'neuralnet' )
  {
    
    model.probs = compute(model.deepnn,test_data[,-27])$net.result[,1]
    
    
  } else if( Model_Alg == "xgboost"){
    test_data_xgboost  = xgb.DMatrix(data = data.partition$test[,-number_label_column] %>% as.matrix(), label = data.partition$test[,number_label_column] %>% as.character() %>% as.numeric() )
  
    model.probs = model.xgboost %>% predict( test_data_xgboost)
    data.partition$test$is_duplicate = data.partition$test$is_duplicate %>% as.character() %>% as.numeric()
  }
  
  
  
  model.pred = ifelse(model.probs > .5,1,0)
  
  # model.pred = model.pred %>% as.numeric()
  cross.classify.table = table(model.pred ,data.partition$test$is_duplicate)
  
  conf_matr = confusionMatrix(model.pred ,data.partition$test$is_duplicate) 
  
  precision <- cross.classify.table[2,2] / sum(cross.classify.table[,2])
  # Recall
  Recall <-  cross.classify.table[2,2] / sum(cross.classify.table[2,])
  
  accuracy             = mean(model.pred  == data.partition$test$is_duplicate ) 
  error                = mean(model.pred  != data.partition$test$is_duplicate ) 
  pred <- ROCR::prediction(model.probs, data.partition$test$is_duplicate)
  perf <- ROCR::performance(pred, 'tpr', 'fpr')
  perf_df <- data.frame(perf@x.values, perf@y.values)
  names(perf_df) <- c("fpr", "tpr")
  
  list( roc_data=perf_df,
        accuracy = accuracy,
        error= error,
        pred.prob=model.probs,
        precision = precision,
        Recall = Recall,
        conf.table = cross.classify.table,
        model = model.fits,
        conf_matr = conf_matr)
  
}

build_ensemble_model <- function(data.partition, model.glm,model.lda,model.qda,model.rpart,model.naiveBayes,model.nnet,model.randomForest)
{
  
  
  model.probs <- (  model.nnet$pred.prob * model.nnet$accuracy+
                      model.randomForest$pred.prob * model.randomForest$accuracy) / (2 * model.randomForest$accuracy)
  
  
  # model.probs <- (  model.glm$pred.prob *model.glm$accuracy  +  model.nnet$pred.prob * model.nnet$accuracy+ 
  #                    model.randomForest$pred.prob * model.randomForest$accuracy) / (3 * model.randomForest$accuracy)
  # 
  
  
  # model.probs <- ( model.glm$pred.prob *model.glm$accuracy  + model.lda$pred.prob *model.lda$accuracy  + 
  #                    model.qda$pred.prob * model.qda$accuracy + model.rpart$pred.prob * model.rpart$accuracy + 
  #                model.naiveBayes$pred.prob * model.naiveBayes$accuracy   + model.nnet$pred.prob * model.nnet$accuracy+ 
  #                  model.randomForest$pred.prob * model.randomForest$accuracy) / (7 * model.randomForest$accuracy)
  model.pred = ifelse(model.probs > .5,1,0)
  
  cross.classify.table = table(model.pred ,data.partition$test$is_duplicate)
  
  conf_matr = confusionMatrix(model.pred ,data.partition$test$is_duplicate) 
  
  precision <- cross.classify.table[2,2] / sum(cross.classify.table[,2])
  # Recall
  Recall <-  cross.classify.table[2,2] / sum(cross.classify.table[2,])
  
  
  # model.pred = model.pred %>% as.numeric()
  cross.classify.table = table(model.pred ,data.partition$test$is_duplicate)
  accuracy             = mean(model.pred  == data.partition$test$is_duplicate ) 
  error                = mean(model.pred  != data.partition$test$is_duplicate ) 
  pred <- ROCR::prediction(model.probs, data.partition$test$is_duplicate)
  perf <- ROCR::performance(pred, 'tpr', 'fpr')
  perf_df <- data.frame(perf@x.values, perf@y.values)
  names(perf_df) <- c("fpr", "tpr")
  
  list( roc_data=perf_df,
        accuracy = accuracy,
        error= error,
        pred.prob=model.probs,
        precision = precision,
        Recall    = Recall,
        conf_matr = conf_matr )
  
}


ggplotConfusionMatrix <- function(m,model_){
  mytitle <- paste( "Confusion Matrix:" ,model_, ':',
                    "Accuracy", percent_format()(m$overall[1]),
                    "Kappa", percent_format()(m$overall[2])  
                    # "dksd1",percent_format()(m$overall[3]),
                    # "dksd2",percent_format()(m$overall[4]), 
                    # "dksd3",percent_format()(m$overall[5]), 
                    # "dksd4",percent_format()(m$overall[6])
                    )
  p <-
    ggplot(data = as.data.frame(m$table) ,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
    theme(legend.position = "none") +
    ggtitle(mytitle)
  return(p)
}

plot_conv_matrix_q1q2 <- function(OneRaw,word_vectors) {
  
  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question2"] %>% as.character()
  v = tibble(question1,question2)
  
  v_q1 <- v %>% select(question1)   %>% unnest_tokens(word,question1) %>% as.vector() %>% t()  
  v_q2 <- v %>% select(question2)   %>% unnest_tokens(word,question2) %>% as.vector() %>% t() 
  
  token_q1 <- v_q1  %>%  word_vectors[.,,drop=FALSE]
  token_q2 <- v_q2  %>%  word_vectors[.,,drop=FALSE]
  
  covMatrixQ1Q2 = token_q1 %>% sim2( y = token_q2, method = "cosine", norm = "l2")
  covMatrixQ1Q2_melt = melt(covMatrixQ1Q2)
  colnames(covMatrixQ1Q2_melt) = c('question1','question2','value')
  
  p <- ggplot(covMatrixQ1Q2_melt , aes(question1,question2)) + 
    geom_tile(aes(fill = value),colour = "white") + 
    # scale_fill_gradient(low = "white", high = "steelblue") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",  midpoint = 0, limit = c(-1.3,1.3), space = "Lab", name="Cosine Similarity")+
    
    theme_grey(base_size = 10) + labs(x = "question1",y = "question2") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0) ) +
    theme_minimal()+ 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                     size = 12, hjust = 1))+
    coord_fixed()
  # theme(legend.position = "none",strip.text.x= element_text(size = 6, angle = 90, hjust = 0, colour = "grey50"))
  p 
}

load_word_vectors_glove <- function()
{
  load('data/word_vectors_glove.RData')
  word_vectors_glove
}





get_features1 <- function(OneRaw )
{

  feature_number_of_words <- function(Question)
  {
    v = tibble(Question)
    token_q1 <- v  %>% select(Question)  %>% unnest_tokens(word,Question) %>% t()
    length(token_q1)
  }
  
  
  feature_delta_of_words  <- function(OneRaw)
  {
    question1 <- OneRaw["question1"] %>% as.character()
    question2 <- OneRaw["question2"]  %>% as.character()
    
    v = tibble(question1,question2)
    
    token_q1 <- v  %>% select(question1)  %>% unnest_tokens(word,question1) %>% t()
    token_q2 <- v  %>% select(question2)  %>% unnest_tokens(word,question2) %>% t()
    
    v1 <- setdiff(token_q1, token_q2 ) %>% length()
    v2 <- setdiff(token_q2, token_q1 ) %>% length()
    
    list(delta_q1_q2 = v1,
         delta_q2_q1 = v2)
  }
  feature_no_stop_words <- function(OneRaw,soptwords_en)
  {
    question1 <- OneRaw["question1"]%>% as.character() 
    question2 <- OneRaw["question2"] %>% as.character()
    
    v = tibble(question1,question2)
    token_q1 <- v  %>% select(question1)  %>% unnest_tokens(word,question1) %>% t()
    token_q2 <- v  %>% select(question2)  %>% unnest_tokens(word,question2) %>% t()
    
    v1 <- intersect(token_q1, soptwords_en ) %>% length()
    v2 <- intersect(token_q2, soptwords_en ) %>% length()
    
    list(question1_no_stop_words = v1, question2_no_stop_words = v2)
    
  }
  
  get_stop_words_en <- function()
  {
    require(tokenizers)
    stopwords("en")
  }  
  
  
  l1 <- OneRaw %>% feature_no_stop_words(get_stop_words_en())
  l2 <- OneRaw %>%  feature_delta_of_words()
  
  l3 <- list( question1_no_words = OneRaw["question1"]%>%  as.character() %>%  feature_number_of_words(),
       question2_no_words = OneRaw["question2"]%>%  as.character() %>%  feature_number_of_words()) 
 
  c(l1,l2,l3) %>% as.data.frame()
 
 
 }

# df[1,] %>% get_features1()

get_features2 <- function(OneRaw)
  {

  ################# traminr sequence of POS tags
  pos_token_annotator_model <- Maxent_POS_Tag_Annotator(language = "en",
                                                        probs = TRUE, model = system.file("models", "en-pos-perceptron.bin",
                                                                                          package = "openNLPmodels.en"))
  wordAnnotator     <- Maxent_Word_Token_Annotator(language = "en", probs = TRUE, model =NULL)
  sentAnnotator     <- Maxent_Sent_Token_Annotator(language = "en", probs = TRUE, model =NULL)
  
  pos_tag_annotator <- Maxent_POS_Tag_Annotator(language = "en", probs =TRUE, model =NULL)
  pos_tag_annotator <- Maxent_POS_Tag_Annotator(language = "en", probs =TRUE, model =NULL)
  
  s= " the of system is not ok"
  chunkAnnotator <- Maxent_Chunk_Annotator(language = "en", probs =        FALSE, model = NULL)
  annotate(s,chunkAnnotator,posTaggedSentence)
  annotated_sentence <- annotate(s,sentAnnotator)
  posTaggedSentence <- annotate(s, pos_tag_annotator, annotated_word)
  annotated_word<- annotate(s,wordAnnotator,annotated_sentence)
  
  # Get the senetence POS Tags -----------
  
  POS_tags_seq <- function(dataRaw)
  {
    posTaggedSentence <-  as.String(dataRaw) %>% annotate( pos_tag_annotator, annotated_word)
    posTaggedWords <- posTaggedSentence %>% subset(type == "word")
    posTaggedWords
    tags <- posTaggedWords$features %>% sapply( `[[`, "POS")
    
    tags %>% paste(collapse = "-")
  }
  
  
  
  POS_tags <- function(dataRaw)
  {
    posTaggedSentence <-  as.String(dataRaw) %>% annotate( pos_tag_annotator, annotated_word)
    posTaggedWords <- posTaggedSentence %>% subset(type == "word")
    posTaggedWords
    tags <- posTaggedWords$features %>% sapply( `[[`, "POS")
    table_tags <- table(tags) %>% as.data.frame() %>% t()
    colnames(table_tags) = table_tags[1,]
    row.names(table_tags) = NULL
    table_tags
  }
  
  
  # Vreate New structure with all fields of tags
  Create_data_frame_for_tags <- function(Nrow)
  {
    
    Vect_Tags = NLP::Penn_Treebank_POS_tags$entry[1:45]
    length(Vect_Tags)
    y <- data.frame(matrix(0,ncol = length(Vect_Tags), nrow = Nrow))
    colnames(y) = Vect_Tags
    y
    
  }
  
  
  # Get the tags and fill one line table of featurs 
  MAP_POS_tags_to_Data <- function(  Question_sentence )
  {
    tag_line = Create_data_frame_for_tags(1)
    question_tag      <- Question_sentence %>% POS_tags()
    question_columns <- question_tag %>% colnames()
    tag_colunms <- tag_line %>% colnames( )
    tag_diff <- setdiff(question_columns,tag_colunms)
    tag_diff
    question_columns <- setdiff(question_columns,tag_diff)
    question_columns  
    tag_line[1,question_columns] <- question_tag[2,question_columns] %>% as.character()
    return(tag_line)
  }
  
  question1_gramatical_entities <- OneRaw["question1"]%>%  as.character()%>%  MAP_POS_tags_to_Data() 
  colnames(question1_gramatical_entities) = paste('question1',colnames(question1_gramatical_entities),sep = "_" )
  
  question2_gramatical_entities <- OneRaw["question2"]%>%  as.character()%>%  MAP_POS_tags_to_Data() 
  colnames(question2_gramatical_entities) = paste('question2',colnames(question2_gramatical_entities),sep = "_" )
  
   cbind(question1_gramatical_entities,question2_gramatical_entities)
}

# df[1,] %>% get_features2()



get_change_ram <- function(){
  change_memory <- pryr::mem_change()
  change_memory <- change_memory  %>% round(3)
  change_memory
}
  

get_free_ram <- function(){

   used_memory <- pryr::mem_used()/1000000000 %>% round(2)
   used_memory <- used_memory  %>% round(3)
   used_memory
  # pryr::me
  
  
   # if(Sys.info()[["sysname"]] == "Windows"){
   # 
   #    x = system2("wmic", args =  "OS get FreePhysicalMemory /Value", stdout = TRUE)
   # 
   #    x %>% grepl("FreePhysicalMemory", .) %>%
   #     x[.] %>%
   #     gsub("FreePhysicalMemory=", "", ., fixed = TRUE)    %>%
   #     gsub("\r", "", ., fixed = TRUE) %>% as.integer( )
   # 
   #  } else {
   #     stop("Only supported on Windows OS")
   #   }
}


get_roc_data_for_keras_model <- function( model, model_name) {
  predction1 = list(question1_test,question2_test) %>% predict(model,.)
  pred <- ROCR::prediction(predction1, df$is_duplicate[test_idx])
  perf <- ROCR::performance(pred, 'tpr', 'fpr')
  perf_df <- data.frame(perf@x.values, perf@y.values,model_name = model_name)
  names(perf_df) <- c("fpr", "tpr","model_name")
  perf_df
}

get_roc_data_for_keras_model <- function( validation, model_name) {
  pred <- ROCR::prediction(validation$prediction, df$is_duplicate[validation$val_sample])
  perf <- ROCR::performance(pred, 'tpr', 'fpr')
  perf_df <- data.frame(perf@x.values, perf@y.values,model_name = model_name)
  names(perf_df) <- c("fpr", "tpr","model_name")
  perf_df
}


multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {

  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


 get_data_for_category_proportion <- function( categories, proprtions,Total_)
 {
  
   get_categories_proption_text <- function(vv)
   {
     paste(vv["numbers"]%>% as.character(),'(', vv["percentages"] %>% as.character()  ,'%)',sep = "")
   }
  
   labels_ <- proprtions %>% c(.,(1-.) ) %>% data.frame( categories = categories , percentages = 100* ., numbers = Total_ *.  ) %>% apply(MARGIN = 1,FUN =get_categories_proption_text ) 
   
   proprtions %>% c(.,(1-.) ) %>% data.frame( categories = categories , percentages = .,proportions = ., numbers = Total_ *. , labels =  labels_ )
   
 }
 
 
 plot_similarity_pair_variables <- function(Dtf,col_comb)
 {

   plot_similarity_pair <- function(Dtf,v1)
   {
     Dtf %>% 
       ggplot(aes(get(v1[1]), get(v1[2]),colour= is_duplicate))+
       xlab(v1[1]) + ylab(v1[2])+
       geom_point()
   }
    Dtf %>%  apply(MARGIN = 1,FUN = plot_similarity_pair,col_comb)
 }
 

 get_column_comb <- function(col_comb) 
 {
   col_comb = crossing(col_comb, col_comb) %>% filter(col_comb !=col_comb1 ) %>% as.matrix()
   col_comb
 }


plot_categories_proportion <- function(Data_,Title_)
{
  Max_ = 100
  Categories <- Data_$categories
  Percentage <- Data_$proportions * 100
  ## create data frame
  colour.df <- data.frame(Categories, Percentage)
  
  ## calculate percentage 
  colour.df$percentage = Max_ * Data_$proportions[1]  %>%  c((1-.),.)
  colour.df = colour.df$percentage %>% order() %>% rev( ) %>% colour.df[. , ]
  colour.df$ymax = colour.df$percentage %>% cumsum()
  colour.df$ymin = colour.df$ymax %>% head(n = -1) %>%  c(0, .)
  
  colour.df$label_data <-   Data_$labels
  
  ggplot(colour.df, aes(fill = Categories, ymax = ymax, ymin = ymin, xmax = Max_, xmin = 80)) +
    geom_rect(colour = "black") +
    coord_polar(theta = "y") + 
    xlim(c(0, 100)) +
    geom_label_repel(aes(label = label_data , x = Max_, y = (ymin + ymax)/2),inherit.aes = F, show.legend = F, size = 5)+
    theme(legend.title = element_text(colour = "black", size = 16, face = "bold"), 
          legend.text = element_text(colour = "black", size = 15), 
          panel.grid = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank(),
          axis.ticks = element_blank()) +
    ggplot2::annotate("text", x = 0, y = 0, size = 5, label = Title_ )
 }
# c("Test","Train") %>% get_data_for_category_proportion(0.5,158888) %>% plot_categories_proportion("Categories")



## Retrain deep learning models
FLAGS <- flags(
  flag_integer("vocab_size",  76027),
  flag_integer("max_len_padding", 20),
  flag_integer("embedding_size", 100),
  flag_numeric("regularization", 0.0001),
  flag_integer("seq_embedding_size", 512)
)




load_keras_question_data <- function()
{
  load('data/question1.RData')
  load('data/question2.RData')
  load('data/word_vectors_glove.RData')
  question1 = question1[,31:50]
  question2 = question2[,31:50]  
  list(question1 = question1, question2 = question2,word_vectors_glove = word_vectors_glove )
}


# rm(c(question1,question2))

# Classification Model Definition --------------------------------------------------------

model1 <- function(FLAGS) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  embedding1 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  embedding2 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  
  seq_emb <- layer_lstm(
    units = FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.12)
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.12)
  
  vector1 <- input1 %>% embedding1() %>%
    seq_gru1()
  vector2 <- input2 %>% embedding2() %>%
    seq_gru2()
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)      
  
  
  out <- layer_dot(list(vector1, vector2), axes = 1) %>% dense_100_relu %>% 
    layer_dropout( rate = 0.13)%>% 
    layer_dense(1, activation = "sigmoid")
  
  
  model <- keras_model(list(input1, input2), out)
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  
  model
}


model2 <- function(FLAGS) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  dense_020_relu1     <- layer_dense( units = 20, activation = 'relu')  
  dense_020_relu2     <- layer_dense( units = 20, activation = 'relu')  
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  
  embedding1 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  embedding2 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  
  seq_emb <- layer_lstm(
    units = FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru1 <- layer_gru(
    units =  100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.12)
  
  seq_gru2 <- layer_gru(
    units =  100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.12)
  
  
  vector1 <- input1 %>% embedding1() %>%
    seq_gru1() %>% 
    dense_020_relu1 %>% 
    layer_dropout( rate = 0.1)
  
  vector2 <- input2 %>% embedding2() %>%
    seq_gru2() %>% 
    dense_020_relu2 %>% 
    layer_dropout( rate = 0.1)
  
  
  out <- list(vector1, vector2) %>% layer_dot( axes = 1) %>%  
    layer_dense(1, activation = "sigmoid")
  
  model <- list(input1, input2) %>% keras_model( out)
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  
  model
}

model3 <- function(FLAGS) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  dense_020_relu1     <- layer_dense( units = 20, activation = 'relu')  
  dense_020_relu2     <- layer_dense( units = 20, activation = 'relu')  
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  
  embedding1 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  embedding2 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  
  seq_emb <- layer_lstm(
    units = FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru1 <- layer_gru(
    units =  100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.12)
  
  seq_gru2 <- layer_gru(
    units =  100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.12)
  
  
  vector1 <- input1 %>% 
    embedding1() %>%
    seq_gru1() %>% 
    dense_020_relu1 %>% 
    layer_dropout( rate = 0.13)
  
  vector2 <-input2 %>% 
    embedding2() %>%
    seq_gru2() %>% 
    dense_020_relu2 %>% 
    layer_dropout( rate = 0.13)
  
  
  out <- layer_concatenate(list(vector1, vector2)) %>%  
    layer_dense(1, activation = "sigmoid")
  
  model <- keras_model(list(input1, input2), out)
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  
  model
}


model4 <- function(FLAGS) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  
  embedding1 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  embedding2 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_emb <- layer_lstm(
    units = FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru <- layer_gru(
    units =  100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.15)
  
  vector1 <- input1 %>% embedding1() 
  vector2 <- input2 %>% embedding2() 
  
  
  out <- layer_dot(list(vector1, vector2),axes = 1) %>% seq_gru %>%  dense_100_relu %>% layer_dropout(rate = 0.12) %>% 
    layer_dense(1, activation = "sigmoid")
  
  model <- keras_model(list(input1, input2), out)
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  
  
  model
}

model5 <- function(FLAGS) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  
  embedding1 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  embedding2 <- layer_embedding(
    input_dim = FLAGS$vocab_size + 2, 
    output_dim = FLAGS$embedding_size, 
    input_length = FLAGS$max_len_padding, 
    embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  
  seq_emb <- layer_lstm(
    units = FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru <- layer_gru(
    units =  100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.15)
  
  vector1 <- input1 %>% embedding1() 
  vector2 <- input2 %>% embedding2() 
  
  
  out <- list(vector1, vector2) %>% layer_dot(axes = 1) %>% seq_gru %>%  
    layer_dense(1, activation = "sigmoid")
  
  model <- list(input1, input2) %>% keras_model( out)
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  
  
  model
}

model6 <- function(FLAGS,word_vectors_glove) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  drop_out_0.1       <- layer_dropout(rate = 0.1) 
  
  
  
  embedding1 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  embedding2 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  
  seq_emb1 <- layer_lstm(
    units = FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.15
  )
  
  seq_emb2 <- layer_lstm(
    units = FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.15
  )
  
  
  seq_gru <- layer_gru(
    units =  FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1 %>% embedding1() %>%
    seq_emb1()
  vector2 <- input2 %>% embedding2() %>%
    seq_emb2()
  
  out <- list(vector1, vector2) %>% layer_dot( axes = 1) %>%
    layer_dense(1, activation = "sigmoid")
  
  model <- list(input1, input2) %>% keras_model( out)
  
  
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
   # v1 = model %>% predict(list(question1_[1:2,],question2_[1:2,]))
  # v1 %>% dim
  model
}


model7 <- function(FLAGS,word_vectors_glove) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  drop_out_0.1       <- layer_dropout(rate = 0.1) 
  
  
  
  embedding1 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  
  embedding2 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  seq_emb <- layer_lstm(
    units = FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.1)
  seq_gru2 <- layer_gru(
    units =  FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.1)
  
  vector1 <- input1 %>% embedding1() %>%
    seq_gru1()
  vector2 <- input2 %>% embedding2() %>%
    seq_gru2()
  
  out <- list(vector1, vector2) %>% layer_concatenate() %>% dense_020_relu %>% drop_out_0.1 %>% 
    layer_dense(1, activation = "sigmoid")
  
  model <- list(input1, input2) %>% keras_model( out)
  
  
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
   # v1 = model %>% predict(list(question1_[1:2,],question2_[1:2,]))
  # v1 %>% dim
  model
}
model8 <- function(FLAGS,word_vectors_glove) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)   
  drop_out_0.1       <- layer_dropout(rate = 0.1) 
  
  
  
  embedding1 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  embedding2 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  seq_emb <- layer_lstm(
    units = FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru <- layer_gru(
    units =  FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1 %>%  embedding1() 
  vector2 <- input2 %>%  embedding2() 
  
  out <- list(vector1, vector2) %>% layer_dot( axes = 1) %>% seq_gru %>% 
    layer_dense(1, activation = "sigmoid")
  
  model <- keras_model(list(input1, input2), out)
  
  
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  # v1 = model %>% predict(list(question1_[1:2,],question2_[1:2,]))
  # v1 %>% dim
  model
}

model9 <- function(FLAGS,word_vectors_glove) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten()   
  drop_out_0.1       <- layer_dropout(rate = 0.3) 
  conv2d_out_0.1     <- layer_conv_1d(filters = 128, kernel_size =  4, activation='relu', padding='same', input_shape = c(100,100) ) 
  max_pooling_2d     <- layer_max_pooling_1d(pool_size = 2)
  
  
  embedding1 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  embedding2 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  seq_emb <- layer_lstm(
    units = FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru <- layer_gru(
    units =  FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1 %>% embedding1() 
  vector2 <- input2 %>% embedding2() 
  
  out <- list(vector1, vector2) %>% layer_dot( axes = 1) %>% conv2d_out_0.1 %>%  max_pooling_2d %>% drop_out_0.1 %>%  flatten_layer_ %>% 
    layer_dense(1, activation = "sigmoid")
  
  model <- keras_model(list(input1, input2), out)
  
  
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  # v1 = model %>% predict(list(question1_[1:2,],question2_[1:2,]))
  # v1 %>% dim
  model
}


model10 <- function(FLAGS,word_vectors_glove) {
  input1 <- layer_input(shape = c(FLAGS$max_len_padding))
  input2 <- layer_input(shape = c(FLAGS$max_len_padding))
  
  drop_out_0.1       <- layer_dropout(rate = 0.1) 
  dense_020_relu     <- layer_dense( units = 20, activation = 'relu')  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  
  flatten_layer_1     <- layer_flatten()   
  conv1d_128_1    <- layer_conv_1d(filters = 128, kernel_size =  4, activation='relu', padding='same', input_shape = 128 ) 
  max_pooling_1d1     <- layer_max_pooling_1d(pool_size = 2)
  
  flatten_layer_2     <- layer_flatten()   
  conv1d_128_2    <- layer_conv_1d(filters = 128, kernel_size =  4, activation='relu', padding='same', input_shape = 128 ) 
  max_pooling_1d2     <- layer_max_pooling_1d(pool_size = 2)
  
  
  
  embedding1 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  embedding2 <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  
  seq_emb <- layer_lstm(
    units = FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
  )
  
  seq_gru <- layer_gru(
    units =  FLAGS$embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1 %>% embedding1() %>% conv1d_128_1() %>% max_pooling_1d1 %>% flatten_layer_1 %>% layer_dropout(rate = 0.15)
  vector2 <- input2 %>% embedding2() %>% conv1d_128_2() %>% max_pooling_1d2 %>% flatten_layer_2 %>% layer_dropout(rate = 0.15)
  
  out <- list(vector1, vector2) %>% layer_concatenate( )  %>% 
    layer_dense(1, activation = "sigmoid")
  
  model <- keras_model(list(input1, input2), out)
  
  
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  # v1 = model %>% predict(list(question1_[1:2,],question2_[1:2,]))
  # v1 %>% dim
  model
}


# Model Fitting -----------------------------------------------------------

train_model <- function(model,question1_,question2_,Labels,ratio,batch_size,epochs)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(question1_), size = ratio*nrow(question1_))
  
  history <- model %>%
    keras:: fit(
      list(question1_[-val_sample,], question2_[-val_sample,]),
      Labels[-val_sample], 
      batch_size = batch_size, 
      epochs = epochs, 
      validation_data = list(
        list(question1_[val_sample,], question2_[val_sample,]), Labels[val_sample]
      ),
      callbacks = list(
        callback_early_stopping(patience = 5),
        callback_reduce_lr_on_plateau(patience = 3)
      )
    )  
  prediction_ = list(question1_[val_sample,], question2_[val_sample,]) %>% predict(model,.)
  
  
  list(history= history, model = model, validation = data.frame(prediction = prediction_ , val_sample = val_sample ))
  
  
}


# model_hist <- FLAGS %>% model7() %>% train_model(question1_[1:10,],question2_[1:10,],df$is_duplicate[1:10] %>% as.numeric(),0.2,1,epochs = 5)
# model_hist$model %>% save_model_hdf5('model10.hdf5',overwrite = TRUE,include_optimizer = TRUE)
# saveRDS(model_hist$history,file='histmodel10.rds')
# histmodel10 = readRDS('histmodel10.rds')
# plot(histmodel10)

