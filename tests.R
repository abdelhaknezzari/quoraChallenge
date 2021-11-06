library(NLP)
library(openNLP)
library(openNLPdata)
library(readr)
library(keras)
library(purrr)
library(openNLPmodels.en)
library(dplyr)
library(data.table)
library(tidytext)
library(tidyr)
library(tokenizers)
library(TraMineR)
library(tm)
library(text2vec)
library(qdap)
library(qdapRegex)
library(stringr)
library(qdap)
library(stringi)
library(caret)
library(SnowballC)
library(lsa)
library(Rtsne)
library(ggrepel)
library(plot3D)
library(textstem)
library(textclean)
library(corrplot)


rm(list = ls(all=TRUE) )

# Downloading Data --------------------------------------------------------
quora_data <- get_file(
  "quora_duplicate_questions.tsv",
  "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv" )


read_rdata_from_github <- function(file_name,githuburl)
{
  githubURL <- paste(githuburl,"?raw=true", sep="")
  download.file(githubURL,destfile =file_name ,mode = "w")
}


# Pre-processing ----------------------------------------------------------
df <- read_tsv(quora_data)

clean_data_set <- function(Df)
{  
   Df <- Df %>% mutate( question1 = (question1 %>% replace_contraction() %>% replace_symbol() %>% tolower()),
                        question2 = (question2 %>% replace_contraction() %>% replace_symbol() %>% tolower()))
  
   Df <- Df %>% mutate( question1 = (question1 %>% str_replace_all( "‘","") %>% str_replace_all( "’","") %>%  str_replace_all( "…","")),
                        question2 = (question2 %>% str_replace_all( "‘","") %>% str_replace_all( "’","") %>%  str_replace_all( "…","")))
   
   
   Df <- Df %>% mutate( question1 = ( question1 %>% rm_bracket(pattern = "all",trim=T,clean = T ) ),
                        question2 = ( question2 %>% rm_bracket(pattern = "all",trim=T,clean = T ) ) )
   
   Df
   
  
}
  
df <- clean_data_set(df)

display_questions <- function(Df,Index)
{
  print(Df[which(Df$id == Index),]$question1)
  print(Df[which(Df$id == Index),]$question2)
  print(Df[which(Df$id == Index),]$is_duplicate)
  
}

display_questions(df,8)


save(df, file = "df.RData")


# Initialize the anotators
s = "Pierre Vinken , 61 years old " %>% as.String

sentAnnotator <- Maxent_Sent_Token_Annotator(language = "en", probs =  TRUE, model =NULL)
annotated_sentence <- annotate(s,sentAnnotator)
wordAnnotator <- Maxent_Word_Token_Annotator(language = "en", probs = TRUE, model =NULL)
annotated_word<- annotate(s,wordAnnotator,annotated_sentence)

pos_token_annotator_model <- Maxent_POS_Tag_Annotator(language = "en",
                                                      probs = TRUE, model = system.file("models", "en-pos-perceptron.bin",
                                                      package = "openNLPmodels.en"))

pos_tag_annotator <- Maxent_POS_Tag_Annotator(language = "en", probs = TRUE, model =NULL)
posTaggedSentence <- annotate(s, pos_tag_annotator, annotated_word)
posTaggedWords <- subset(posTaggedSentence, type == "word")
tags <- sapply(posTaggedWords$features, `[[`, "POS")
tags
chunkAnnotator <- Maxent_Chunk_Annotator(language = "en", probs =FALSE, model = NULL)
annotate(s,chunkAnnotator,posTaggedSentence)

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

####################
vb <- df$question1[1:10] %>% lapply(FUN = MAP_POS_tags_to_Data) 
question1_featurs <- vb %>% rbindlist( fill = TRUE)
write.csv(question1_featurs,file= "question1_featurs.csv")
save(question1_featurs, file = "question1_featurs.RData")

vb <- df$question2 %>% lapply(FUN = MAP_POS_tags_to_Data) 
question2_featurs <- vb %>% rbindlist( fill = TRUE)
write.csv(question1_featurs,file= "question2_featurs.csv")
save(question2_featurs, file = "question2_featurs.RData")


##############


Get_tag_seq <- function(Data_)
{
  Q1_tags <- Data_ %>% select(question1)%>% as.matrix() %>% apply(MARGIN = 1,POS_tags_seq ) %>% as.matrix()
  Q2_tags <- Data_ %>% select(question2)%>% as.matrix() %>% apply(MARGIN = 1,POS_tags_seq ) %>% as.matrix()
  rbind(Q1_tags,Q2_tags) %>% seqdef()
}

#---------------------------

Tag_sequences_Q1Q2 <- df %>% Get_tag_seq()
Tag_sequences_Q1Q2 %>% class()

seqLLCS(Tag_sequences_Q1Q2[1,], Tag_sequences_Q1Q2[(nrow(df)+ 1),])
plot(Tag_sequences_Q1Q2)

save(Tag_sequences_Q1Q2, file = "Tag_sequences_Q1Q2.RData")
######################

library(cluster)
couts <- seqsubm(Tag_sequences_Q1Q2, method = "TRATE")
round(couts)
? seqdist
seqdist_Q1Q2 <- seqdist(Tag_sequences_Q1Q2, method = "OM", indel = 3, sm = couts)
clusterward <- agnes(seqdist_Q1Q2, diss = TRUE, method = "ward")
plot(clusterward, which.plots = 2)

cluster3 <- cutree(clusterward, k = 3)
cluster3 <- factor(cluster3, labels = c("Type 1", "Type 2", "Type 3"))
table(cluster3)
seqfplot(Tag_sequences_Q1Q2, group = cluster3, pbarw = T)
seqmtplot(Tag_sequences_Q1Q2, group = cluster3)



# traminr distances  ------------------------------

more_features <- function(OnRaw)
{
  tokenQ1 =   tokenize_words( OnRaw["question1"] ) %>% unlist()
  tokenQ2 =   tokenize_words( OnRaw["question2"] ) %>% unlist()
  
  v1 <- setdiff(tokenQ1, tokenQ2 )
  v2 <- setdiff(tokenQ2, tokenQ1 ) 
  
  idx = OnRaw["id"] %>% as.integer() + 1
  
  list(  id           = OnRaw["id"] , 
         is_duplicate = OnRaw["is_duplicate"] , 
         q1_q2 = length(v1),q2_q1 = length(v2), 
         q1_length =  length(tokenQ1), 
         q2_length =  length(tokenQ2),
         traminr_seqLLCS = seqLLCS(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]),
         traminr_seqLLCP = seqLLCP(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]),
         traminr_seqmpos = seqmpos(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]) )
  
}

library(tokenizers)
soptwords_en <-stopwords("en")
soptwords_en %>% length()

more_features2 <- function(OnRaw)
{

   documents <- OnRaw["question1"]  %>% 
     c() %>% 
     VectorSource() %>%
     Corpus() %>% 
     tm_map(content_transformer(tolower)) %>% 
     tm_map( removePunctuation) %>% 
     tm_map(removeWords, soptwords_en )
   tokenQ1 <- documents[1]$content %>% tokenize_words()%>% unlist()
   tokenQ1
   
   
   documents <- OnRaw["question2"]  %>% 
     c() %>% 
     VectorSource() %>%
     Corpus() %>% 
     tm_map(content_transformer(tolower)) %>% 
     tm_map( removePunctuation) %>% 
     tm_map(removeWords, soptwords_en)
   tokenQ2 <-  documents[1]$content %>% tokenize_words()%>% unlist()
 
   
   v1_no_stop <- setdiff(tokenQ1, tokenQ2 )
   v2_no_stop <- setdiff(tokenQ2, tokenQ1 ) 
   

   tokenQ1 <- OnRaw["question1"] %>% tokenize_words %>% unlist()
   tokenQ1
   
   tokenQ2 <- OnRaw["question2"] %>% tokenize_words %>% unlist()
   tokenQ2

  
   v1 <- setdiff(tokenQ1, tokenQ2 )
   v2 <- setdiff(tokenQ2, tokenQ1 ) 
  
   idx = OnRaw["id"] %>% as.integer() + 1
  
   list(  id         = OnRaw["id"]  %>% as.numeric() , 
         is_duplicate = OnRaw["is_duplicate"]  , 
         q1_q2 = length(v1),
         q2_q1 = length(v2), 
         q1_q2_no_stopwords = v1_no_stop %>% length(),
         q2_q1_no_stopwords = v2_no_stop %>% length(),
         q1_stopwords = which(tokenQ1 %in% soptwords_en) %>% length( ),
         q2_stopwords = which(tokenQ2 %in% soptwords_en) %>% length( ),
         q1_length =  length(tokenQ1), 
         q2_length =  length(tokenQ2),
         traminr_seqLLCS = seqLLCS(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]),
         traminr_seqLLCP = seqLLCP(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]),
         traminr_seqmpos = seqmpos(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]) )
  
}
# Test Getting More Feature

more_featurs <- df %>%   apply(MARGIN=1,FUN = more_features2) 
more_featurs <- more_featurs %>% rbindlist( fill = TRUE)

write.csv(more_featurs,file= "more_featurs.csv")
save(more_featurs,file= "more_featurs.RData")

load("more_featurs.RData")

head(more_featurs)


# Feature 1: number of words -- 
feature_number_of_words <- function(Question)
{
  v = tibble(Question)
  token_q1 <- v  %>% select(Question)  %>% unnest_tokens(word,Question) %>% t()
  length(token_q1)
}

df[1,]$question1 %>% feature_number_of_words()

question1_number_of_words = df$question1 %>% as.data.frame() %>%  apply(MARGIN = 1 , FUN = feature_number_of_words)
question1_number_of_words =question1_number_of_words   %>% as.data.frame()
colnames(question1_number_of_words) = c('question1_no_words')
save(question1_number_of_words,file='question1_number_of_words.RData')

question2_number_of_words = df$question2 %>% as.data.frame() %>%  apply(MARGIN = 1 , FUN = feature_number_of_words) 
question2_number_of_words = question2_number_of_words %>% as.data.frame()
colnames(question2_number_of_words) = c('question2_no_words')
save(question2_number_of_words,file='question2_number_of_words.RData')

load('question2_number_of_words.RData') 
load('question1_number_of_words.RData') 


# Feature 2: delta of words  ------- 
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

df[1,]  %>% feature_delta_of_words()
df[2,]  %>% feature_delta_of_words()
df[3,]  %>% feature_delta_of_words()

library(data.table)
question_delta_words = df %>%   apply( MARGIN = 1,feature_delta_of_words) %>% rbindlist(fill =  F)
save(question_delta_words,file='question_delta_words.RData')


# Feature 3: no of stop words  ------- 
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

df[1,] %>% feature_no_stop_words(soptwords_en)
question_no_stop_words = df %>% apply(MARGIN = 1,feature_no_stop_words,soptwords_en)%>% rbindlist(fill =  F)
save(question_no_stop_words,file='question_no_stop_words.RData')




#  Features 4: similarity distances -- 
calculate_delta_distance_min_max <- function(OneRaw,word_vectors)
{
  
  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question2"] %>% as.character()
  
  v = tibble(question1 = question1,question2 = question2)
  #v = tibble(question1,question2)
  
  Q1 <- v  %>% select(question1)  %>% unnest_tokens(word,question1)
  Q2 <- v  %>% select(question2)  %>% unnest_tokens(word,question2)
  
  token_q1 = Q1 %>% as.vector() %>% t() 
  token_q2 = Q2 %>% as.vector() %>% t() 
  
  token_q1_q2 <- setdiff(token_q1, token_q2 )
  v1 = token_q1_q2 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  v2 = token_q2    %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  covMatrixQ1Q2 = v1 %>% sim2( y = v2, method = "cosine", norm = "l2")
  # covMatrixQ1Q2 = ifelse(token_q2 %>% length() == 0 | token_q1_q2 %>% length() == 0 ,0,v1 %>% sim2( y = v2, method = "cosine", norm = "l2"))
  minimum_cov_distQ1Q2_l <- apply(covMatrixQ1Q2, 1, min)  %>% sum()
  maximum_cov_distQ1Q2_l <- apply(covMatrixQ1Q2, 1, max)  %>% sum()
  moyenne_cov_distQ1Q2_l <- apply(covMatrixQ1Q2, 1, mean) %>% sum()

  minimum_cov_distQ1Q2_c <- apply(covMatrixQ1Q2, 2, min)  %>% sum()
  maximum_cov_distQ1Q2_c <- apply(covMatrixQ1Q2, 2, max)  %>% sum()
  moyenne_cov_distQ1Q2_c <- apply(covMatrixQ1Q2, 2, mean) %>% sum()

    
  token_q2_q1 <- setdiff(token_q2, token_q1 )  
  v1 = token_q2_q1 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  v2 = token_q1    %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  # covMatrixQ2Q1 = ifelse(token_q1 %>% length() == 0 | token_q2_q1 %>% length() == 0 ,0,v1 %>% sim2( y = v2, method = "cosine", norm = "l2"))
  covMatrixQ2Q1 = v1 %>% sim2( y = v2, method = "cosine", norm = "l2")
  minimum_cov_distQ2Q1_l <- apply(covMatrixQ2Q1, 1, min)  %>% sum()
  maximum_cov_distQ2Q1_l <- apply(covMatrixQ2Q1, 1, max)  %>% sum()
  moyenne_cov_distQ2Q1_l <- apply(covMatrixQ2Q1, 1, mean) %>% sum()

  minimum_cov_distQ2Q1_c <- apply(covMatrixQ2Q1, 2, min)  %>% sum()
  maximum_cov_distQ2Q1_c <- apply(covMatrixQ2Q1, 2, max)  %>% sum()
  moyenne_cov_distQ2Q1_c <- apply(covMatrixQ2Q1, 2, mean) %>% sum()
  
  
    
  list(  q1_q2_cos_dist_min_l  = ifelse(minimum_cov_distQ1Q2_l %>% is.infinite(),0,minimum_cov_distQ1Q2_l), 
         q1_q2_cos_dist_max_l  = ifelse(maximum_cov_distQ1Q2_l %>% is.infinite(),0,maximum_cov_distQ1Q2_l),
         q1_q2_cos_dist_mean_l = ifelse(moyenne_cov_distQ1Q2_l %>% is.infinite()|moyenne_cov_distQ1Q2_l %>% is.nan,0,moyenne_cov_distQ1Q2_l),
         q2_q1_cos_dist_min_l  = ifelse(minimum_cov_distQ2Q1_l %>% is.infinite(),0,minimum_cov_distQ2Q1_l), 
         q2_q1_cos_dist_max_l  = ifelse(maximum_cov_distQ2Q1_l %>% is.infinite(),0,maximum_cov_distQ2Q1_l),
         q2_q1_cos_dist_mean_l = ifelse(moyenne_cov_distQ2Q1_l %>% is.infinite()|moyenne_cov_distQ2Q1_l %>% is.nan,0,moyenne_cov_distQ2Q1_l),
         
         q1_q2_cos_dist_min_c  = ifelse(minimum_cov_distQ1Q2_c %>% is.infinite() | minimum_cov_distQ1Q2_c %>% is.nan,0,minimum_cov_distQ1Q2_c), 
         q1_q2_cos_dist_max_c  = ifelse(maximum_cov_distQ1Q2_c %>% is.infinite() | maximum_cov_distQ1Q2_c %>% is.nan,0,maximum_cov_distQ1Q2_c),
         q1_q2_cos_dist_mean_c = ifelse(moyenne_cov_distQ1Q2_c %>% is.infinite() | moyenne_cov_distQ1Q2_c %>% is.nan,0,moyenne_cov_distQ1Q2_c),
         q2_q1_cos_dist_min_c  = ifelse(minimum_cov_distQ2Q1_c %>% is.infinite() | minimum_cov_distQ2Q1_c %>% is.nan,0,minimum_cov_distQ2Q1_c), 
         q2_q1_cos_dist_max_c  = ifelse(maximum_cov_distQ2Q1_c %>% is.infinite() | maximum_cov_distQ2Q1_c %>% is.nan,0,maximum_cov_distQ2Q1_c),
         q2_q1_cos_dist_mean_c = ifelse(moyenne_cov_distQ2Q1_c %>% is.infinite() | moyenne_cov_distQ2Q1_c %>% is.nan,0,moyenne_cov_distQ2Q1_c))
  
  
}
load('word_vectors_glove.RData')

OneRaw = df[1,] 
df[1,] %>% calculate_delta_distance_min_max(word_vectors_glove )
question_similarity_distances <- df %>% apply(MARGIN = 1,calculate_delta_distance_min_max,word_vectors_glove )%>% rbindlist(fill =  F)
save(question_similarity_distances,file='question_similarity_distances.RData')

question_similarity_distances %>% View()

# Feature 5: Gramatical entities numbers ----- 
question1_gramatical_entities_no <- df$question1 %>% lapply(FUN = MAP_POS_tags_to_Data) %>% rbindlist( fill = TRUE)
colnames(question1_gramatical_entities_no) = paste('question1',colnames(question1_gramatical_entities_no),sep = "_" )
save(question1_gramatical_entities_no,file='question1_gramatical_entities_no.RData')


question2_gramatical_entities_no <- df$question2 %>% lapply(FUN = MAP_POS_tags_to_Data) %>% rbindlist( fill = TRUE)
colnames(question2_gramatical_entities_no) = paste('question2',colnames(question2_gramatical_entities_no),sep = "_" )
save(question2_gramatical_entities_no,file='question2_gramatical_entities_no.RData')

# merge both and save
question_gramatical_entities_no = cbind(question1_gramatical_entities_no,question2_gramatical_entities_no)
question_gramatical_entities_no <- question_gramatical_entities_no %>% mutate_all(as.numeric)
save(question_gramatical_entities_no,file='question_gramatical_entities_no.RData')

# Feature 6: Gramatical entities cosine distance --- 
load('question_gramatical_entities_no.RData')
question_gramatical_entities_no %>% head() %>% View()

OneRaw = question_gramatical_entities_no[1,]
OneRaw
cosign_distance_pos_tags <- function(OneRaw)
{
  x = OneRaw[1:45] %>% as.numeric()
  y = OneRaw[46:(46+44)] %>% as.numeric()
  
  return(cosine(x ,y  ))
}
question_gramatical_entities_no[1,] %>% cosign_distance_pos_tags()
question_gramatical_entities_cosine <- question_gramatical_entities_no %>% apply(MARGIN = 1,FUN = cosign_distance_pos_tags) %>% as.data.frame()
colnames(question_gramatical_entities_cosine) = c('gramatical_entities_cosine')
save(question_gramatical_entities_cosine,file='question_gramatical_entities_cosine.RData')

# Feature 6: Gramatical entities PCA  ------
load('question2_gramatical_entities_no.RData')
load('question1_gramatical_entities_no.RData')

question1_gramatical_entities_no = question1_gramatical_entities_no %>% mutate_all(as.numeric)
question2_gramatical_entities_no = question2_gramatical_entities_no %>% mutate_all(as.numeric)
save(question2_gramatical_entities_no,file='question2_gramatical_entities_no.RData')
save(question1_gramatical_entities_no,file='question1_gramatical_entities_no.RData')


question1_gramatical_entities_pca <- question1_gramatical_entities_no %>% prcomp( scale. = F, center = F)
question2_gramatical_entities_pca <- question2_gramatical_entities_no %>% prcomp( scale. = F, center = F)
question1_gramatical_entities_pca <- question1_gramatical_entities_pca$x[,1:4]
question2_gramatical_entities_pca <- question2_gramatical_entities_pca$x[,1:4]
save(question1_gramatical_entities_pca,file='question1_gramatical_entities_pca.RData')
save(question2_gramatical_entities_pca,file='question2_gramatical_entities_pca.RData')

question_gramatical_entities_pca = cbind(question1_gramatical_entities_pca,question2_gramatical_entities_pca)
save(question_gramatical_entities_pca,file='question_gramatical_entities_pca.RData')


# 6 Features of Sequence distances using TramineR --------
questions_Tag_sequences_Q1Q2 <- df %>% Get_tag_seq()
questions_Tag_sequences_Q1Q2 = Tag_sequences_Q1Q2
save(questions_Tag_sequences_Q1Q2, file = "questions_Tag_sequences_Q1Q2.RData")

feature_gramatical_entity_sequence <- function(OneRaw,Tag_sequences_Q1Q2,df)
{
  idx = OneRaw["id"] %>% as.integer() + 1
  
  list(  traminr_seqLLCS = seqLLCS(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]),
         traminr_seqLLCP = seqLLCP(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]),
         traminr_seqmpos = seqmpos(Tag_sequences_Q1Q2[idx,], Tag_sequences_Q1Q2[(nrow(df)+ idx),]) )
}

df[1,] %>% feature_gramatical_entity_sequence(questions_Tag_sequences_Q1Q2,df)
questions_Tag_sequences_distances <- df %>% 
     apply(MARGIN = 1,feature_gramatical_entity_sequence,questions_Tag_sequences_Q1Q2,df ) %>% 
     rbindlist( fill = TRUE)
save(questions_Tag_sequences_distances, file= 'questions_Tag_sequences_distances.RData')



# Features 7: similiarity distances by: dtm, tfidf, lsa for cosine, jaccard -----



# Feature 8 : Relaxed Word Mover ----
load('word_vectors_glove.RData')
load('df.RData')

dtm1  = dtm_tfidf_lsa$dtm1
dtm2  = dtm_tfidf_lsa$dtm2

i = 2

get_Relaxed_Word_Mover_dist <- function(dtm1,dtm2,word_vectors_glove)
{
  start = 0
  end_   = 0
  nrow_df = nrow(dtm1 )
  pkg_size = 1000
  V = (nrow_df/pkg_size) %>% round() + 1
  
  rwmd_model = RWMD$new(word_vectors_glove,method = "cosine",progressbar = F )
  data_frame_  = data.frame()
  
  
  for (i in 1:V)
  {
    start = ( i - 1 ) * pkg_size + 1 
    end_ = ifelse( end_ >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))


    rwmd_dist_mat = dist2(dtm1[start:end_,],dtm2[start:end_,], method = rwmd_model, norm = 'none') %>%  as.matrix()
    
    rwmd_dist_colmean =  rwmd_dist_mat %>% 
      colMeans() 
    
    rwmd_dist_rowmean =  rwmd_dist_mat %>% 
      rowMeans() 
    
    rwmd_dist_diag =  rwmd_dist_mat %>% 
      diag() 
    
    
    data_frame_ = rbind(data_frame_,data.frame( rwmd_dist_colmean = rwmd_dist_colmean, 
                                                rwmd_dist_rowmean = rwmd_dist_rowmean,
                                                rwmd_dist_diag    = rwmd_dist_diag)  )
    
  }
  data_frame_
}


Relaxed_Word_Mover_dist = get_Relaxed_Word_Mover_dist(dtm_tfidf_lsa$dtm1,dtm_tfidf_lsa$dtm2,word_vectors_glove) 

save(Relaxed_Word_Mover_dist,file='Relaxed_Word_Mover_dist.RData')



# feature 9: Word sequences in sentences -----
load('question1.RData')
load('question2.RData')

question1_ = question1[,493:512]
question2_ = question2[,493:512]

Q1_tags <- question1_ %>% as.matrix()
Q2_tags <- question2_ %>% as.matrix()

seq_q12 = rbind(Q1_tags,Q2_tags) %>% seqdef()


seqLLCS(seq_q12[1,], seq_q12[(10000+1),])
seqLLCS(seq_q12[8,], seq_q12[(10000+8),])
load('df.RData')


feature_word_sequences <- function(OneRaw,word_sequences_Q1Q2)
{
  idx = OneRaw["id"] %>% as.integer() + 1
  lnrow = nrow(word_sequences_Q1Q2) / 2
  
  list(  traminr_seqLLCS = seqLLCS(word_sequences_Q1Q2[idx,], word_sequences_Q1Q2[lnrow+ idx,]),
         traminr_seqLLCP = seqLLCP(word_sequences_Q1Q2[idx,], word_sequences_Q1Q2[lnrow+ idx,]),
         traminr_seqmpos = seqmpos(word_sequences_Q1Q2[idx,], word_sequences_Q1Q2[lnrow+ idx,]))
}

df[13,] %>% feature_word_sequences(seq_q12)

questions_word_sequence_distances <- df %>% 
  apply(MARGIN = 1,feature_word_sequences,seq_q12 ) %>% 
  rbindlist( fill = TRUE)

save(questions_word_sequence_distances, file= 'questions_word_sequence_distances.RData')

load('questions_word_sequence_distances.RData')



# features 10: Diagonal clustering, proportion of similar words in pair questions ----

get_proportion_similar_word <- function(OneRaw,word_vectors_glove)
{
  
  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question2"] %>% as.character()
  
  v = tibble(question1,question2)
  
  token_q1 <- v  %>% dplyr::select(question1)  %>% unnest_tokens(word,question1)%>% as.vector() %>% t()  %>%  word_vectors_glove[.,,drop=FALSE]
  token_q2 <- v  %>% dplyr::select(question2)  %>% unnest_tokens(word,question2)%>% as.vector() %>% t() %>%   word_vectors_glove[.,,drop=FALSE]
  

  covMatrixQ1Q2 = token_q1 %>% sim2( y = token_q2, method = "cosine", norm = "l2")
  
  
  dim1 = covMatrixQ1Q2 %>% dim
  
  
  v1 <- covMatrixQ1Q2 %>% apply(MARGIN = 1,FUN = max)
  v2 <- covMatrixQ1Q2 %>% apply(MARGIN = 2,FUN = max)
  
  
  p1_10 = ifelse(dim1[1] == 0,0 , which( v1 >= 1  ) %>% length()/dim1[1])
  p2_10 = ifelse(dim1[2] == 0,0 , which( v2 >= 1  ) %>% length()/dim1[2])
  
  p1_9 =  ifelse(dim1[1] == 0,0 , which( v1 >.9  ) %>% length()/dim1[1])
  p2_9  = ifelse(dim1[2] == 0,0 , which( v2 >.9  ) %>% length()/dim1[2])
  
  p1_8 =  ifelse(dim1[1] == 0,0 , which( v1 >.8  ) %>% length()/dim1[1])
  p2_8  = ifelse(dim1[2] == 0,0 , which( v2 >.8  ) %>% length()/dim1[2])
  
  p1_7 =  ifelse(dim1[1] == 0,0 , which( v1 >.7  ) %>% length()/dim1[1])
  p2_7  = ifelse(dim1[2] == 0,0 , which( v2 >.7  ) %>% length()/dim1[2])
  
  p1_6 =  ifelse(dim1[1] == 0,0 , which( v1 >.6  ) %>% length()/dim1[1])
  p2_6  = ifelse(dim1[2] == 0,0 , which( v2 >.6  ) %>% length()/dim1[2])
  
  p1_5 =  ifelse(dim1[1] == 0,0 , which( v1 >.5  ) %>% length()/dim1[1])
  p2_5  = ifelse(dim1[2] == 0,0 , which( v2 >.5  ) %>% length()/dim1[2])
  
  p1_4 =  ifelse(dim1[1] == 0,0 , which( v1 >.4  ) %>% length()/dim1[1])
  p2_4  = ifelse(dim1[2] == 0,0 , which( v2 >.4  ) %>% length()/dim1[2])
  
  
  p1_4 <- ifelse(p1_4 %>% is.nan,0,p1_4)
  p2_4 <- ifelse(p2_4 %>% is.nan,0,p1_4)
  
  p1_5 <- ifelse(p1_5 %>% is.nan,0,p1_5)
  p2_5 <- ifelse(p2_5 %>% is.nan,0,p1_5)
  
  p1_6 <- ifelse(p1_6 %>% is.nan,0,p1_6)
  p2_6 <- ifelse(p2_6 %>% is.nan,0,p1_6)
  
  p1_7 <- ifelse(p1_7 %>% is.nan,0,p1_7)
  p2_7 <- ifelse(p2_7 %>% is.nan,0,p1_7)
  
  p1_8 <- ifelse(p1_8 %>% is.nan,0,p1_8)
  p2_8 <- ifelse(p2_8 %>% is.nan,0,p1_8)
  
  p1_9 <- ifelse(p1_9 %>% is.nan,0,p1_9)
  p2_9 <- ifelse(p2_9 %>% is.nan,0,p1_9)
  
  p1_10 <- ifelse(p1_10 %>% is.nan,0,p1_10)
  p2_10 <- ifelse(p2_10 %>% is.nan,0,p1_10)
  
  
  list(sim_words_on_line_proportion_10=p1_10,sim_words_on_column_proportion10=p2_10,
       sim_words_on_line_proportion_9=p1_9,sim_words_on_column_proportion9=p2_9,
       sim_words_on_line_proportion_8=p1_8,sim_words_on_column_proportion8=p2_8,
       sim_words_on_line_proportion_7=p1_7,sim_words_on_column_proportion7=p2_7,
       sim_words_on_line_proportion_6=p1_6,sim_words_on_column_proportion6=p2_6,
       sim_words_on_line_proportion_5=p1_5,sim_words_on_column_proportion5=p2_5,
       sim_words_on_line_proportion_4=p1_4,sim_words_on_column_proportion4=p2_4)
  
}



df[17,] %>% get_proportion_similar_word(word_vectors_glove)

similar_words_proprtion <- df %>% apply(MARGIN = 1,FUN=get_proportion_similar_word,word_vectors_glove) %>% 
  rbindlist(fill =  F)

# similar_words_proprtion <- similar_words_proprtion %>% mutate( sim_words_on_line_proportion = ifelse( sim_words_on_line_proportion %>% is.nan,0 ,sim_words_on_line_proportion ),
#                                                                sim_words_on_column_proportion = ifelse( sim_words_on_column_proportion %>% is.nan,0 ,sim_words_on_column_proportion ) )

save(similar_words_proprtion,file = 'similar_words_proprtion.RData')



df_more_features <- cbind(df_more_features, similar_words_proprtion)


# Build table of all extracted features -----

build_table_of_features <- function( no_cosine_distance_word_vec, no_cosine_distance_by_dtm,no_traminr_seq,no_gramatic,no_word_numbers,no_line_col_word_prop)
{

  load('df.RData')
  
  if (!no_cosine_distance_word_vec){
    load('question_similarity_distances.RData')
    df_more_features1 = data.frame(q2_q1_cos_dist_max_l = question_similarity_distances$q2_q1_cos_dist_max_l,
                                   q1_q2_cos_dist_max_c = question_similarity_distances$q1_q2_cos_dist_max_c,
                                   q1_q2_cos_dist_max_l = question_similarity_distances$q1_q2_cos_dist_max_l,
                               q2_q1_cos_dist_max_c = question_similarity_distances$q2_q1_cos_dist_max_c,
                               q2_q1_cos_dist_min_l = question_similarity_distances$q2_q1_cos_dist_min_l,
                               q1_q2_cos_dist_min_c = question_similarity_distances$q1_q2_cos_dist_min_c,
                               q1_q2_cos_dist_min_l = question_similarity_distances$q1_q2_cos_dist_min_l,
                               q2_q1_cos_dist_min_c = question_similarity_distances$q2_q1_cos_dist_min_c
                               # is_duplicate = factor(df$is_duplicate ,levels=c(0,1),labels=c('Not Duplicated','Duplicated')) 
                               )
 
      df_more_features= df_more_features1
  
   }
  if(!no_cosine_distance_by_dtm ) {
    load('df_simelarities.RData')
    load('Relaxed_Word_Mover_dist.RData')
  df_more_features1 = data.frame(tfidf_lsa_cos_sim   = df_simelarities$tfidf_lsa_cos_sim,
                                 tfidf_cos_sim       = df_simelarities$tfidf_cos_sim,
                                 tfidf_lsa_dist_m2   = df_simelarities$tfidf_lsa_dist_m2,
                                 tfidf_lsa_dist_m3   = df_simelarities$tfidf_lsa_dist_m3,
                                 
                                 d1_d2_cosine_sim    = df_simelarities$d1_d2_cosine_sim,
                                 rwmd_dist_colmean = Relaxed_Word_Mover_dist$rwmd_dist_colmean,
                                 rwmd_dist_rowmean = Relaxed_Word_Mover_dist$rwmd_dist_rowmean,
                                 rwmd_dist_diag    = Relaxed_Word_Mover_dist$rwmd_dist_diag
# #######################################

#                             d1_d2_jac_sim       = df_simelarities$d1_d2_jac_sim,
# tfidf_lsa_dist_m1   = df_simelarities$tfidf_lsa_dist_m1,
                                  # d1_d2_jac_psim      = df_simelarities$d1_d2_jac_psim,
                                  # d1_d2_tfidf_lsa_cos_psim2 = df_simelarities$d1_d2_tfidf_lsa_cos_psim2
                                 # is_duplicate = factor(df$is_duplicate ,levels=c(0,1),labels=c('Not Duplicated','Duplicated')) 
            )
    # df_more_features1 = df_more_features1 %>% mutate (d1_d2_jac_psim  = ifelse( d1_d2_jac_psim %>% is.na() , 0, d1_d2_jac_psim ) )
  
    if (no_cosine_distance_word_vec ) {df_more_features = df_more_features1 
    } else { df_more_features = cbind(df_more_features,df_more_features1) }
   
  }
  if(!no_traminr_seq)
  {
    load('questions_Tag_sequences_distances.RData')
    load('questions_word_sequence_distances.RData')
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
    load('question_gramatical_entities_pca.RData')
    load('question_gramatical_entities_cosine.RData')
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
    load('question1_number_of_words.RData')
    load('question2_number_of_words.RData')
    load('question_delta_words.RData')
    load('question_no_stop_words.RData')
    
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
    load('similar_words_proprtion.RData')
    df_more_features1 = data.frame(
      # sim_words_on_line_proportion_10   = similar_words_proprtion$sim_words_on_line_proportion_10,
      # sim_words_on_column_proportion10  = similar_words_proprtion$sim_words_on_column_proportion10,
      # sim_words_on_line_proportion_9    = similar_words_proprtion$sim_words_on_line_proportion_9,
      # sim_words_on_column_proportion9   = similar_words_proprtion$sim_words_on_column_proportion9,
      # sim_words_on_line_proportion_8    = similar_words_proprtion$sim_words_on_line_proportion_8,
      # sim_words_on_column_proportion8   = similar_words_proprtion$sim_words_on_column_proportion8,
      sim_words_on_line_proportion_7    = similar_words_proprtion$sim_words_on_line_proportion_7
      # sim_words_on_column_proportion7   = similar_words_proprtion$sim_words_on_column_proportion7
      # sim_words_on_line_proportion_6    = similar_words_proprtion$sim_words_on_line_proportion_6,
      # sim_words_on_column_proportion6   = similar_words_proprtion$sim_words_on_column_proportion6,
      # sim_words_on_line_proportion_5    = similar_words_proprtion$sim_words_on_line_proportion_5,
      # sim_words_on_column_proportion5   = similar_words_proprtion$sim_words_on_column_proportion5,
      # sim_words_on_line_proportion_4    = similar_words_proprtion$sim_words_on_line_proportion_4,
      # sim_words_on_column_proportion4   = similar_words_proprtion$sim_words_on_column_proportion4
    )
    
    if (no_cosine_distance_word_vec | no_cosine_distance_by_dtm | no_traminr_seq | no_word_numbers |no_line_col_word_prop ) {
      df_more_features = df_more_features1
    } else { df_more_features = cbind(df_more_features,df_more_features1) }
    
    
  }
  
  df_more_features = cbind(df_more_features, is_duplicate = df$is_duplicate %>% as.factor( ) ) 
  df_more_features
  }
df_more_features = build_table_of_features(F,F,F,F,F,F) %>% as.data.frame()
df_more_features %>% colnames %>% length()

save(df_more_features,file='df_more_features.RData')
rm(df_more_features)
# Machine Learnings ---- 
Model_Alg = 'glm'
get_model_train.predict_metrics <- function(data.partition,Model_Alg)
{
  require(class)
  require(MASS)
  require(randomForest)
  require(e1071)
  require(nnet)
  require(ROCR)
  require(xgboost)
  
  number_label_column = which( data.partition$train %>% colnames() == "is_duplicate")
  if (Model_Alg == 'glm')
  {
     model.fits = glm(is_duplicate~. , data = data.partition$train ,family = binomial(link = "logit")  )
  } else if(Model_Alg == 'lda') {
    model.fits = lda(is_duplicate~. , data = data.partition$train  )
    
  } else if(Model_Alg == 'qda') {
    
    model.fits = qda(is_duplicate~. , data = data.partition$train  ) 
  } else if(Model_Alg == 'knn') {
    model.fits = knn( train = data.partition$train , test = data.partition$test ,
                            cl = data.partition$train$is_duplicate, k=20)     
    
     
   } else if(Model_Alg == 'rpart')
   {
     model.fits = rpart(is_duplicate ~.  , data = data.partition$train )
     
   } else if(Model_Alg == 'randomForest')
   {
     model.fits = randomForest(is_duplicate ~. , data = data.partition$train, ntree = 201, mtry = 10)
   } else if(Model_Alg == 'naiveBayes'){
     model.fits <- naiveBayes(is_duplicate ~. , data = data.partition$train)
     
   } else if(Model_Alg == 'nnet') {
     # maxs <- apply(data.partition$train[,-28] , 2, max) 
     # mins <- apply(data.partition$train[,-28] , 2, min) 
     # train_data <- data.partition$train[,-28] %>% scale(center = mins, scale = maxs-mins) %>% as.data.frame()
     # train_data = cbind( train_data , is_duplicate = data.partition$train[,28] %>% as.numeric)
     # 
     # maxs <- apply(data.partition$test[,-28] , 2, max) 
     # mins <- apply(data.partition$test[,-28] , 2, min) 
     # test_data <- data.partition$test[,-28] %>% scale(center = mins, scale = maxs-mins) %>% as.data.frame()
     # test_data = cbind( test_data , is_duplicate = data.partition$test[,28] %>% as.numeric() )
     # 
     
     model.fits <- nnet(is_duplicate ~., data = data.partition$train, size = 25)
   } else if(Model_Alg == 'svm' )
   {
     model.fits <- svm(is_duplicate ~. , data = data.partition$train[1:10000,],kernel="linear", probability = T)
     
     
   } else if(Model_Alg == 'neuralnet'){
     library(neuralnet)
     collabel = 25
     maxs <- apply(data.partition$train[,-collabel] , 2, max) 
     mins <- apply(data.partition$train[,-collabel] , 2, min) 
     train_data <- data.partition$train[,-collabel] %>% scale(center = mins, scale = maxs-mins) %>% as.data.frame()
     train_data = cbind( train_data, is_duplicate = data.partition$train[,collabel] %>% as.numeric)
    
     maxs <- apply(data.partition$test[,-collabel] , 2, max) 
     mins <- apply(data.partition$test[,-collabel] , 2, min) 
     test_data <- data.partition$test[,-collabel] %>% scale(center = mins, scale = maxs-mins) %>% as.data.frame()
     test_data = cbind( test_data , is_duplicate = data.partition$test[,collabel] %>% as.numeric() )
     
     test_data[,-6 ] %>% colnames
     formula1 = test_data[,-collabel] %>% colnames %>%  paste0(collapse = ' + ') 
     formula1 = paste0('is_duplicate ~ ' ,formula1) %>%  as.formula()
     
     model.deepnn <- neuralnet(formula1 , data=train_data , hidden=c(50,50,50,50),act.fct = "logistic",
                               linear.output = FALSE)
   }  else if( Model_Alg == 'xgboost'){
     
     train_data_xgboost  = xgb.DMatrix(data = data.partition$train[,-number_label_column] %>% as.matrix(),
                                       label = data.partition$train[,number_label_column] %>% as.character() %>% as.numeric() )
     
     model.fits  <- xgboost(data = train_data_xgboost , max_depth =50,  nthread = 2, nrounds = 2, objective = "binary:logistic")
     
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
    model.probs = ifelse(model.probs == 2,1,0)
  } else if(Model_ALg == 'neuralnet' )
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


create_data_partition <- function(Data)
{
  set.seed(504023)  
  trainIndex <- createDataPartition(Data$is_duplicate, p = .8, list = FALSE, times = 1)
  df_more_features_train_ <- Data[ trainIndex,]
  df_more_features_test_  <- Data[-trainIndex,]  
  list( train = df_more_features_train_, test = df_more_features_test_)
}
# GLM / LDA/ QDA / NAIVE Bayes / Random Forest / nnet / Ensembel ----


data.partition <- create_data_partition(Data = df_more_features )



data.partition %>% get_model_train.predict_metrics('glm') -> roc.data.glm    
roc.data.lda   <- data.partition %>% get_model_train.predict_metrics('lda')
roc.data.qda   <- data.partition %>% get_model_train.predict_metrics('qda')
roc.data.rpart <- data.partition %>% get_model_train.predict_metrics('rpart')
roc.data.naivb <- data.partition %>% get_model_train.predict_metrics('naiveBayes')
roc.data.nnet  <- data.partition %>% get_model_train.predict_metrics('nnet')
roc.data.xgboost <- data.partition %>% get_model_train.predict_metrics('xgboost')
roc.data.rfrst <- data.partition %>% get_model_train.predict_metrics('randomForest')
roc.data.ensemble <- data.partition %>% build_ensemble_model(roc.data.glm,
                                                             roc.data.lda,
                                                             roc.data.qda,
                                                             roc.data.rpart,
                                                             roc.data.rpart,
                                                             roc.data.nnet,
                                                             roc.data.rfrst)

roc.data.all.models = cbind(roc.data.glm$roc_data, model = 'glm')
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.lda$roc_data, model = 'lda'))
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.qda$roc_data, model = 'qda'))
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.rpart$roc_data, model = 'rpart'))
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.naivb$roc_data, model = 'naiveBayes'))
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.nnet$roc_data, model = 'nnet'))
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.rfrst$roc_data, model = 'randomForest'))
roc.data.all.models = rbind(roc.data.all.models, cbind(roc.data.ensemble$roc_data, model = 'ensemble'))
 
roc.data.ensemble$accuracy


roc <- ggplot(data = roc.data.all.models, aes(x = fpr, y = tpr)) +
  geom_line(aes(color = model)) + geom_abline(intercept=0, slope=1, lty=3) + 
  ylab('True Positive Rate') + xlab('False Positive Rate')

roc

model.accuracy.error = cbind( accuracy = roc.data.lda$accuracy, error = roc.data.lda$error, model = 'lda')
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.qda$accuracy,error = roc.data.qda$error, model = 'qda'))
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.glm$accuracy,error = roc.data.glm$error, model = 'glm'))
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.rpart$accuracy,error= roc.data.rpart$error, model = 'rpart'))
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.naivb$accuracy,error= roc.data.naivb$error, model = 'naiveBayes'))
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.nnet$accuracy, error = roc.data.nnet$error,model = 'nnet'))
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.rfrst$accuracy,error = roc.data.rfrst$error ,model = 'randomForest'))
model.accuracy.error = rbind( model.accuracy.error , cbind(accuracy = roc.data.ensemble$accuracy,error = roc.data.ensemble$error ,model = 'ensemble'))


model.accuracy.error = model.accuracy.error %>% as.data.frame()

ggplot(data = model.accuracy.error , mapping = aes(x=model, y = accuracy  ) ) + geom_point(color = 'blue') 
ggplot(data = model.accuracy.error , mapping = aes(x=model, y = error  ) ) + geom_point(color = 'red') 


# K fold validation  -----
k_fold = 5
v = 1

get_ROC_cross_kfold_validation <- function(df_more_features,k_fold)
{
  model_metrics_kfold = data.frame()
  n <- nrow(df_more_features)
  Vf <- sample(n) %% k_fold + 1 
  for (v in 1:k_fold) {
    df.train <- df_more_features[Vf != v, ]
    df.test  <- df_more_features[Vf == v, ]
   
    
    model_metrics_cv_nt =  list(train = df.train, test = df.test) %>%  
      get_model_train.predict_metrics(Model_Alg = 'nnet')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_nt$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'nnet',
                                                               accuracy = model_metrics_cv_nt$accuracy,
                                                               error    = model_metrics_cv_nt$error,
                                                               Recall   = model_metrics_cv_nt$Recall,
                                                               precision= model_metrics_cv_nt$precision))
    
    
    model_metrics_cv_lm =  list(train = df.train, test = df.test) %>%  
                get_model_train.predict_metrics(Model_Alg = 'glm')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_lm$roc_data, 
                                                              fold = v, 
                                                              alg_model = 'glm',
                                                              accuracy = model_metrics_cv_lm$accuracy,
                                                              error    = model_metrics_cv_lm$error,
                                                              Recall   = model_metrics_cv_lm$Recall,
                                                              precision= model_metrics_cv_lm$precision))
    
    model_metrics_cv_ld =  list(train = df.train, test = df.test) %>%  
      get_model_train.predict_metrics(Model_Alg = 'lda')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_ld$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'lda',
                                                               accuracy = model_metrics_cv_ld$accuracy,
                                                               error    = model_metrics_cv_ld$error,
                                                               Recall   = model_metrics_cv_ld$Recall,
                                                               precision= model_metrics_cv_ld$precision))
    
    
    model_metrics_cv_qd =  list(train = df.train, test = df.test) %>%  
      get_model_train.predict_metrics(Model_Alg = 'qda')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_qd$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'qda',
                                                               accuracy = model_metrics_cv_qd$accuracy,
                                                               error    = model_metrics_cv_qd$error,
                                                               Recall   = model_metrics_cv_qd$Recall,
                                                               precision= model_metrics_cv_qd$precision))
    
    
    model_metrics_cv_rf =  list(train = df.train, test = df.test) %>%  
      get_model_train.predict_metrics(Model_Alg = 'randomForest')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_rf$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'randomForest',
                                                               accuracy = model_metrics_cv_rf$accuracy,
                                                               error    = model_metrics_cv_rf$error,
                                                               Recall   = model_metrics_cv_rf$Recall,
                                                               precision= model_metrics_cv_rf$precision))
    
    
    model_metrics_cv_rp =  list(train = df.train, test = df.test) %>%  
      get_model_train.predict_metrics(Model_Alg = 'rpart')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_rp$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'rpart',
                                                               accuracy = model_metrics_cv_rp$accuracy,
                                                               error    = model_metrics_cv_rp$error,
                                                               Recall   = model_metrics_cv_rp$Recall,
                                                               precision= model_metrics_cv_rp$precision))
    
    
    
    model_metrics_cv_nv =  list(train = df.train, test = df.test) %>%  
      get_model_train.predict_metrics(Model_Alg = 'naiveBayes')
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_nv$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'naiveBayes',
                                                               accuracy = model_metrics_cv_nv$accuracy,
                                                               error    = model_metrics_cv_nv$error,
                                                               Recall   = model_metrics_cv_nv$Recall,
                                                               precision= model_metrics_cv_nv$precision))
    
    model_metrics_cv_en = list(train = df.train, test = df.test) %>% 
      build_ensemble_model( model.glm = model_metrics_cv_lm,
                            model.lda = model_metrics_cv_ld,
                            model.qda = model_metrics_cv_qd,
                            model.rpart = model_metrics_cv_rp,
                            model.naiveBayes = model_metrics_cv_nv,
                            model.nnet = model_metrics_cv_nt,
                            model.randomForest = model_metrics_cv_rf)
    
    model_metrics_kfold = rbind(model_metrics_kfold,data.frame(roc_data = model_metrics_cv_en$roc_data, 
                                                               fold = v, 
                                                               alg_model = 'ensemble',
                                                               accuracy = model_metrics_cv_en$accuracy,
                                                               error    = model_metrics_cv_en$error,
                                                               Recall   = model_metrics_cv_en$Recall,
                                                               precision= model_metrics_cv_en$precision))
  
    
 }
  model_metrics_kfold

}

models_metrics_kfold <- df_more_features %>% get_ROC_cross_kfold_validation(5)


models_metrics_kfold_mean= models_metrics_kfold %>% 
  group_by(roc_data.fpr,alg_model) %>% 
   summarise( roc_data.tpr = mean(roc_data.tpr)) 
 

models_metrics_kfold_mean1= models_metrics_kfold %>% 
  group_by(alg_model) %>% 
  summarise( accuracy = mean(accuracy),
             error = mean(error),
             Recall = mean(Recall),
             precision = mean(precision) ) 


ggplot(data = models_metrics_kfold_mean, aes( x=roc_data.fpr, y= roc_data.tpr)) +
  geom_line(aes(color = alg_model)) + geom_abline(intercept=0, slope=1, lty=3) + 
  ylab('True Positive Rate') + xlab('False Positive Rate')
 
 # Confusion matrix plot ----
library(ggplot2)
library(scales)

ggplotConfusionMatrix <- function(m,model_){
  mytitle <- paste( model_, ':',
                   "Accuracy", percent_format()(m$overall[1]),
                   "Kappa", percent_format()(m$overall[2]))
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

p1 = ggplotConfusionMatrix(roc.data.glm$conf_matr,'glm')
p2 = ggplotConfusionMatrix(roc.data.lda$conf_matr,'lda')
p3 = ggplotConfusionMatrix(roc.data.qda$conf_matr,'qda')
p4 = ggplotConfusionMatrix(roc.data.rpart$conf_matr,'rpart')
p5 = ggplotConfusionMatrix(roc.data.naivb$conf_matr,'Naive Bayes')
p6 = ggplotConfusionMatrix(roc.data.nnet$conf_matr,'nnet')
p7 = ggplotConfusionMatrix(roc.data.ensemble$conf_matr,'ensemble')
p8 = ggplotConfusionMatrix(roc.data.rfrst$conf_matr,'Random forest')
multiplot(p1,p2,p3,p4,p5,p6,p7,p8,cols = 4)

# LDA ---pl


# plot some features -----

# 1. plot Number of words variables

data_plot_no_words <- cbind( question1_number_of_words,question2_number_of_words,question_delta_words,question_no_stop_words) %>% 
                      scale() %>% 
                      as.data.frame()

data_plot_no_words <- data_plot_no_words %>% mutate_all(as.numeric)
data_plot_no_words <- data_plot_no_words %>% gather( key = variable_name, value= no_of_words  )
data_plot_no_words <- cbind(data_plot_no_words,is_duplicate = df$is_duplicate)
data_plot_no_words %>% View()

ggplot(data_plot_no_words, aes(x=variable_name, y=no_of_words,colour = is_duplicate)) + 
  geom_boxplot() + ylab('Number of words Scaled') + xlab('Number of Words Variable')
ggplot(data_plot_no_words, aes(x=variable_name, y=no_of_words,colour = is_duplicate)) + 
   geom_violin() + ylab('Number of words Scaled') + xlab('Number of Words Variable')


col_comb <- c('question1_no_words','question2_no_words','delta_q1_q2','delta_q2_q1')
col_comb = crossing(col_comb, col_comb) %>% filter(col_comb !=col_comb1 ) %>% as.matrix()
col_comb

plot_similarity_pair_variables <- function(v1,df_more_features)
{
  cbind(df_more_features,is_duplicate=df$is_duplicate) %>% 
    ggplot(aes(get(v1[1]), get(v1[2]),colour= is_duplicate))+
    xlab(v1[1]) + ylab(v1[2])+
    geom_point()
}

plots_similariy_pairs <- col_comb %>%  apply(MARGIN = 1,FUN = plot_similarity_pair_variables,df_more_features)
plots_similariy_pairs %>% length
multiplot(plots_similariy_pairs[[1]],plots_similariy_pairs[[2]],
          plots_similariy_pairs[[3]],plots_similariy_pairs[[4]], 
          plots_similariy_pairs[[5]],plots_similariy_pairs[[6]],
          plots_similariy_pairs[[7]],plots_similariy_pairs[[8]],
          plots_similariy_pairs[[9]],plots_similariy_pairs[[10]], 
          plots_similariy_pairs[[11]],plots_similariy_pairs[[12]],
          cols = 4)


df_more_features %>% colnames 
plot_data_word_numbers = df_more_features[,-7] 
plot_data_word_numbers <- plot_data_word_numbers %>% gather( key = variable_name, value= number_of_words  )

plot_data_word_numbers <- cbind(plot_data_word_numbers,is_duplicate = df$is_duplicate)

plot_data_word_numbers %>% ggplot( aes(number_of_words,  colour=variable_name)) + 
  geom_density() + xlim(-1,30) +facet_grid(is_duplicate ~ .)

df_more_features[,-27]  %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
  corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)


# 2 . plot Gramatical Entities Cosine + PCA
load('question_gramatical_entities_pca.RData')
colnames(question_gramatical_entities_pca) = c('question1_PCA1','question1_PCA2','question1_PCA3','question1_PCA4','question2_PCA1','question2_PCA2','question2_PCA3','question2_PCA4')
save(question_gramatical_entities_pca,file='question_gramatical_entities_pca.RData')

load('question_gramatical_entities_cosine.RData')

data_plot_gramtical_entities <- cbind( question_gramatical_entities_pca, question_gramatical_entities_cosine) %>% 
  as.data.frame()

data_plot_gramtical_entities %>% View()

colnames(data_plot_gramtical_entities) = c('question1_PCA1','question1_PCA2','question1_PCA3','question1_PCA4','question2_PCA1','question2_PCA2','question2_PCA3','question2_PCA4','Cosine distance')
data_plot_gramtical_entities <- data_plot_gramtical_entities %>% mutate_all(as.numeric)

data_plot_gramtical_entities <- data_plot_gramtical_entities %>% gather( key = variable_name, value= values  )
data_plot_gramtical_entities <- cbind(data_plot_gramtical_entities,is_duplicate = df$is_duplicate)


ggplot(data_plot_gramtical_entities, aes(x=variable_name, y=values,colour = is_duplicate)) + 
  geom_boxplot() + ylab('Values') + xlab('Variable Name')


ggplot(data_plot_gramtical_entities, aes(x=variable_name, y=values,colour = is_duplicate)) + 
  geom_violin() + ylab('Values') + xlab('Variable Name')

data_plot_gramtical_entities <- cbind( question_gramatical_entities_pca, question_gramatical_entities_cosine,is_duplicate = df$is_duplicate)
splom(~data_plot_gramtical_entities)
data_plot_gramtical_entities %>% colnames()
colnames(data_plot_gramtical_entities) = c('question1_PCA1','question1_PCA2','question1_PCA3','question1_PCA4','question2_PCA1','question2_PCA2','question2_PCA3','question2_PCA4','Cosine distance','is_duplicate')


# 3. plot sequence of gramatical entities
data_plot_grama_entities_seq_dist =  questions_Tag_sequences_distances %>% 
  # scale() %>% 
  as.data.frame()

data_plot_grama_entities_seq_dist <- data_plot_grama_entities_seq_dist %>% gather( key = variable_name, value= distance_value  )
data_plot_grama_entities_seq_dist <- cbind(data_plot_grama_entities_seq_dist,is_duplicate = df$is_duplicate)
data_plot_grama_entities_seq_dist %>% View()
colnames(data_plot_grama_entities_seq_dist)
splom(~data_plot_grama_entities_seq_dist)

ggplot(data_plot_grama_entities_seq_dist, aes(x=variable_name, y=distance_value,colour = is_duplicate)) + 
  geom_boxplot() + ylab('Values') + xlab('Variable Name')

ggplot(data_plot_grama_entities_seq_dist, aes(x=variable_name, y=distance_value,colour = is_duplicate)) + 
  geom_violin() + ylab('Values') + xlab('Variable Name')

questions_Tag_sequences_distances = cbind(questions_Tag_sequences_distances,is_duplicate = df$is_duplicate)
questions_Tag_sequences_distances %>% pairs()
splom(~questions_Tag_sequences_distances)
load('df.RData')
data_plot_gramtical_entities = cbind(data_plot_gramtical_entities[,-c(2,3,4,6,7,8)],is_duplicate = df$is_duplicate)
splom(~ data_plot_gramtical_entities)


question_gramatical_entities_cosine %>% colnames()

col_comb <- c('traminr_seqLLCS','traminr_seqLLCP','traminr_seqmpos')
col_comb = crossing(col_comb, col_comb) %>% filter(col_comb !=col_comb1 ) %>% as.matrix()
col_comb

plot_similarity_pair_variables <- function(v1,df_more_features)
{
  cbind(df_more_features,is_duplicate=df$is_duplicate) %>% 
    ggplot(aes(get(v1[1]), get(v1[2]),colour= is_duplicate))+
    xlab(v1[1]) + ylab(v1[2])+
    geom_point()
}

plots_similariy_pairs <- col_comb %>%  apply(MARGIN = 1,FUN = plot_similarity_pair_variables,df_more_features)

multiplot(plots_similariy_pairs[[1]],plots_similariy_pairs[[2]],plots_similariy_pairs[[3]],plots_similariy_pairs[[4]], 
          plots_similariy_pairs[[5]],plots_similariy_pairs[[6]],cols = 4)



plot_data_tag_pos_sequences = df_more_features[,-4] 
plot_data_tag_pos_sequences <- plot_data_tag_pos_sequences %>% gather( key = variable_name, value= sequence_distance  )

plot_data_tag_pos_sequences <- cbind(plot_data_tag_pos_sequences,is_duplicate = df$is_duplicate)

plot_data_tag_pos_sequences %>% ggplot( aes(sequence_distance,  colour=variable_name)) + 
  geom_density() + xlim(-1,+7) +facet_grid(is_duplicate ~ .)



# Correlation between features tag sequences  ------
library(corrplot)
load('questions_Tag_sequences_distances.RData')
questions_Tag_sequences_distances %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
  corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)



scatter3D( df_more_features$traminr_seqLLCS ,questions_Tag_sequences_distances$traminr_seqLLCP, questions_Tag_sequences_distances$traminr_seqmpos,
          xlab = 'traminr_seqLLCS ', ylab = 'traminr_seqLLCP',   zlab = 'traminr_seqmpos',
          pch = 20,  theta = 20, phi = 20, colvar = df$is_duplicate %>% as.numeric(), type = 'p' )




# Correlation between features of gramatical entities ------
library(corrplot)
cbind(question_gramatical_entities_cosine,question_gramatical_entities_pca) %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
  corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)


# 4.  Plot similarity distances  ---- 
load('question_similarity_distances.RData')

colnames(question_similarity_distances)
plot_data_similarity_distances = cbind(question_similarity_distances,is_duplicate = df$is_duplicate)
splom(~ plot_data_similarity_distances)

# 5. plot densities density of similarity distances ---- 
plot_data_similarity_distances %>% View()
plot_data_similarity_distances = question_similarity_distances
plot_data_similarity_distances <- plot_data_similarity_distances %>% gather( key = variable_name, value= distance_value  )

plot_data_similarity_distances <- cbind(plot_data_similarity_distances,is_duplicate = df$is_duplicate)

ggplot(plot_data_similarity_distances, aes(distance_value,  colour=variable_name)) + 
 geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)

plot_data_similarity_distances = plot_data_similarity_distances %>% 
  mutate(is_duplicate =  as.factor(is_duplicate,labels = c("Not Duplicated","Duplicated")  ) )

plot_data_similarity_distances$is_duplicate = factor(plot_data_similarity_distances$is_duplicate,labels = c("Not Duplicated","Duplicated"))
plot_data_similarity_distances$is_duplicate %>% labels()

p1 <- plot_data_similarity_distances %>% filter(variable_name == "q1_q2_cos_dist_min_l"
                                          | variable_name == "q1_q2_cos_dist_max_l"
                                          | variable_name == "q1_q2_cos_dist_mean_l") %>% 
ggplot( aes(distance_value,  colour=variable_name)) + 
  xlab("Cosine Distance By Line")+
  geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)

p2 <- plot_data_similarity_distances %>% filter(variable_name == "q2_q1_cos_dist_min_l"
                                                | variable_name == "q2_q1_cos_dist_max_l"
                                                | variable_name == "q2_q1_cos_dist_mean_l") %>% 
  ggplot( aes(distance_value,  colour=variable_name)) + 
  xlab("Cosine Distance By Line")+
  geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)


p3 <- plot_data_similarity_distances %>% filter(variable_name == "q1_q2_cos_dist_min_c"
                                                | variable_name == "q1_q2_cos_dist_max_c"
                                                | variable_name == "q1_q2_cos_dist_mean_c") %>% 
  ggplot( aes(distance_value,  colour=variable_name)) + 
  xlab("Cosine Distance By Column")+
  geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)

p4 <- plot_data_similarity_distances %>% filter(variable_name == "q2_q1_cos_dist_min_c"
                                                | variable_name == "q2_q1_cos_dist_max_c"
                                                | variable_name == "q2_q1_cos_dist_mean_c") %>% 
  ggplot( aes(distance_value,  colour=variable_name)) + 
  xlab("Cosine Distance By Column")+
  geom_density() + xlim(-10,+30) +facet_grid(is_duplicate ~ .)

multiplot(p1,p2,p3,p4,cols=2)


col_comb <- c('q2_q1_cos_dist_min_c','q2_q1_cos_dist_min_l','q1_q2_cos_dist_min_c','q1_q2_cos_dist_min_l')
col_comb = crossing(col_comb, col_comb) %>% filter(col_comb !=col_comb1 ) %>% as.matrix()
col_comb



plot_similarity_pair_variables <- function(v1)
{
  cbind(question_similarity_distances,is_duplicate=df$is_duplicate) %>% 
    ggplot(aes(get(v1[1]), get(v1[2]),colour= is_duplicate))+
    xlab(v1[1]) + ylab(v1[2])+
    geom_point()
}

plots_similariy_pairs <- col_comb %>%  apply(MARGIN = 1,FUN = plot_similarity_pair_variables)

multiplot(plots_similariy_pairs[[1]],plots_similariy_pairs[[2]],plots_similariy_pairs[[3]],plots_similariy_pairs[[4]], 
          plots_similariy_pairs[[5]],plots_similariy_pairs[[6]],plots_similariy_pairs[[7]],
          plots_similariy_pairs[[8]],plots_similariy_pairs[[9]],plots_similariy_pairs[[10]],
          plots_similariy_pairs[[11]],plots_similariy_pairs[[12]],cols = 4)

# Correlation between features of cosine distances ----
library(corrplot)
df_more_features[,-32] %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)


df_more_features %>% length()

# plot similirities distances by tfidf,lsa,tdm,  for jaccard , cosine ----
load('df.RData')

plot_data_sim_dist_by_tfidf <- df_more_features[,-7] %>% gather( key = variable_name, value= distance_value  )
plot_data_sim_dist_by_tfidf <- cbind(plot_data_sim_dist_by_tfidf,is_duplicate = df$is_duplicate)
p1 <- plot_data_sim_dist_by_tfidf %>% 
  ggplot( aes(distance_value,  colour=variable_name)) + 
  xlab("Cosine Distance")+
  geom_density() + xlim(-.5,+3) +facet_grid(is_duplicate ~ .)

p1


df_more_features[,-c(7)] %>% colnames()

df_more_features[,-c(7)] %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
  corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)



col_comb <- df_more_features[,-7] %>% colnames()
col_comb = crossing(col_comb, col_comb ) %>% filter(col_comb !=col_comb1 ) %>% as.data.frame()
col_comb <- col_comb[!duplicated(t(apply(col_comb, 1, sort))),]

plot_similarity_pair_variables <- function(v1,df_more_features)
{
  cbind(df_more_features,is_duplicate=df$is_duplicate) %>% 
    ggplot(aes(get(v1[1]), get(v1[2]),colour= is_duplicate))+
    xlab(v1[1]) + ylab(v1[2])+
    geom_point()
}

plots_similariy_pairs <- col_comb %>%  apply(MARGIN = 1,FUN = plot_similarity_pair_variables,df_more_features)
plots_similariy_pairs %>% length
multiplot(plots_similariy_pairs[[1]],plots_similariy_pairs[[2]],plots_similariy_pairs[[2]],plots_similariy_pairs[[3]],plots_similariy_pairs[[4]],plots_similariy_pairs[[5]],
          plots_similariy_pairs[[6]],plots_similariy_pairs[[7]],plots_similariy_pairs[[8]],plots_similariy_pairs[[9]],plots_similariy_pairs[[10]],plots_similariy_pairs[[11]],
        plots_similariy_pairs[[12]],plots_similariy_pairs[[13]],plots_similariy_pairs[[14]],plots_similariy_pairs[[15]],          cols = 5)


df_more_features %>% colnames 
plot_data_features = df_more_features[,-10] 
plot_data_features <- plot_data_features %>% gather( key = variable_name, value= similarity_distance  )

plot_data_features <- cbind(plot_data_features,is_duplicate = df$is_duplicate)

plot_data_features %>% ggplot( aes(similarity_distance,  colour=variable_name)) + 
  geom_density() + xlim(-.5,5) +facet_grid(is_duplicate ~ .)

df_more_features[,-27]  %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
  corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)



# Plot word sequences -----

load('df.RData')

plot_data_word_sequneces_dist <- questions_word_sequence_distances%>% gather( key = variable_name, value= distance_value  )
plot_data_word_sequneces_dist <- cbind(plot_data_word_sequneces_dist,is_duplicate = df$is_duplicate)

p1 <- plot_data_word_sequneces_dist %>% 
  ggplot( aes(distance_value,  colour=variable_name)) + 
  xlab("Sequence Distance")+
  geom_density() + xlim(-3,25) +facet_grid(is_duplicate ~ .)

p1



df$is_duplicate = df$is_duplicate %>% as.factor()


questions_word_sequence_distances %>% cor(method = c("pearson", "kendall", "spearman")) %>% 
  corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)



col_comb <- questions_word_sequence_distances %>% colnames()
col_comb = crossing(col_comb, col_comb ) %>% filter(col_comb !=col_comb1 ) %>% as.data.frame()
col_comb <- col_comb[!duplicated(t(apply(col_comb, 1, sort))),]

plot_similarity_pair_variables <- function(v1,df_more_features)
{
  cbind(df_more_features,is_duplicate=df$is_duplicate) %>% 
    ggplot(aes(get(v1[1]), get(v1[2]),colour= is_duplicate))+
    xlab(v1[1]) + ylab(v1[2])+
    geom_point()
}

plots_similariy_pairs <- col_comb %>%  apply(MARGIN = 1,FUN = plot_similarity_pair_variables,questions_word_sequence_distances)
plots_similariy_pairs %>% length
multiplot(plots_similariy_pairs[[1]],plots_similariy_pairs[[2]],plots_similariy_pairs[[3]],cols = 2)

df_more_features = questions_word_sequence_distances



# 3 D for word sequences 

scatter3D( df_more_features$traminr_seqLLCS ,
           df_more_features$traminr_seqLLCP,
           df_more_features$traminr_seqmpos,
           xlab = 'traminr_seqLLCS',
           ylab = 'traminr_seqLLCP',
           zlab = 'traminr_seqmpos',
           pch = 20,  theta = 20, phi = 20,
           colvar = df_more_features$is_duplicate %>% as.numeric(),
           type = 'p'
           
)



library(rgl)


plot3d(df_more_features$traminr_seqLLCS, 
       df_more_features$traminr_seqLLCP,
       df_more_features$traminr_seqmpos, type="s", size=0.25, lit=FALSE,col = df_more_features$is_duplicate %>% as.numeric() )




# 3D plots ---- 
library(plot3D)

label_index <- function(df_)
{
   columns <- df_ %>% colnames()
   which(columns == 'is_duplicate') 
}




non_dupl_comb <- function(comb_column)
{
  colm1 = comb_column[1] %>% as.character()
  colm2 = comb_column[2] %>% as.character()
  colm3 = comb_column[3] %>% as.character()
  ifelse( colm1 == colm2 | colm1 == colm3 |colm2 == colm3,F,T) 
}


lv_index = df_more_features %>% label_index()
col_comb <- df_more_features %>% colnames()
col_comb <- col_comb[-lv_index]

col_comb = crossing(col_comb, col_comb ,col_comb) %>% filter(col_comb !=col_comb1  ) %>% as.data.frame()
colnames(col_comb) = c('c1','c2','c3')

col_comb = col_comb %>% apply(MARGIN = 1,FUN = non_dupl_comb ) %>% col_comb[.,]


column_comb = col_comb[10,]
calculate_plot_3d <- function(column_comb,df_more_features)
{

 scatter3D( df_more_features[,column_comb[1]],
            df_more_features[,column_comb[2]],
            df_more_features[,column_comb[3]],
            xlab = column_comb[1],
            ylab = column_comb[2],
            zlab = column_comb[3],
            pch = 20,  theta = 20, phi = 20,
            colvar = df_more_features$is_duplicate %>% as.numeric(),
            type = 'p'
           
            )

}
  




column_comb = col_comb[1,]


col_comb[1,] %>% apply(MARGIN = 1,FUN = calculate_plot_3d,df_more_features)

col_comb %>% dim

library(rgl)

lvidx = 9

plot3d(df_more_features[,col_comb[lvidx,1]], 
       df_more_features[,col_comb[lvidx,2]],
       df_more_features[,col_comb[lvidx,3]], type="s", size=0.25, lit=FALSE,col = df_more_features$is_duplicate %>% as.numeric() )

play3d(spin3d()) 




#########  Cleaning the data to optimize the memory

clear_all_data <- function(ListVars)
{ 
 rm(list = ListVars)

}

clear_all_data( ls(all=TRUE) )

load_all_data <- function()
{
  load("df.RData")
  load("Tag_sequences_Q1Q2.RData") 
  load("question1_featurs.RData")
  load("question2_featurs.RData")
  load("more_featurs.RData")
  load("word_vectors.RData")
  load("word_vectors_q1.RData")
  load("word_vectors_q2.RData")
}

load_all_data()



#################text2vec with glove
build_word_vector_qura_questions <- function()
{
   # Create iterator over tokens
   tokens <- rbind(df$question1 ,df$question2) %>% space_tokenizer()
   
   # Create vocabulary. Terms will be unigrams (simple words).
   it = tokens %>% itoken( progressbar = FALSE)
   
   vocab <- it %>% create_vocabulary( )

   # Use our filtered vocabulary
   vectorizer <- vocab %>% vocab_vectorizer( )
   
   # use window of 5 for context words
   tcm <- it %>% create_tcm( vectorizer, skip_grams_window = 2L)
   
   glove   <- vocab %>% GlobalVectors$new(word_vectors_size = 100, vocabulary = ., x_max = 15)
   wv_main <- tcm %>% glove$fit_transform(n_iter = 40)
   
   wv_context = glove$components
   word_vectors <- wv_main + t(wv_context)
   
   word_vectors
   
}

# tsne to see the word vector -----


word_vectors = word_vectors_glove


get_similar_words <- function( word ) {
  word_vec = word_vectors[word, , drop = FALSE]                           
  cos_sim = sim2(x = word_vectors, y = word_vec, method = "cosine", norm = "l2")
  word_nighbours_ <- head(sort(cos_sim[,1], decreasing = TRUE), 10)
  word_nighbours_ %>% names()
}
get_similar_words("family") 

get_some_word_of_interst <- function() {
  word_of_interst <- c("children","family","health","safety","happy")
  
  word_nighbours <- word_of_interst %>% lapply(FUN = get_similar_words)
  word_nighbours <- word_nighbours %>% unlist() %>%sort() %>%  unique()
  
  word_nighbours %>% unlist()
}                  

words_of_interst1 <- get_some_word_of_interst() %>% word_vectors[., , drop = FALSE]




tsne_word_of_interest <- function(words_of_interst1) {

  set.seed(9)  
  tsne_model_1 = Rtsne(as.matrix(words_of_interst1), check_duplicates=FALSE, pca=TRUE, perplexity=3, theta=0.5, dims=2)
  ## getting the two dimension matrix
  d_tsne_1 = data.frame(tsne_model_1$Y, term =row.names(words_of_interst1) ) 
  row.names(d_tsne_1) =  row.names(words_of_interst1)
    
  
  
  # ggplot(d_tsne_1, aes(x=X1, y=X2)) +  
  #    geom_point(size=0.25) +
  #    geom_text(aes(label = term))+
  #   guides(colour=guide_legend(override.aes=list(size=6))) +
  #   xlab("") + ylab("") +
  #   ggtitle("t-SNE") +
  #   theme_light(base_size=20) +
  #   theme(axis.text.x=element_blank(),
  #         axis.text.y=element_blank()) +
  #   scale_colour_brewer(palette = "Spectral")
  
  
  ggplot(d_tsne_1, aes(x= X1, y = X2)) + 
    geom_point(color = "blue", size = 1) + 
    geom_label_repel(aes(label = term),
                     segment.color = "grey50") +
    theme_classic()
}



get_some_word_of_interst() %>% word_vectors[., , drop = FALSE] %>% tsne_word_of_interest()


plot_word_vector_questions_vectors <- function(OnRaw)
{
   tokenQ1 =   tokenize_words( OnRaw["question1"] ) %>% unlist() %>% sort() %>%  unique()
   tokenQ2 =   tokenize_words( OnRaw["question2"] ) %>% unlist() %>% sort() %>%  unique()
  
   d1 <- setdiff(tokenQ1, tokenQ2 ) %>% sort() %>%  unique()
   d2 <- setdiff(tokenQ2, tokenQ1 )  %>% sort() %>%  unique()
  
   words_of_d1  <- d1 %>% word_vectors[., , drop = FALSE]
   words_of_q2  <- tokenQ2 %>% word_vectors[., , drop = FALSE]
   
   word_interest <- rbind(d1, tokenQ2)  %>% sort() %>%  unique()
   word_interest_v <- word_interest %>%  word_vectors[., , drop = FALSE]
   
   set.seed(50)  
   tsne_model_1 = word_interest_v %>%  as.matrix( ) %>%  Rtsne(. , check_duplicates=FALSE, pca=TRUE, perplexity=3, theta=0.5, dims=2)
   ## getting the two dimension matrix
   d_tsne_1 = data.frame(tsne_model_1$Y, term =row.names(word_interest_v) ) 
   row.names(d_tsne_1) =  row.names(word_interest_v)
   

   ggplot(d_tsne_1, aes(x= X1, y = X2)) + 
     geom_point(color = "blue", size = 1) + 
     geom_label_repel(aes(label = term),
                      segment.color = "grey50") +
     geom_segment(aes(x = 0, y = 0, xend = X1, yend = X2), data = d_tsne_1[tokenQ2,-3],arrow = arrow())+
     geom_segment(aes(x = 0, y = 0, xend = X1, yend = X2, color="red"), data = d_tsne_1[d1,-3],arrow = arrow())+
     geom_vline(xintercept = 0,color = "blue")+
     geom_hline(yintercept = 0,color = "blue")+
     theme_classic()  
   
}


df[2,] %>% plot_word_vector_questions_vectors()



#  Glove Word Vector (similarity matrixes ) -----  
build_word_vector_qura_question1 <- function()
{
  # Create iterator over tokens
  tokens <- space_tokenizer(df$question1)
  
  # Create vocabulary. Terms will be unigrams (simple words).
  it = itoken(tokens, progressbar = FALSE)
  
  vocab <- create_vocabulary(it)
 # vocab <- prune_vocabulary(vocab, term_count_min = 1L)
  
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = 3L)
  
  glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
  wv_main <- glove$fit_transform(tcm, n_iter = 20)
  
  wv_context = glove$components
  word_vectors <- wv_main + t(wv_context)
  
  word_vectors
  
  

}

build_word_vector_qura_question2 <- function()
{
  
  # Create iterator over tokens
  tokens <- space_tokenizer(df$question1)
  
  # Create vocabulary. Terms will be unigrams (simple words).
  it = itoken(tokens, progressbar = FALSE)
  
  vocab <- create_vocabulary(it)
 # vocab <- prune_vocabulary(vocab, term_count_min = 1L)
  
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = 1L)
  
  glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
  wv_main <- glove$fit_transform(tcm, n_iter = 20)
  
  wv_context = glove$components
  word_vectors <- wv_main + t(wv_context)
  
  word_vectors
  
}



build_word_vector_qura_question2 <- function()
{
  
  # Create iterator over tokens
  tokens <- space_tokenizer(df$question1)
  
  # Create vocabulary. Terms will be unigrams (simple words).
  it = itoken(tokens, progressbar = FALSE)
  
  vocab <- create_vocabulary(it)
  # vocab <- prune_vocabulary(vocab, term_count_min = 1L)
  
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = 1L)
  
  glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
  wv_main <- glove$fit_transform(tcm, n_iter = 20)
  
  wv_context = glove$components
  word_vectors <- wv_main + t(wv_context)
  
  word_vectors
  
}




covariance_matrix_Q1Q2 <- function(Data_Raw)
{
  Q1 <- Data_Raw %>% select(question1)  %>% unnest_tokens(word,question1)
  Q2 <- Data_Raw %>% select(question2)  %>% unnest_tokens(word,question2)
  
  v1 = Q1 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  v2 = Q2 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  
  covMatrixQ1Q2 = v1 %>% sim2( y = v2, method = "cosine", norm = "l2")
  covMatrixQ1Q2
  
}
  

covariance_matrix_Q1Q2_jaccard <- function(Data_Raw)
{
  Q1 <- Data_Raw %>% select(question1)  %>% unnest_tokens(word,question1)
  Q2 <- Data_Raw %>% select(question2)  %>% unnest_tokens(word,question2)
  
  v1 = Q1 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  v2 = Q2 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  
  covMatrixQ1Q2 = v1 %>% sim2( y = v2, method = "jaccard", norm = "l2")
  covMatrixQ1Q2
  
}



find_words_not_in_word_vec <- function(word_vectors)
{

  Q1 <- df %>% select(question1)  %>% unnest_tokens(word,question1)%>% as.vector() %>% t() %>% unique()
  Q2 <- df %>% select(question2)  %>% unnest_tokens(word,question2)%>% as.vector() %>% t() %>% unique() 
  
  r1 <-  which(! Q1 %in%( word_vectors %>% rownames()) )
  MissingInQ1 <- Q1[r1] %>% unique()

  r2 <- which(! Q2 %in%( word_vectors %>% rownames()) )
  MissingInQ2 <- Q2[r2] %>% unique()

    Missing_Q_ <- cbind(MissingInQ1 %>% t(),MissingInQ2 %>% t()) %>% t() %>% unique()
  Missing_Q_
  
}


extract_number_urls_nonascii <- function(df)
{

  df2 <- df %>% mutate( NonAsciiQ1 = (ex_non_ascii(question1)),
                         NonAsciiQ2 = (ex_non_ascii(question2)),
                         URLsQ1     = ex_url(question1 ),
                         URLsQ2     = ex_url(question2 ),
                         NumbQ1     = ex_number(question1),
                         NumbQ2     = ex_number(question2))
 
}


#   Cleaning functions -------------
clean_appostrophes <- function(Df)
{
  patterns     <- c("it's"   ,"he's"   ,"she's"  ,"i'm" ,"'re "   ) 
  replacements <- c("it is"  ,"he is"  ,"she is" ,"i am"," are "  ) 
 

  patterns    <- cbind(patterns     %>% t(),"won't"   ,"ain't"  ,"n't " )
  replacements<- cbind(replacements %>% t(),"will not","are not"," not ")
  
  
  patterns    <- cbind(patterns    ,"'ve "    , "'d "     , "'ll "     )
  replacements<- cbind(replacements," have "  , " would " , " will "   )  
  
  
  pattern_replace <- data.frame( patterns %>% t(), replacements %>% t())
  
  apply(X = pattern_replace,MARGIN = 1,FUN = function(x){ 
    
    Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = x[[1]],replacement = x[[2]]) ), 
                          question2 = ( question2 %>%  str_replace_all(pattern = x[[1]],replacement = x[[2]]) ) )    
    
    })
  
  Df
}




clean_non_ascii <- function(Df,Missing_words)
{

  patterns     <- c( "bigdata" , "onmyoji"  ,"heigou"  , "tarteist"  ,"delhi" ,"company" , "flatland" ,
                    "？","'" ,"“" ,"”" ,"？" ,"–" ,"—" , 
                    stri_unescape_unicode(paste0("\U2206")) , stri_unescape_unicode(paste0("\U222B")),
                    "flatland"  , "onmyoji"   , "∫"         ,"i̇stanbul", "i̇ve")
  replacements <- c("bigdata ", "onmyoji ","heigou " , "tarteist " ,"delhi ","company ", "flatland ",
                    " " ," " ," "," "  ,""   ," " ," " , 
                    "d"                                     ,"integral"                             ,
                    "flatland " , "onmyoji "   , "integral ","istanbul", "ive") 
  
  pattern_replace <- data.frame( patterns , replacements,stringsAsFactors = F )


  Df <- Df %>% mutate(  question1 = ( question1 %>%  rm_bracket(pattern = "round")  ), 
                        question2 = ( question2 %>%  rm_bracket(pattern = "round") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  rm_stopwords(c(".","?"),separate = F,strip = T)  ), 
                        question2 = ( question2 %>%  rm_stopwords(c(".","?") ,separate = F,strip = T) ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "'",replacement = " ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "'",replacement = " ") ) )   
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "“",replacement = " ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "“",replacement = " ") ) )   
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "”",replacement = " ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "”",replacement = " ") ) )   
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "—",replacement = " ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "—",replacement = " ") ) )   
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "–",replacement = " ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "–",replacement = " ") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "？",replacement = " ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "？",replacement = " ") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "flatland",replacement = "flatland ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "flatland",replacement = "flatland ") ) )
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "company",replacement = "company ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "company",replacement = "company ") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "delhi",replacement = "delhi ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "delhi",replacement = "delhi ") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "tarteist",replacement = "tarteist ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "tarteist",replacement = "tarteist ") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "heigou",replacement = " heigou ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "heigou",replacement = " heigou ") ) )
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "onmyoji",replacement = "onmyoji ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "onmyoji",replacement = "onmyoji ") ) )
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = "bigdata",replacement = "bigdata ") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = "bigdata",replacement = "bigdata ") ) )
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = stri_unescape_unicode(paste0("\U2206")),replacement = "d") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = stri_unescape_unicode(paste0("\U2206")),replacement = "d") ) )
  
  
  Df <- Df %>% mutate(  question1 = ( question1 %>%  str_replace_all(pattern = stri_unescape_unicode(paste0("\U222B")),replacement = "integral") ), 
                        question2 = ( question2 %>%  str_replace_all(pattern = stri_unescape_unicode(paste0("\U222B")),replacement = "integral") ) )
  
  Df
}


delta_number_words_q1q2 <- function(question1,question2)
{
  v = tibble(question1,question2)
  token_q1 <- v  %>% select(question1)  %>% unnest_tokens(word,question1) %>% as.vector() %>% t() 
  token_q2 <- v  %>% select(question2)  %>% unnest_tokens(word,question2) %>% as.vector() %>% t() 
  delta_words_q1_q2 <- setdiff(token_q1, token_q2 )
  delta_words_q2_q1 <- setdiff(token_q2, token_q1 )
  return(list(delta_words_q1_q2 = (delta_words_q1_q2 %>% length()),delta_words_q2_q1 = (delta_words_q2_q1%>% length())))
}
#######################>>>>>>>>>>>>>>>>>>>>

v = delta_number_words_q1q2(df[1,]$question1,df[1,]$question2)
v %>% print()

#######################<<<<<<<<<<<<<<<<<<<<



calculate_attribute_delta_number_words <- function(OneRaw)
{
  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question2"] %>% as.character()
  delta_words <- delta_number_words_q1q2(question1,question2)
  
  list(  id             = OneRaw["id"]  %>% as.numeric() , 
         is_duplicate   = OneRaw["is_duplicate"]  , 
         q1_q2_distance = delta_words[[1]],
         q2_q1_distance = delta_words[[2]] )
  
}

########################  Test >>>>>>>>>>>>>>>
question_feature_delta_words <- df %>% apply(MARGIN = 1,FUN = calculate_attribute_delta_number_words) %>% rbindlist( fill = TRUE)
save(question_feature_delta_words,file="question_feature_delta_words.RData")
########################  Test <<<<<<<<<<<<<<





calculate_delta_distance <- function(question1,question2)
{
  
  v = tibble(question1 = question1,question2 = question2)
  #v = tibble(question1,question2)
  
  Q1 <- v  %>% select(question1)  %>% unnest_tokens(word,question1)
  Q2 <- v  %>% select(question2)  %>% unnest_tokens(word,question2)
  
  token_q1 = Q1 %>% as.vector() %>% t() 
  token_q2 = Q2 %>% as.vector() %>% t() 
  
  token_q1_q2 <- setdiff(token_q1, token_q2 )
  
  v1 = token_q1_q2 %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  v2 = token_q2    %>% as.vector() %>% t() %>% word_vectors[.,,drop=FALSE]
  
  covMatrixQ1Q2 = v1 %>% sim2( y = v2, method = "cosine", norm = "l2")
  
  covMatrixQ1Q2 %>% as.matrix() %>% heatmap()
  
  minimum_cov_distQ1Q2 <- apply(covMatrixQ1Q2, 1, min) %>% sum()
  return(minimum_cov_distQ1Q2)

}

seq(-pi,+pi,0.01) %>% plot(.,cos(.), xlab = "angle",ylab="cosine distance" )
?plot

############  Test Function>>>>>>>>>>>>>>>
i = 8
df %>% display_questions(0)
calculate_delta_distance(df[i,]$question1,df[i,]$question2)
calculate_delta_distance(df[i,]$question2,df[i,]$question1)
############  Test <<<<<<<<<<<<<<<<<<<<<<

calculate_attribute_distance_q1q2 <- function(OneRaw)
{
  
  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question2"] %>% as.character()
  
  list(  id             = OneRaw["id"]  %>% as.numeric() , 
         is_duplicate   = OneRaw["is_duplicate"]  , 
         q1_q2_distance = calculate_delta_distance(question1,question2),
         q2_q1_distance = calculate_delta_distance(question2,question1 ) )
  
}

############  Test Function
df[i,] %>% calculate_attribute_distance_q1q2()
############  Test Function

question_feature_distances <- df %>% apply(MARGIN = 1,FUN = calculate_attribute_distance_q1q2) %>% rbindlist( fill = TRUE)
save(question_feature_distances,file="question_feature_distances.RData")

load("question_feature_distances.RData")


############## Merge features of cosine distance and delta of words

question_feature_delta_words %>% colnames()

colnames(question_feature_delta_words)   = c("id","is_duplicate","words_delta_q1_q2","words_delta_q2_q1")
feature_cosine_distance_word_delta <- cbind(question_feature_distances,question_feature_delta_words$words_delta_q1_q2,question_feature_delta_words$words_delta_q2_q1)
colnames(feature_cosine_distance_word_delta) = c("id","is_duplicate","q1_q2_cosine_distance","q2_q1_cosine_distance","words_delta_q1_q2","words_delta_q2_q1")


load("more_featurs.RData")
more_featurs %>% colnames()
more_featurs %>% View()


more_featurs %>% select(id,is_duplicate,traminr_seqmpos,traminr_seqLLCP) %>% View()

df_more_features <- question_feature_delta_words %>% inner_join( more_featurs )
save(df_more_features,file="df_more_features.RData")


# normalize the cosine distance 
display_questions(df,42)

feature_cosine_distance_word_delta <- feature_cosine_distance_word_delta %>% 
                               mutate( q1_q2_cosine_distance = ifelse(q1_q2_cosine_distance ==0,q1_q2_cosine_distance,q1_q2_cosine_distance/words_delta_q1_q2),
                                       q2_q1_cosine_distance = ifelse(q2_q1_cosine_distance ==0,q2_q1_cosine_distance,q2_q1_cosine_distance/words_delta_q2_q1) ) 

save(feature_cosine_distance_word_delta,file = "feature_cosine_distance_word_delta.RData")
load("feature_cosine_distance_word_delta.RData")
View(feature_cosine_distance_word_delta)


df_more_features <-cbind(df_more_features, question_feature_distances$q1_q2_distance,question_feature_distances$q2_q1_distance)
col1 <- df_more_features %>% colnames()
col1[16]  = "q1_q2_cosine_distance"
col1[17]  = "q2_q1_cosine_distance"
colnames(df_more_features) = col1

load("question_feature_delta_words.RData")

df_more_features <- cbind(df_more_features,question_feature_delta_words$q1_q2_distance,question_feature_delta_words$q2_q1_distance )
col1 <- df_more_features %>% colnames()
col1[18] = "q1_q2_distance"
col1[19] = "q2_q1_distance"

colnames(df_more_features) = col1

save(df_more_features, file = "df_more_features.RData")
rm(question_feature_delta_words)
rm(question_feature_distances)
rm(more_featurs)

load("df_more_features.RData")


df_more_features <- df_more_features %>% 
  mutate( q1_q2_cos_dist_norm  = ifelse(q1_q2_cosine_distance ==0,q1_q2_cosine_distance,q1_q2_cosine_distance/words_delta_q1_q2),
          q2_q1_cos_dist_norm  = ifelse(q2_q1_cosine_distance ==0,q2_q1_cosine_distance,q2_q1_cosine_distance/words_delta_q2_q1) ) 


#  Some Graphics for calculated features  ------------   
library(ggplot2)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
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




# Plots boxplot,histogram , density --------
ggplot(feature_cosine_distance_word_delta, aes(q1_q2_cosine_distance, q2_q1_cosine_distance,colour=is_duplicate)) +
  geom_point()

ggplot(feature_cosine_distance_word_delta, aes(words_delta_q1_q2, words_delta_q2_q1,colour=is_duplicate)) +
  geom_point()

stats_is_duplicate <-  feature_cosine_distance_word_delta %>% select(is_duplicate) %>% 
                         mutate()

# plot is_duplicate prportion
ggplot(feature_cosine_distance_word_delta, aes(is_duplicate, fill= is_duplicate) ) +
  geom_bar(stat="count")


ggplot(feature_cosine_distance_word_delta, aes( x=q1_q2_cosine_distance,colour = is_duplicate)) +
  geom_density() +
  facet_grid(is_duplicate ~ .)

ggplot(feature_cosine_distance_word_delta, aes( x=q2_q1_cosine_distance,colour = is_duplicate)) +
  geom_density() +
  facet_grid(is_duplicate ~ .)

ggplot(feature_cosine_distance_word_delta, aes( x=words_delta_q1_q2, colour = is_duplicate)) +
  geom_density() +
  facet_grid(is_duplicate ~ .)

ggplot(feature_cosine_distance_word_delta, aes( x=words_delta_q2_q1,colour = is_duplicate)) +
  geom_density() +
  facet_grid(is_duplicate ~ .)

v = which(feature_cosine_distance_word_delta$q1_q2_cosine_distance == 0 & feature_cosine_distance_word_delta$q2_q1_cosine_distance)
v %>% length()

v1 = which(df[v,]$is_duplicate == 0) 
v1[[1]]
display_questions(df,v1[[5]])


df_more_features$traminr_seqLLCS
df_more_features$traminr_seqLLCP
df_more_features$traminr_seqmpos

ggplot(df_more_features, aes(traminr_seqmpos, traminr_seqLLCP,colour=is_duplicate)) +
  geom_point()

ggplot(df_more_features, aes(traminr_seqLLCP, traminr_seqLLCS,colour=is_duplicate)) +
  geom_point()

ggplot(df_more_features, aes(traminr_seqmpos, traminr_seqLLCS,colour=is_duplicate)) +
  geom_point()


df_more_features$words_delta_q1_q2
ggplot(df_more_features, aes(words_delta_q1_q2, traminr_seqLLCS,colour=is_duplicate)) +
  geom_point()



load("df_more_features.RData")
ggplot(df_more_features, aes(x=is_duplicate, y=traminr_seqLLCS,colour = is_duplicate )) + geom_violin()
ggplot(df_more_features, aes(x=is_duplicate, y=traminr_seqLLCP,colour = is_duplicate )) + geom_violin()
ggplot(df_more_features, aes(x=is_duplicate, y=traminr_seqmpos,colour = is_duplicate )) + geom_violin()



df_more_features %>% colnames()

ggplot(df_more_features, aes(x=is_duplicate, y=q2_q1_distance,colour = is_duplicate)) + geom_violin()
ggplot(df_more_features, aes(x=is_duplicate, y=q1_q2_distance,colour = is_duplicate)) + geom_violin()

ggplot(df_more_features, aes(x=is_duplicate, y=q1_q2_cosine_distance,colour = is_duplicate)) + geom_violin()
ggplot(df_more_features, aes(x=is_duplicate, y=q2_q1_cosine_distance,colour = is_duplicate)) + geom_violin()


ggplot(df_more_features, aes(x=is_duplicate, y=q1_q2_cosine_distance,colour = is_duplicate)) + geom_boxplot()
ggplot(df_more_features, aes(x=is_duplicate, y=q2_q1_cosine_distance,colour = is_duplicate)) + geom_boxplot()

ggplot(df_more_features, aes(x=is_duplicate, y=q2_q1_distance,colour = is_duplicate)) + geom_boxplot()
ggplot(df_more_features, aes(x=is_duplicate, y=q1_q2_distance,colour = is_duplicate)) + geom_boxplot()


ggplot(df_more_features, aes(x=is_duplicate, y=traminr_seqLLCS,colour = is_duplicate)) + geom_boxplot()
ggplot(df_more_features, aes(x=is_duplicate, y=traminr_seqLLCP,colour = is_duplicate)) + geom_boxplot()
ggplot(df_more_features, aes(x=is_duplicate, y=traminr_seqmpos,colour = is_duplicate)) + geom_boxplot()


ggplot(df_more_features, aes(x=is_duplicate, y=q1_q2_cos_dist_norm,colour = is_duplicate )) + geom_boxplot()
ggplot(df_more_features, aes(x=is_duplicate, y=q2_q1_cos_dist_norm,colour = is_duplicate )) + geom_boxplot()

df_more_features$cosin_distance_bag_pos
ggplot(df_more_features, aes(x=is_duplicate, y=cosin_distance_bag_pos,colour = is_duplicate )) + geom_boxplot()
ggplot(df_more_features, aes(x=is_duplicate, y=cosin_distance_bag_pos,colour = is_duplicate)) + geom_violin()


############################################# Correlation between variables
library(corrplot)

fg <- df_more_features %>%  
  filter( !q2_q1_cos_dist_norm    %>% is.infinite() & 
          !q1_q2_cos_dist_norm    %>% is.infinite() &
          !q1_q2_cosine_distance  %>% is.infinite() &
          !q2_q1_cos_dist_norm    %>% is.infinite() & 
          !cosin_distance_bag_pos %>% is.infinite()   )


fg <- fg %>%  select(-is_duplicate) %>% cor(method = c("pearson", "kendall", "spearman"))
fg %>% corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)
################################################   Corelation between features





load("question1_featurs.RData")
load("question2_featurs.RData")
col1 = question1_featurs %>% colnames()
colnames(question1_featurs) = paste('q1_', col1,sep = "")

col2 = question2_featurs %>% colnames()
colnames(question2_featurs) = paste('q2_', col2,sep = "")


q1q2_features = cbind(df_more_features$id,question1_featurs,question2_featurs,df_more_features$is_duplicate)

col = q1q2_features %>% colnames()
col[1] = "id"
col[92] = "is_duplicate"

colnames(q1q2_features) = col

q1q2_features = q1q2_features %>% mutate_all(as.numeric)

cosign_distance_pos_tags <- function(OneRaw)
{
  x = OneRaw[2:46] %>% as.numeric()
  y = OneRaw[47:(47+44)] %>% as.numeric()
  return(cosine(x ,y  ))
}

q1q2_features[1,] %>% cosign_distance_pos_tags()

cosin_distance_bag_pos <- q1q2_features %>% apply(MARGIN = 1,FUN = cosign_distance_pos_tags)

q1q2_features <- cbind(q1q2_features,cosin_distance_bag_pos)

col <- q1q2_features %>% colnames()
col[93] = "cosin_distance_pos_tags"
colnames(q1q2_features) = col

q1q2_bags_pos_features =q1q2_features
save(q1q2_bags_pos_features,file='q1q2_bags_pos_features.RData')
load("q1q2_bags_pos_features.RData")

df_more_features <- cbind(df_more_features,cosin_distance_bag_pos)
save(df_more_features,file='df_more_features.RData')

load('df_more_features.RData')

q1q2_features <- q1q2_features %>% mutate(is_duplicate = is_duplicate %>% as.factor())
ggplot(q1q2_features, aes(x=is_duplicate, y=cosign_distance_vector )) + geom_boxplot()

ggplot(q1q2_features, aes(x=is_duplicate, y=cosign_distance_vector)) + geom_violin()


ggplot(q1q2_features, aes( x=cosign_distance_vector,colour =is_duplicate)) +
  geom_density() +
  facet_grid(is_duplicate ~ .)

q1q2_features$is_duplicate %>% class()

q1q2_features %>% View()


x = q1q2_features[8,2:46] %>% as.numeric()
y = q1q2_features[8,47:(47+44)] %>% as.numeric()
cosine(x ,y  )

f = data.frame(a=runif(5,0,1),b=runif(5,0,1))

f$a %>% norm(type = "2")
f$b %>% norm(type = "2")


f %>% mutate(c = (.[2,] %>% norm(type = "2")))
#q1q2_features = q1q2_features %>% mutate(cosine_distane = cosine([2:46] ,[47:(47+44)]  ) )



############################### Data cleaning


question_has_non_asccii <- function(question)
{
  non_ascii_q = question %>% ex_non_ascii() %>% unlist()
  
  logic_nonacii <- non_ascii_q %>% is_empty()
  if (!logic_nonacii )  logic_nonacii <- ! ( non_ascii_q %>% is.na() %>% mean() >0 )
  logic_nonacii
}

df[125136,]$question2 %>% question_has_non_asccii()
df[125136,]$question1 %>% question_has_non_asccii()
df[1,]$question2 %>% question_has_non_asccii()
df[1,]$question1 %>% question_has_non_asccii()


split_q <- function(question)
{
  v = tibble(question = question ) 
  Q <-   v %>% select(question)  %>% unnest_tokens(word,question)
  Q <- Q %>% as.vector()  %>% t() %>% paste(collapse = ' ')    
  Q
}

df[125136,]$question1 %>% split_q()
df[125136,]$question2 %>% split_q()


split_non_ascii <- function(question)
{
  ifelse( question %>% question_has_non_asccii(),split_q(question) ,question)
  
}

df[125136,]$question1 %>% split_non_ascii()
df[125136,]$question2 %>% split_non_ascii()

df[1,]$question1 %>% split_non_ascii()
df[1,]$question2 %>% split_non_ascii()



##########  Test function has_non_ascii_chars   -----------<<<<<<<

# Insert space between the non ASCII charachters like chinees , so the that later tokenization will split them correctely 
split_non_ascii_words <- function(OneRaw)
{
  #v = tibble(question1 = OneRaw["question1"],question2 = OneRaw["question2"] ) 
  
  list(  id              = OneRaw["id"]    %>% as.numeric() , 
         qid1            = OneRaw["qid1"]  %>% as.numeric() , 
         qid2            = OneRaw["qid2"]  %>% as.numeric() , 
         question1       = OneRaw["question1"] %>% split_non_ascii(),
         question2       = OneRaw["question2"] %>% split_non_ascii(),
         is_duplicate    = OneRaw["is_duplicate"] %>% as.character()  )
  
  
}


split_non_ascii_words_df <- function(Df)
{
  Df <- Df %>% apply(MARGIN = 1,FUN = split_non_ascii_words) %>% rbindlist( fill = TRUE)  
  Df
}
  
####### Test split_non_ascii_words -------->

df[125136:125137,]  %>% split_non_ascii_words()

df <- df %>% apply(MARGIN = 1,FUN = split_non_ascii_words) %>% rbindlist( fill = TRUE)

df[125136,]$question2
df[193837,]$question1

####### Test split_non_ascii_words <------------

### #   Example of exception handling in R
inputs = list(1, 2, 4, -5, 'oops', 0, 10)
for(input in inputs) {
       tryCatch(
                  print(paste("log of", input, "=", log(input))
                ),
             warning = function(w) {print(paste("negative argument"))},
             error   = function(e) {print(paste("non-numeric argument"))})
   }


# Main ------------------

test1 <- function()
{
  df <- read_tsv(quora_data)
  df <- clean_data_set(df)
  df <- clean_appostrophes(df)
  df <- split_non_ascii_words_df(df)
  df <- clean_non_ascii(df, Missing_Q)
  save(df,file="df.RData")
  
  load("df.RData")
  
  # URL Example
  df %>% display_questions(Index = 6549)
  
  # Chinees
  display_questions(df,403750)
  
  Missing_Q <- build_word_vector_qura_questions() %>% find_words_not_in_word_vec()

  View(Missing_Q)
  
  display_questions(df,8638)
  display_questions(df,2567)
 
  
}



#read_rdata_from_github("df_more_features.RData","https://github.com/abdelhaknezzari/mydata/blob/master/df_more_features.RData")  
#load("df_more_features.RData")





# GLM and other machine learnings ---------

save(df_more_features,file="df_more_features.RData")
load("df_more_features.RData")

df_more_features <- df_more_features %>% mutate( is_duplicate =  (is_duplicate %>% as.factor()) )

df_more_features= data.frame(q2_q1_cos_dist_max_l = question_similarity_distances$q2_q1_cos_dist_max_l,
                        q1_q2_cos_dist_max_c = question_similarity_distances$q1_q2_cos_dist_max_c,
                        q1_q2_cos_dist_max_l = question_similarity_distances$q1_q2_cos_dist_max_l,
                        q2_q1_cos_dist_max_c = question_similarity_distances$q2_q1_cos_dist_max_c,
                        q2_q1_cos_dist_min_l = question_similarity_distances$q2_q1_cos_dist_min_l,
                        q1_q2_cos_dist_min_c = question_similarity_distances$q1_q2_cos_dist_min_c,
                        q1_q2_cos_dist_min_l = question_similarity_distances$q1_q2_cos_dist_min_l,
                        q2_q1_cos_dist_min_c = question_similarity_distances$q2_q1_cos_dist_min_c,
                        is_duplicate = df$is_duplicate)

summary(df_more_features)
colnames(df_more_features)

# Filter the data to remove infinite

filter_data <- function(Data)
{
  
  Data <- Data %>% mutate( is_duplicate =  (is_duplicate %>% as.factor()) )
  #Data <- Data %>% select(-is_duplicate)
  Data <- Data %>% filter( !q1_q2_cosine_distance %>% is.infinite() & !q2_q1_cosine_distance %>% is.infinite()   )
  Data
  
}

create_data_partition <- function(Data)
{
  set.seed(504023)  
  trainIndex <- createDataPartition(Data$is_duplicate, p = .8, list = FALSE, times = 1)
  df_more_features_train_ <- Data[ trainIndex,]
  df_more_features_test_  <- Data[-trainIndex,]  
  list( train = df_more_features_train_, test = df_more_features_test_)
}

df_more_features_filtered = cbind(df_more_features,df$is_duplicate)
df_more_features_filtered = df_more_features
df_more_features_filtered = df_more_features_filtered[-c(13)]

df_more_features_filtered <- filter_data(df_more_features)
df_data_partition         <- create_data_partition(df_more_features_filtered)


get_glm_train_test_metrics <- function(df_data_partition)
{
  # Linear regression 
  glm.fits = glm(is_duplicate~. , data = df_data_partition$train ,family = binomial(link = "logit")  )
  glm.probs = predict (glm.fits,df_data_partition$test, type="response")
  n1 = df_data_partition$test %>% nrow()
  glm.pred=rep("0" ,n1)
  glm.pred[glm.probs > .5]="1"
  list(model.train          = glm.fits,
       model.predict        = glm.probs,
       cross.classify.table = table(glm.pred ,df_data_partition$test$is_duplicate),
       accuracy             = mean(glm.pred  == df_data_partition$test$is_duplicate ) ,
       error                = mean(glm.pred  != df_data_partition$test$is_duplicate ) )
}


get_roc_plot_data <- function(train_test_metrics,df_data_partition)
{
  
  s.glm <- seq(0,1.01,.01)
  
  absc.glm <- numeric(length(s.glm))
  ordo.glm <- numeric(length(s.glm))
  
  fit.glm <- glm(is_duplicate~.,data=df_data_partition$train,family=binomial)
  score.glm <- predict(fit.glm, df_data_partition$train, type = "response")
  for (i in 1:length(s.glm)){
    ordo.glm[i]=sum( score.glm >= s.glm[i] & df_data_partition$train$is_duplicate == "1")/sum(df_data_partition$train$is_duplicate == "1")
    absc.glm[i]=sum( score.glm >= s.glm[i] & df_data_partition$train$is_duplicate == "0")/sum(df_data_partition$train$is_duplicate == "0")
  }
  
  ROC = data.frame(FPR=absc.glm, TPR=ordo.glm)
  

  
  ROC
}

train_test_metrics = get_glm_train_test_metrics(df_data_partition)
ROC                = get_roc_plot_data(train_test_metrics, df_data_partition)

ggplot(ROC,aes(x=FPR,y=TPR)) + geom_path(color="red")  +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = 2, color = "black") 


get_cross_validation_error <- function(k_fold, df_data_partition  )
{
  n <- nrow(df_data_partition$train)
  
  Vf <- sample(n) %% k_fold + 1
  ErrCV <- numeric(k_fold)
  for (v in 1:k_fold) {
    model.train <- df_data_partition$train[Vf != v, ]
    model.test  <- df_data_partition$train[Vf == v, ]
    fit.glm     <- glm(is_duplicate~.,data=model.train,family=binomial(link="logit"))
    score.glm   <- predict(fit.glm, model.train, type = "response")
    pred.glm    <- as.numeric(score.glm >=0.5)
    ErrCV[v]    <- mean(pred.glm != df_data_partition$train$is_duplicate)
  }
  mean(ErrCV)
  
}

##################  Machine Learning: ROC, Train, Compare Model

library(caret)
library(pROC)


df_more_features_filtered$is_duplicate %>% levels()
df_more_features_filtered %>% summary()

df_more_features_filtered <- df_more_features_filtered %>% select(-id )
df_more_features_filtered %>% glimpse()
df_more_features_filtered %>% colnames()

ctrl <- trainControl(method="cv", 
                     number = 5,
                     # summaryFunction=twoClassSummary,
                     savePredictions = T
                     # classProbs=T
                     
                     )

rfFit <- caret::train( is_duplicate ~ . , data = df_more_features_filtered, 
               method="glm", 
               preProc=c("center", "scale"), 
               trControl=ctrl)


# Select a parameter setting
selectedIndices <- rfFit$pred$mtry == 2
# Plot:
plot.roc(rfFit$pred$obs[selectedIndices],
         rfFit$pred$pred[selectedIndices])

library(plotROC)
ggplot(rfFit$pred[selectedIndices, ], 
       aes(m = "0", d = factor(obs, levels = c("0", "1")))) + 
  geom_roc(hjust = -0.4, vjust = 1.5) + coord_equal()


rfFit$pred$obs
rfFit$pred$pred



#ROC


ErrCV = get_cross_validation_error(5,df_data_partition)
ErrCV


get_ROC_cross_validation <- function(k_fold,df_data_partition)
{
  s.glm <- seq(0,1.01,.01)
  absc.glm.CV=matrix(nrow = k_fold, ncol = length(s.glm))
  ordo.glm.CV=matrix(nrow = k_fold, ncol = length(s.glm))
  n <- nrow(df_data_partition$train)
  Vf <- sample(n) %% k_fold + 1 
  for (v in 1:k_fold) {
    df.train <- df_data_partition$train[Vf != v, ]
    df.test <- df_data_partition$train[Vf == v, ]
    fit.glm <- glm(is_duplicate~.,data=df.train,family=binomial)
    score.glm <- predict(fit.glm, df.test, type = "response")
    
    for (i in 1:length(s.glm)){
      ordo.glm.CV[v,i]=sum( score.glm >= s.glm[i] & df.test$type == "1")/sum(df.test$type == "1")
      absc.glm.CV[v,i]=sum( score.glm >= s.glm[i] & df.test$type == "0")/sum(df.test$type =="0")
    }
  }
  ordo.glm.CV <- apply(ordo.glm.CV,2,mean)
  absc.glm.CV <- apply(absc.glm.CV,2,mean)
  
  ROC.CV = data.frame(FPR=absc.glm.CV, TPR=ordo.glm.CV)
  ROC.CV
}


get_ROC_cross_validation2 <- function(k_fold,df_data_partition,Model.name)
{
  s.glm <- seq(0,1.01,.01)
  absc.glm.CV=matrix(nrow = k_fold, ncol = length(s.glm))
  ordo.glm.CV=matrix(nrow = k_fold, ncol = length(s.glm))
  n <- nrow(df_data_partition$train)
  Vf <- sample(n) %% k_fold + 1 
  for (v in 1:k_fold) {
    df.train <- df_data_partition$train[Vf != v, ]
    df.test  <- df_data_partition$train[Vf == v, ]
    fit.glm <- train(is_duplicate~.,data=df.train,method = Model.name)
    score.glm <- predict(fit.glm, df.test, type = "response")
    
    for (i in 1:length(s.glm)){
      ordo.glm.CV[v,i]=sum( score.glm >= s.glm[i] & df.test$is_duplicate == "1")/sum(df.test$is_duplicate == "1")
      absc.glm.CV[v,i]=sum( score.glm >= s.glm[i] & df.test$is_duplicate == "0")/sum(df.test$is_duplicate == "0")
    }
  }
  ordo.glm.CV <- apply(ordo.glm.CV,2,mean)
  absc.glm.CV <- apply(absc.glm.CV,2,mean)
  
  ROC.CV = data.frame(FPR=absc.glm.CV, TPR=ordo.glm.CV)
  ROC.CV
}

ROC.CV = get_ROC_cross_validation2(5,df_data_partition,"rf")

ROC.CV = get_ROC_cross_validation(5,df_data_partition)
ggplot(ROC.CV,aes(x=FPR,y=TPR)) + geom_path(color="red")  + geom_path(data = ROC, color = "blue") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = 2, color = "black")


get_ROC_cross_validation_any_model <- function(df_data_partition,Model.name,k_fold)
{
  s.glm <- seq(0,1.01,.01)
  
  absc.glm.CV=matrix(nrow = k_fold, ncol = length(s.glm))
  ordo.glm.CV=matrix(nrow = k_fold, ncol = length(s.glm))

    n <- nrow(df_data_partition$train)
  Vf <- sample(n) %% k_fold + 1 
  
  for (v in 1:k_fold) {
    df.train <- df_data_partition$train[Vf != v, ]
    df.test  <- df_data_partition$train[Vf == v, ]
    
    
    
    fit.model <- caret::train( is_duplicate ~ ., data = df.train ,method = Model.name  )
    score.glm <- caret::predict(fit.model, df.test)
    
    for (i in 1:length(s.glm)){
      ordo.glm.CV[v,i]=sum( score.glm >= s.glm[i] & df.test$type == "1")/sum(df.test$type == "1")
      absc.glm.CV[v,i]=sum( score.glm >= s.glm[i] & df.test$type == "0")/sum(df.test$type == "0")
    }
  }
  ordo.glm.CV <- apply(ordo.glm.CV,2,mean)
  absc.glm.CV <- apply(absc.glm.CV,2,mean)
  
  ROC.CV = data.frame(FPR=absc.glm.CV, TPR=ordo.glm.CV)
  ROC.CV
}

ROC.CV = df_data_partition %>% get_ROC_cross_validation_any_model("glm",5)
ggplot(ROC.CV,aes(x=FPR,y=TPR)) + geom_path(color="red")  + geom_path(data = ROC, color = "blue") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = 2, color = "black")


ROC.CV = df_data_partition %>% get_ROC_cross_validation_any_model("rf",5)
ggplot(ROC.CV,aes(x=FPR,y=TPR)) + geom_path(color="red")  + geom_path(data = ROC, color = "blue") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = 2, color = "black")

save(df_simelarities , file = "df_simelarities.RData")
ErrCaretAccuracy <- function(Errs)
{
  Errs <- Errs %>% group_by( model) %>% summarize( mAccuracy = mean(Accuracy, na.rm = TRUE), mKappa = mean(Kappa, na.rm = TRUE) ,
                                                   sAccuracy = sd(Accuracy, na.rm = TRUE)  , sKappa = sd(Kappa, na.rm = TRUE) )
  cbind( Errs )

}
ErrsCaret <- function(model, name)
{
  Errs <- data.frame(model$resample)
  Errs %>% mutate( model = name )
}

compare_learning_models <- function(Models.List,Data)
{

  trControl <- trainControl( method = "CV", number = 5  )
  Err.List <- NULL
  Errs.List <- c()
  for(Model.name in Models.List)
  {
    train.model <- train( is_duplicate ~ ., data = Data ,method = Model.name,trControl = trControl  )
    Err <- ErrsCaret(train.model,Model.name)  
    Err
    Err.line <- ErrCaretAccuracy(Err)
    Err.line
    Err.List <- rbind(Err.List,Err.line )
  }
  save(Err.List,file = "Err.List.RData")  
  Err.List
}

Models.List <- c("glm","nb","knn","svmLinear",  "svmRadial", "nnet", "treebag",  "rf", "C5.0", "xgbLinear", "xgbTree")
Err.List = compare_learning_models(Models.List, df_more_features_train_[1:1000,] )
ggplot(data = Err.List , mapping = aes(x=model, y = mAccuracy  ) ) + geom_point() 


# Multi ROCs plot with cross validation -----
library(partykit)
library(mosaic)
library(class)
library(randomForest)
library(e1071)
library(nnet)

plot_multi_rocs <- function(train,test) {
  model_glm <- glm(is_duplicate ~ ., data = train, family = binomial)
  model_tree <- rpart(form, data = train)
  model_RandomForest <- randomForest(form, data = train, ntree = 201, mtry = 3)
  # model_knn <- knn(train_q, test = train_q, cl = train$is_duplicate, k = 10)
  model_NaiveBayes    <- naiveBayes(form, data = train)
  model_NeuralNetwork  <- nnet(form, data = train, size = 5)
  
  
  get_roc <- function(x, y) {
    pred <- ROCR::prediction(x$y_hat, y)
    perf <- ROCR::performance(pred, 'tpr', 'fpr')
    perf_df <- data.frame(perf@x.values, perf@y.values)
    names(perf_df) <- c("fpr", "tpr")
    return(perf_df)
  }
  
  
  
  form <- as.formula("is_duplicate ~  traminr_seqLLCS + traminr_seqLLCP + traminr_seqmpos  + q1_q2_cos_dist_norm + q2_q1_cos_dist_norm")
  
  
  predictions_train <- data.frame(
    y = as.character(train$is_duplicate),
    type = "train",
    glm = predict(model_glm, type = "response"),
    # mod_tree = predict(mod_tree, type = "class"),
    RandomForest = predict(model_RandomForest, type = "class"),
    NeuralNetwork = predict(model_NeuralNetwork, type = "class"),
    NaiveBayes = predict(model_NaiveBayes, newdata = train, type = "class"))
  
  
  predictions_test <- data.frame(
    y = as.character(test$is_duplicate),
    type = "test",
    glm = predict(model_glm, newdata = test, type = "response"),
    # mod_tree = predict(mod_tree, newdata = test, type = "class"),
    RandomForest = predict(model_RandomForest, newdata = test, type = "class"),
    NeuralNetwork = predict(model_NeuralNetwork, newdata = test, type = "class"),
    NaiveBayes = predict(model_NaiveBayes, newdata = test, type = "class")
  )
  
  
  
  predictions <- bind_rows(predictions_train, predictions_test)
  
  predictions_tidy <- predictions %>%
    mutate(glm = ifelse(glm < 0.5, "0", "1")) %>%
    gather(key = "model", value = "y_hat", -type, -y)
  
  
  
  predictions_summary <- predictions_tidy %>%
    group_by(model, type) %>%
    summarize(N = n(), correct = sum(y == y_hat, 0),
              positives = sum(y == "1"),
              true_pos = sum(y_hat == "1" & y == y_hat),
              false_pos = sum(y_hat == "1" & y != y_hat)) %>%
    mutate(accuracy = correct / N,
           tpr = true_pos / positives,
           fpr = false_pos / (N - positives)) %>%
    ungroup() %>%
    gather(val_type, val, -model, -type) %>%
    unite(temp1, type, val_type, sep = "_") %>% # glue variables
    spread(temp1, val) %>%
    arrange(desc(test_accuracy)) %>%
    select(model, train_accuracy, test_accuracy, test_tpr, test_fpr)
  
  
  outputs <- c(  "response","prob"     ,"prob"             , "raw"               , "raw")
  models <- list(model_glm , model_tree,model_RandomForest ,  model_NeuralNetwork, model_NaiveBayes)
  
  roc_test <- mapply(predict, models, type = outputs, MoreArgs = list(newdata = test)) %>%
    as.data.frame() %>%
    select(1,3,5,6,8)
  
  
  
  names(roc_test) <-
    c(        "glm",  "tree"     ,"RandomForest"       , "NeuralNetwork"    , "NaiveBayes")
  
  
  roc_tidy <- roc_test %>%
    gather(key = "model", value = "y_hat") %>%
    group_by(model) %>%
    dplyr::do(get_roc(., y = test$is_duplicate))
  
  
  ggplot(data = roc_tidy, aes(x = fpr, y = tpr)) +
    geom_line(aes(color = model)) +
    geom_abline(intercept = 0, slope = 1, lty = 3) +
    ylab(perf@y.name) + xlab(perf@x.name) +
    geom_point(data = predictions_summary, size = 3,
               aes(x = test_fpr, y = test_tpr, color = model))
}



plot1 <- plot_multi_rocs(train,test)





############################  TFIDF #################
# https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html

library(stringr)
library(text2vec)

rm(list = ls(all=TRUE) )
load("df.RData")


data.vectorizer <- get_vocab_vectorizer(df)




rm(d1_d2_jac_sim1)
calc_similarities <- function(dtm)
{
  start = 0
  end   = 0
  nrow_df = nrow(dtm$df )
  pkg_size = 10000
  V = (nrow_df/pkg_size) %>% round() + 1
  
  d1_d2_jac_sim    = data.frame()
  d1_d2_cosine_sim = data.frame()
  
  d1_d2_sim = data.frame()
 
   for (i in 1:V)
  {
    start = ( i - 1 ) * pkg_size + 1 
    end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
    
    d1_d2_jac_sim1 = sim2(dtm$dtm1[start:end,], dtm$dtm2[start:end,], method = "jaccard", norm = "none") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    d1_d2_cosine_sim1 = sim2(dtm$dtm1[start:end,], dtm$dtm2[start:end,], method = "cosine", norm = "l2") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
  
    d1_d2_jac_sim    = rbind(d1_d2_jac_sim,d1_d2_jac_sim1)
    d1_d2_cosine_sim = rbind(d1_d2_cosine_sim,d1_d2_cosine_sim1)
    
    if( end >= nrow_df) break
    
  }
  d1_d2_jac_psim = psim2(dtm$dtm1, dtm$dtm2, method = "jaccard", norm = "none") %>% as.data.frame()
  data_frame_ = data.frame( d1_d2_jac_sim,  d1_d2_cosine_sim,d1_d2_jac_psim )
  colnames( data_frame_) =  c("d1_d2_jac_sim","d1_d2_cosine_sim","d1_d2_jac_psim")
  data_frame_
  
  }

d1_d2_sim <- df %>% get_dtm_() %>% calc_similarities()

sim2(dtm$dtm1[1:10,], dtm$dtm2[1:10,], method = "cosine", norm = "l2") %>% as.matrix() %>% heatmap()


dtm_tfidf  <- df %>% get_dtm_( ) %>% get_tfidf_()

dtm_tfidf$dtm1_tfidf %>% dim()
dtm_tfidf$dtm1_tfidf %>% dim()


sim2(x = dtm_tfidf$dtm1_tfidf[1:10,],y = dtm_tfidf$dtm2_tfidf[1:10,] , method = "cosine", norm = "l2") %>% 
  as.matrix() %>% 
  heatmap()


calc_tfidf_cos_similarities <- function(dtm_tfidf)
{
  start = 0
  end   = 0
  nrow_df = nrow(dtm_tfidf$df )
  pkg_size = 10000
  V = (nrow_df/pkg_size) %>% round() + 1
  d1_d2_tfidf_cos_sim = data.frame()
  for (i in 1:V)
  {
    start = ( i - 1 ) * pkg_size + 1 
    end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
    
    d1_d2_tfidf_cos_sim1 = sim2(dtm_tfidf$dtm1_tfidf[start:end,], dtm_tfidf$dtm2_tfidf[start:end,], method = "cosine", norm = "l2") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    d1_d2_tfidf_cos_sim = rbind(d1_d2_tfidf_cos_sim,d1_d2_tfidf_cos_sim1)
    
    if( end >= nrow_df) break
    
  }
  d1_d2_tfidf_cos_sim
}

d1_d2_tfidf_cos_sim = df %>% get_dtm_( ) %>% get_tfidf_() %>%  calc_tfidf_cos_similarities( )






dtm_tfidf_lsa <- df %>% get_dtm_( ) %>% get_tfidf_() %>% get_lsa_() 


sim2(dtm_tfidf_lsa$dtm1_tfidf_lsa[1:10,], dtm_tfidf_lsa$dtm2_tfidf_lsa[1:10,], method = "cosine", norm = "l2") %>% as.matrix() %>% 
heatmap()

calc_tfidf_lsa_cos_similarities <- function(df,dtm_tfidf_lsa)
{
  start = 0
  end   = 0
  nrow_df = nrow(df )
  pkg_size = 10000
  V = (nrow_df/pkg_size) %>% round() + 1
  d1_d2_tfidf_lsa_cos_sim = data.frame()
  for (i in 1:V)
  {
    start = ( i - 1 ) * pkg_size + 1 
    end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
    
    d1_d2_tfidf_cos_sim1 = sim2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "cosine", norm = "l2") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    d1_d2_tfidf_lsa_cos_sim = rbind(d1_d2_tfidf_lsa_cos_sim,d1_d2_tfidf_cos_sim1)
    
    if( end >= nrow_df) break
    
  }
  d1_d2_tfidf_lsa_cos_sim
}

d1_d2_tfidf_lsa_cos_sim = dtm_tfidf_lsa %>% calc_tfidf_lsa_cos_similarities(df,.)

dtm_tfidf_lsa$dtm1_tfidf_lsa %>% dim()
dtm_tfidf_lsa$dtm2_tfidf_lsa %>% dim()



d1_d2_tfidf_lsa_cos_psim2 = psim2(dtm_tfidf_lsa$dtm1_tfidf_lsa, dtm_tfidf_lsa$dtm2_tfidf_lsa, method = "cosine", norm = "l2") %>% as.data.frame()



dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[1:10,], dtm_tfidf_lsa$dtm2_tfidf_lsa[1:10,], method = "euclidean") %>% as.matrix() %>% 
heatmap()


calc_tfidf_lsa_dist2 <- function(df,dtm_tfidf_lsa)
{
  start = 0
  end   = 0
  nrow_df = nrow(df )
  pkg_size = 10000
  V = (nrow_df/pkg_size) %>% round() + 1
  M1 = data.frame()
  M2 = data.frame()
  M3 = data.frame()
  for (i in 1:V)
  {
    start = ( i - 1 ) * pkg_size + 1 
    end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
    
    m1 = dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "euclidean") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    

    m2 = dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "euclidean",norm="l1") %>%
      as.matrix()%>%
      diag(.) %>%
      as.data.frame()

    m3 = dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "euclidean",norm="none") %>%
      as.matrix()%>%
      diag(.) %>%
      as.data.frame()

    M1 = rbind(M1,m1)
    M2 = rbind(M2,m2)
    M3 = rbind(M3,m3)
    
    if( end >= nrow_df) break
    
  }
  data_frame_ = data.frame(M1, M2, M3)
  colnames(data_frame_) = c("tfidf_lsa_dist_m1","tfidf_lsa_dist_m2","tfidf_lsa_dist_m3" )
  data_frame_
}

dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[1:10,], dtm_tfidf_lsa$dtm2_tfidf_lsa[1:10,], method = "euclidean", norm="none") %>% 
  as.matrix() %>% 
  heatmap()


d1_d2_tfidf_lsa_dist2 = dtm_tfidf_lsa %>% calc_tfidf_lsa_dist2(df,. )


dtm_tfidf_lsa$

  
get_vocab_vectorizer <- function(Data) 
  {

      vectorizer <- rbind(Data$question1 ,Data$question2) %>% 
      itoken()  %>% 
      create_vocabulary( ) %>% 
      prune_vocabulary(term_count_min = 1L) %>% 
      vocab_vectorizer( )
    
    list(vectorizer = vectorizer, df = Data)
}


get_dtm_ <- function(Input_params) {

  dtm1 = Input_params$df$question1 %>% itoken(progressbar = FALSE) %>% create_dtm( Input_params$vectorizer)
  dtm2 = Input_params$df$question2 %>% itoken(progressbar = FALSE) %>% create_dtm( Input_params$vectorizer)
  list(dtm1 = dtm1, dtm2 = dtm2,df = Input_params$df)
}

get_tfidf_ <- function(dtm) {
  tfidf = TfIdf$new()
  dtm1_tfidf = fit_transform(dtm$dtm1, tfidf)
  dtm2_tfidf = fit_transform(dtm$dtm2, tfidf)
  list(dtm1_tfidf = dtm1_tfidf, dtm2_tfidf = dtm2_tfidf, df = dtm$df, dtm1 = dtm$dtm1, dtm2 = dtm$dtm2 )
}


get_lsa_ <- function( dtm_tfidf) {
  lsa = LSA$new( n_topics = 100)
  dtm1_tfidf_lsa = fit_transform(dtm_tfidf$dtm1_tfidf, lsa)
  dtm2_tfidf_lsa = fit_transform(dtm_tfidf$dtm2_tfidf, lsa)
  list(dtm1_tfidf_lsa = dtm1_tfidf_lsa, 
       dtm2_tfidf_lsa = dtm2_tfidf_lsa,
       dtm1_tfidf     = dtm_tfidf$dtm1_tfidf,
       dtm2_tfidf     = dtm_tfidf$dtm2_tfidf,
       dtm1 = dtm_tfidf$dtm1,
       dtm2 = dtm_tfidf$dtm2,
       df = dtm_tfidf$df )
}


calc_similarities <- function(dtm_tfidf_lsa)
{
  
  start = 0
  end   = 0
  nrow_df = nrow(dtm_tfidf_lsa$df )
  pkg_size = 10000
  V = (nrow_df/pkg_size) %>% round() + 1
  
  d1_d2_tfidf_lsa_sim = data.frame()
  
  d1_d2_tfidf_cos_sim = data.frame()
  
  tfidf_lsa_dist_m1    = data.frame()
  tfidf_lsa_dist_m2    = data.frame()
  tfidf_lsa_dist_m3    = data.frame()
  
  d1_d2_tfidf_lsa_cos_sim  = data.frame()

  d1_d2_jac_sim    = data.frame()
  d1_d2_cosine_sim = data.frame()
  
    
  for (i in 1:V)
  {
    start = ( i - 1 ) * pkg_size + 1 
    end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
  
    m1 = dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "euclidean") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    
    m2 = dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "euclidean",norm="l1") %>%
      as.matrix()%>%
      diag(.) %>%
      as.data.frame()
    
    m3 = dist2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "euclidean",norm="none") %>%
      as.matrix()%>%
      diag(.) %>%
      as.data.frame()
    d1_d2_jac_sim1 = sim2(dtm_tfidf_lsa$dtm1[start:end,], dtm_tfidf_lsa$dtm2[start:end,], method = "jaccard", norm = "none") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    d1_d2_cosine_sim1 = sim2(dtm_tfidf_lsa$dtm1[start:end,], dtm_tfidf_lsa$dtm2[start:end,], method = "cosine", norm = "l2") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    d1_d2_tfidf_cos_sim1 = sim2(dtm_tfidf_lsa$dtm1_tfidf[start:end,], dtm_tfidf_lsa$dtm2_tfidf[start:end,], method = "cosine", norm = "l2") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    d1_d2_tfidf_lsa_cos_sim1 = sim2(dtm_tfidf_lsa$dtm1_tfidf_lsa[start:end,], dtm_tfidf_lsa$dtm2_tfidf_lsa[start:end,], method = "cosine", norm = "l2") %>% 
      as.matrix()%>% 
      diag(.) %>%
      as.data.frame()
    
    d1_d2_jac_sim    = rbind(d1_d2_jac_sim,d1_d2_jac_sim1)
    d1_d2_cosine_sim = rbind(d1_d2_cosine_sim,d1_d2_cosine_sim1)
    
    d1_d2_tfidf_lsa_cos_sim = rbind(d1_d2_tfidf_lsa_cos_sim,d1_d2_tfidf_lsa_cos_sim1)
    d1_d2_tfidf_cos_sim     = rbind(d1_d2_tfidf_cos_sim,d1_d2_tfidf_cos_sim1)
    tfidf_lsa_dist_m1        = rbind(tfidf_lsa_dist_m1, m1 )
    tfidf_lsa_dist_m2        = rbind(tfidf_lsa_dist_m2, m2 )
    tfidf_lsa_dist_m3        = rbind(tfidf_lsa_dist_m3, m3 )
    
    if( end >= nrow_df) break
    
  }
  d1_d2_jac_psim = psim2(dtm_tfidf_lsa$dtm1, dtm_tfidf_lsa$dtm2, method = "jaccard", norm = "none") %>% as.data.frame()
  d1_d2_tfidf_lsa_cos_psim2 = psim2(dtm_tfidf_lsa$dtm1_tfidf_lsa, dtm_tfidf_lsa$dtm2_tfidf_lsa, method = "cosine", norm = "l2") %>% as.data.frame()
 
   data_frame_ = data.frame(d1_d2_tfidf_lsa_cos_sim,
                           d1_d2_tfidf_cos_sim,
                           tfidf_lsa_dist_m1,
                           tfidf_lsa_dist_m2 ,
                           tfidf_lsa_dist_m3,
                           d1_d2_jac_sim,
                           d1_d2_cosine_sim,
                           d1_d2_jac_psim,
                           d1_d2_tfidf_lsa_cos_psim2,
                           dtm_tfidf_lsa$df$is_duplicate)
  colnames(data_frame_) = c("tfidf_lsa_cos_sim",
                            "tfidf_cos_sim",
                            "tfidf_lsa_dist_m1",
                            "tfidf_lsa_dist_m2",
                            "tfidf_lsa_dist_m3",
                            "d1_d2_jac_sim",
                            "d1_d2_cosine_sim",
                            "d1_d2_jac_psim",
                            "d1_d2_tfidf_lsa_cos_psim2",
                            "is_duplicate")
  data_frame_
}

rm(dtm_tfidf_lsa)
dtm_tfidf_lsa <- df %>% get_vocab_vectorizer()  %>%  get_dtm_( ) %>% get_tfidf_() %>% get_lsa_()

df_simelarities <- dtm_tfidf_lsa %>% calc_similarities()

OneRaw = df[1,]
word_vectors = word_vectors_glove
calculate_covariance_matrix_q1q2 <- function(OneRaw,word_vectors)
{

  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question2"] %>% as.character()
  
  v = tibble(question1,question2)
  
  token_q1 <- v  %>% select(question1)  %>% unnest_tokens(word,question1)%>% as.vector() %>% t()  %>%  word_vectors[.,,drop=FALSE]
  token_q2 <- v  %>% select(question2)  %>% unnest_tokens(word,question2)%>% as.vector() %>% t() %>%   word_vectors[.,,drop=FALSE]
  
  
  covMatrixQ1Q2 = token_q1 %>% sim2( y = token_q2, method = "cosine", norm = "l2")
  
  dim_ <- covMatrixQ1Q2 %>% dim()
  
  covMatrixQ1Q2 <- covMatrixQ1Q2 %>% as.vector() %>% t()
  
  vec1 <- c(rep(0,20000 - (dim_[1]*dim_[2] + 2 ))) %>% t()
  covMatrixQ1Q2 = cbind(dim_[1],dim_[2],covMatrixQ1Q2,vec1)
  
  return(covMatrixQ1Q2)
  
}

save(df_simelarities, file = "df_simelarities.RData")
load("df_simelarities.RData")
head(df_simelarities) %>% View()

library(corrplot)
which(df_simelarities$d1_d2_jac_psim %>% is.na())

fg <- df_simelarities %>% select(-d1_d2_jac_psim) %>% scale( ) %>% corrplot()

fg <- df_simelarities %>% select(- d1_d2_jac_psim ) %>% select(-is_duplicate) %>% cor(method = c("pearson", "kendall", "spearman"))
fg %>% corrplot( type="upper", order="hclust", tl.col="black", tl.srt=45)





OneRaw = df[1,]
OneRaw[,'question1']

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

df[1,] %>% plot_conv_matrix_q1q2(word_vectors = word_vectors_glove)


# Multiplot heatmaps ---

v1 = which(df$is_duplicate == 1)[100:200] %>% df[.,] %>%   apply(MARGIN=1,FUN = plot_conv_matrix_q1q2,word_vectors) 
v1 %>% multiplot(cols = 3)

# word_vectors_glove
# 
# word_vectors <- rbind(df$question1,df$question2) %>% 
#   tokenize_text_() %>% 
#   glove_create_vocab() %>% 
#   glove_create_vectorizer() %>% 
#   glove_create_tcm() %>% 
#   glove_get_word_vector()

v <- cbind( question1_number_of_words,question2_number_of_words, is_duplicate = df$is_duplicate,id= df$id )
v <- v %>% filter( question1_no_words < 10 & question2_no_words <10 & is_duplicate == 0 ) %>% select(id) %>% mutate(id= id+1)
v1 = v[1:40,] %>% df[.,] %>%   apply(MARGIN=1,FUN = plot_conv_matrix_q1q2,word_vectors) 
v1 %>% multiplot(cols = 3)



show_similarity_heatmaps <- function(word_vectors)
{

plots_ = c()
for( i in which(df$is_duplicate == 0)[100:110] )
{
  OneRaw <- df[i,]
  
  plt <-  OneRaw %>% plot_conv_matrix_q1q2(word_vectors)
  plt
  


  
  # covMatrixQ1Q2[which(covMatrixQ1Q2 >  0.9)] = 10 
  # covMatrixQ1Q2[which(covMatrixQ1Q2 <= 0.9)] = 0
  # dim_ = covMatrixQ1Q2 %>% dim()
  # if( dim_[1] > dim_[2]) { 
  #   covMatrixQ1Q2 <-  cbind(covMatrixQ1Q2, matrix(0,dim_[1], (dim_[1] - dim_[2])))
  # }
  # if(dim_[1] < dim_[2])
  # {
  #   covMatrixQ1Q2 <-  rbind(covMatrixQ1Q2,  matrix(0, (dim_[2] - dim_[1]), dim_[2]) )
  # }
   # covMatrixQ1Q2 %>% as.matrix() %>% heatmap(Rowv = NA ,Colv = NA )
  # covMatrixQ1Q2 %>% as.matrix() %>% image()
}
plots_

}


pltm <- show_similarity_heatmaps(word_vectors_glove)
pltm[1]

calculate_covariance_matrix_q1q2 <- function(OneRaw,word_vectors)
{
  
  # wvec <-  df %>% build_word_vector_qura_questions()
  
  question1 <- OneRaw["question1"] %>% as.character()
  question2 <- OneRaw["question1"] %>% as.character()
  
  v = tibble(question1,question2)
  # 
  v_q1 <- v %>% select(question1)   %>% unnest_tokens(word,question1) %>% as.vector() %>% t()  
  v_q2 <- v %>% select(question2)   %>% unnest_tokens(word,question2) %>% as.vector() %>% t() 
  
  token_q1 <- v_q1  %>%  word_vectors[.,,drop=FALSE]
  token_q2 <- v_q2  %>%  word_vectors[.,,drop=FALSE]
  
  
  covMatrixQ1Q2 = token_q1 %>% sim2( y = token_q2, method = "cosine", norm = "l2")
  
  dim_ = covMatrixQ1Q2 %>% dim()
  if( dim_[1] > dim_[2]) { 
    covMatrixQ1Q2 <-  cbind(covMatrixQ1Q2, matrix(0,dim_[1], (dim_[1] - dim_[2])))
  }
  if(dim_[1] < dim_[2])
  {
    covMatrixQ1Q2 <-  rbind(covMatrixQ1Q2,  matrix(0, (dim_[2] - dim_[1]), dim_[2]) )
  }
  
  covMatrixQ1Q2 %>% as.matrix() %>% heatmap(Rowv = NA ,Colv = NA )
  
  
  eigvalues_ <- covMatrixQ1Q2 %>% eigen() 
  eigvalues_$vectors %>% heatmap()
  eigvalues_$values %>% diag() %>% heatmap()
  
  
  eigvalues_ <- covMatrixQ1Q2 %>% eigen(only.values = T) %>% .$values
  
  eigenvalue_length = eigvalues_ %>% length()
  eigenvalue_length
  
  vec1 <- c(rep(0,20 - (eigvalues_ %>% length())  )) %>% t()
  vec2 = cbind(eigvalues_ %>% t(),vec1)
  
  return(vec2)
  
}

df[7,] %>% calculate_covariance_matrix_q1q2(.,word_vectors_glove )

calculate_covariance_matrix_sim_q1q2 <- function(word_vec)
{
  
  sim_q1q2 <- word_vec$df %>% apply(MARGIN = 1,FUN = calculate_covariance_matrix_q1q2,word_vec$word_vectors ) %>% rbindlist( fill = TRUE) 
  
  sim_q1q2
}


build_word_vector_qura_questions <- function(df)
{
  # Create iterator over tokens
  
  tokens <- rbind(df$question1 ,df$question2) %>% space_tokenizer() 
  
  # Create vocabulary. Terms will be unigrams (simple words).
  it = itoken(tokens, progressbar = FALSE)
  
  vocab <- create_vocabulary(it)
  # vocab <- prune_vocabulary(vocab, term_count_min = 1L)
  
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = 1L)
  
  glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab,x_max = 1)
  wv_main <- glove$fit_transform(tcm, n_iter = 20)
  
  wv_context = glove$components
  word_vectors <- wv_main + t(wv_context)
  
  list(word_vectors = word_vectors,df = df)
  
}


question_feature_distances <- df %>% build_word_vector_qura_questions() %>%  calculate_covariance_matrix_sim_q1q2( )



############3333 Download Word Ebeding 
# download data if necessary


# Glove word2vec -----------------

get_text_for_text2vec <- function(Path)
{
  
  texts <- character()  # text samples
  fpath_table <- Path %>% file.path(., list.files(.))
  pb <- txtProgressBar( style = 3, title = "Read",min = 0,max =fpath_table %>% length, initial = 0 )
  for (fpath in fpath_table)
  { 
    t <- readLines(fpath, encoding = "latin1")
    t <- paste(t, collapse = "\n")
    i <- regexpr(pattern = '\n\n', t, fixed = TRUE)[[1]]
    if (i != -1L)
      t <- substring(t, i)
    texts <- c(texts, t)
    
  } 
  texts
  
}


tokenize_text_ <- function(texts_)
{
  # Create iterator over tokens
  it <- texts_ %>% space_tokenizer( ) %>% itoken( progressbar = FALSE)
  it
}



glove_create_vocab<- function(itokens)
{
  vocab <- itokens %>% create_vocabulary( )
  # vocab <- prune_vocabulary(vocab, term_count_min = 1L)
  list(vocab=vocab, itokens = itokens)
}


glove_create_vectorizer <- function(parameters)
{
  # Use our filtered vocabulary
  vectorizer <- parameters$vocab %>% vocab_vectorizer( )
  list( vectorizer = vectorizer, 
        vocab= parameters$vocab,
        itokens = parameters$itokens)
}


glove_create_tcm <- function(parameters,window_size)
{
  # use window of 5 for context words
  tcm <- parameters$itokens %>% create_tcm(parameters$vectorizer, 
                                           skip_grams_window = window_size)
  list(tcm = tcm,
       vocab= parameters$vocab)
}


glove_get_word_vector <- function(parameters,numb_iter,word_dim)
{
  glove = GlobalVectors$new( word_vectors_size = word_dim,
                            vocabulary = parameters$vocab, x_max = 10)
  wv_main <- glove$fit_transform(parameters$tcm, n_iter = numb_iter)
  
  wv_context = glove$components
  word_vectors <- wv_main + t(wv_context)
  
  word_vectors
}
  
merge_with_pretrained_word_vector <- function( word_vectors_glove,embeddings_index) {
  vect_words <- word_vectors_glove %>% row.names() %>% as.character() %>% as.data.frame( )
  colnames(vect_words) = c('word')
  row.names(vect_words )= vect_words$word
  vect_words <- vect_words %>% mutate(word = as.character(word))
  i = 1
  for( word in vect_words$word)
  {
    wvec = embeddings_index[[word]] 
    if( ! wvec %>% is.null() ) word_vectors_glove[i,] = wvec
    i = i + 1 
  }
#   Add two empty vectors at the end 
   v <- rep_len(0.0,length.out = 100)
   word_vectors_glove <-  rbind(word_vectors_glove,v)
   word_vectors_glove <-  rbind(word_vectors_glove,v)
   word_vectors_glove <-  rbind(word_vectors_glove,v)
   word_vectors_glove
  }



# Keras tokenization and word2vec --------
GLOVE_DIR <- 'glove.6B'
TEXT_DATA_DIR <- '20_newsgroup'
MAX_NUM_WORDS <- 2000000
EMBEDDING_DIM <- 100
VALIDATION_SPLIT <- 0.2


download_data <- function(data_dir, url_path, data_file) {
  if (!dir.exists(data_dir)) {
    download.file(paste0(url_path, data_file), data_file, mode = "wb")
    if (tools::file_ext(data_file) == "zip")
      unzip(data_file, exdir = tools::file_path_sans_ext(data_file))
    else
      untar(data_file)
    unlink(data_file)
  }
}
download_data(GLOVE_DIR, 'http://nlp.stanford.edu/data/', 'glove.6B.zip')
download_data(TEXT_DATA_DIR, "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/", "news20.tar.gz")


get_tokenizer_keras <- function(Texts,MAX_NUM_WORDS) {
  # finally, vectorize the text samples into a 2D integer tensor
  tokenizer <- text_tokenizer(num_words=MAX_NUM_WORDS)
  tokenizer <- tokenizer %>% fit_text_tokenizer(Texts)
  # sequences <- texts_to_sequences(tokenizer, texts)
  tokenizer
}

Path = GLOVE_DIR
create_embeding_index <- function(Path) {
  # fpath_table <- Path %>% file.path(., list.files(.))

  embeddings_index <- new.env(parent = emptyenv())
  fpath = "glove.6B/glove.6B.100d.txt"
  
  # for (fpath in fpath_table )
  # {
    lines <- readLines(fpath)
    for (line in lines) {
      values <- strsplit(line, ' ', fixed = TRUE)[[1]]
      word <- values[[1]]
      coefs <- as.numeric(values[-1])
      embeddings_index[[word]] <- coefs
     }
  # }
  embeddings_index
}



prepare_embedding_matrix <- function( Word_index ) {
  embedding_matrix <- matrix(0L, nrow = num_words, ncol = EMBEDDING_DIM)
  for (word in names(Word_index)) {
    index <- Word_index[[word]]
    if (index >= MAX_NUM_WORDS)
      next
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector)) {
      # words not found in embedding index will be all-zeros.
      embedding_matrix[index,] <- embedding_vector
    }
  }
  embedding_matrix
}

Sentence = df[1,]$question1

get_sentence_word_sequence <- function(Sentence,word_vectors_glove)
{
  
  v_q <- Sentence %>% as.character() %>% tibble(v=.)   %>% unnest_tokens(word,v) 
  v <- v_q$word %>%   sapply(function(x){ which( row.names(word_vectors_glove) == x)[[1]] } ) %>% na.omit 
  names(v) = c()
  v
  
}


prepare_questions_for_keras <- function(Questions,word_vectors_glove)
{
  n = word_vectors_glove %>% nrow()
  v <- Questions %>% sapply(FUN = get_sentence_word_sequence,word_vectors_glove) %>% unname()
  v <- v %>% pad_sequences( maxlen = 50, value = n - 1)
  v

}



embeddings_index <- GLOVE_DIR %>% create_embeding_index()

word_vectors_glove <- rbind(df$question1,df$question2) %>% 
  tokenize_text_() %>% 
  glove_create_vocab() %>% 
  glove_create_vectorizer() %>% 
  glove_create_tcm(15L) %>% 
  glove_get_word_vector(20,100) %>% 
  merge_with_pretrained_word_vector(embeddings_index)


question1 <- df$question1 %>% prepare_questions_for_keras(word_vectors_glove)
question2 <- df$question2 %>% prepare_questions_for_keras(word_vectors_glove)
save(question1,file='question1.RData')
save(question2,file='question2.RData')
save(word_vectors_glove,file='word_vectors_glove.RData')


word_vectors_glove_1d_1 <- rbind(df$question1,df$question2) %>% 
  tokenize_text_() %>% 
  glove_create_vocab() %>% 
  glove_create_vectorizer() %>% 
  glove_create_tcm(5L) %>% 
  glove_get_word_vector(120,1) 


word_vectors_glove_1d_2 <- rbind(df$question1,df$question2) %>% 
  tokenize_text_() %>% 
  glove_create_vocab() %>% 
  glove_create_vectorizer() %>% 
  glove_create_tcm(15L) %>% 
  glove_get_word_vector(120,1) 

save(word_vectors_glove_1d_1,file='word_vectors_glove_1d_1.RData')
save(word_vectors_glove_1d_2,file='word_vectors_glove_1d_2.RData')
save(word_vectors_glove,file="word_vectors_glove.RData")

load("df.RData")
load("word_vectors_glove_1d_1.RData")
load("word_vectors_glove_1d_2.RData")





Texts     <- GLOVE_DIR %>% get_text_for_text2vec( )
tokenizer <- Texts %>% get_tokenizer_keras(MAX_NUM_WORDS )

word_index <- tokenizer$word_index %>% unlist() %>%  as.data.frame() 

word_index_rows <- word_index %>% row.names() %>%  as.data.frame() 


embeddings_index <- GLOVE_DIR %>% create_embeding_index()

row.names(embeddings_index) = word_index_rows



# prepare embedding matrix
Word_index <- tokenizer$word_index
num_words <- min(MAX_NUM_WORDS, length(Word_index) + 1)



word_index_rows = word_index_rows %>% as.vector()

word_index_rows = word_index_rows %>% as.data.frame()

row.names(embedding_matrix) = word_index_rows


question1 %>% head %>% View



save(embedding_matrix,file="embedding_matrix.RData")
save(embeddings_index, file ="embeddings_index.RData")
save(word_index, file ="word_index.RData")

save(tokenizer, file = "tokenizer.RData")
load("tokenizer.RData")
load("embedding_matrix.RData")
load("embeddings_index.RData")


d1 <- word_index %>% unlist()

embeddings_index %>% dim()
Sentence = df[5,]$question1 

embeddings_index[['quick']]

library(keras)
word_vectors_glove %>% head %>% View

# Keras Model defintition ----
save(word_vectors_glove,file="word_vectors_glove.RData")
load("word_vectors_glove.RData")
load("question1.RData")
load("question2.RData")

word_vectors_glove %>% dim

sequence_input <- layer_input(shape = list(512))

embedding_layer <- layer_embedding(
  input_dim = 94562 ,
  output_dim = 100,
  weights = list(word_vectors_glove),
  input_length = 512,
  trainable = FALSE
)



preds_embedding  <- sequence_input %>% embedding_layer
model_embeding <- keras_model(sequence_input, preds_embedding)


seq_emb1 <- layer_lstm(
  units = 100, 
  recurrent_regularizer = regularizer_l2(l = 0.0001),
  return_sequences = T
)

seq_emb2 <- layer_lstm(
  units = 100, 
  recurrent_regularizer = regularizer_l2(l = 0.0001),
  return_sequences = TRUE
)

dense_100_tanh1 <- layer_dense( units = 100, activation = 'relu')
dense_100_tanh2 <- layer_dense( units = 100, activation = 'relu')
dense_020_relu1 <- layer_dense( units = 20, activation = 'tanh')
dense_020_relu2 <- layer_dense( units = 20, activation = 'tanh')

sequence_input <- layer_input(shape = list(512,100))

preds <- sequence_input %>% seq_emb1 %>% dense_100_tanh1 %>% 
  dense_020_relu1 %>% 
  layer_dropout(rate = 0.5)  %>% 
  dense_020_relu2 %>% 
  dense_100_tanh2 %>% seq_emb2 

model <- keras_model(sequence_input, preds)
model %>% compile(
  optimizer='adadelta', 
  loss='mse',
  metrics = c('accuracy')  )


encoder <- sequence_input %>% seq_emb1 %>% dense_100_tanh1 %>% 
  dense_020_relu1
encoder_model <- keras_model(sequence_input, encoder)



decoder       <- encoder %>%  
  dense_020_relu2 %>% 
  dense_100_tanh2 %>% seq_emb2

decoder_model <- keras_model(sequence_input, encoder)


start = 0
end   = 0
nrow_df = nrow(question1 )
pkg_size = 2000
V = (nrow_df/pkg_size) %>% round() + 1

histories = list()

for( i in seq_len(V))
{
  start = ( i - 1 ) * pkg_size + 1 
  end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
  nnet_data_input         = question1[start:end,]

  random_samples <- sample.int(pkg_size,size= 0.1 * pkg_size)
  
  nnet_data_input_train    = nnet_data_input[-random_samples,]
  nnet_data_input_valid   = nnet_data_input[random_samples,]
  
  # output_embedding <- model_embeding %>% predict(nnet_data_input)
  
  data_train <- model_embeding %>% predict(nnet_data_input_train)
  data_valid <- model_embeding %>% predict(nnet_data_input_valid)

  history = model %>%
    fit(
      data_train,
      data_train,
      batch_size = 1, 
      epochs = 1,
      callbacks = list(
        callback_early_stopping(patience = 5),
        callback_reduce_lr_on_plateau(patience = 3)
        
      ),
      validation_data = list(data_valid,data_valid) 
    )
  
  histories <- rbind(histories,history$metrics)
  save(histories, file= "histories.RData")
  
}


library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gcookbook)
library(plyr)

tg <- ddply(ToothGrowth, c("supp", "dose"), summarise, length=mean(len))
ggplot(tg, aes(x=dose, y=length, colour=supp)) + geom_line()


ggplot(worldpop, aes(x=Year, y=Population)) + geom_line() + geom_point()
worldpop %>% View()

load("histories.RData")
histories <- histories %>% as.data.frame()
v1 %>% rownames() %>% class()

plot(c(1:7),histories$acc)

histories %>% class()

histories <- histories %>% mutate(epochs = c(1:7) )

v1 %>% colnames()
v1 <- v1 %>% mutate( btc_nbr = seq_len(7) %>% as.numeric() )
v1 <- v1 %>% select(acc,val_acc,btc_nbr) %>% gather(key = accuR ,acc,-btc_nbr )


ggplot(uspopage, aes(x=Year, y=Thousands, fill=AgeGroup, order=desc(AgeGroup))) +
  geom_area(colour="black", size=.2, alpha=.4) 

ggplot(histories, aes(x=epochs )) + geom_point( aes(y=loss) )

plot(seq_len(7) %>% as.factor(),v1$loss)

v1 %>% ggplot( aes( x = btc_nbr, y = acc) ) +
  geom_point() +scale_y_log10()

  geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab('accuracy') + xlab('batch number') +
  geom_point(data = v1, size = 3, aes(x = batch_number, y = acc, color = accuracy))


library(gcookbook) # For the data set
ggplot(worldpop, aes(x=Year, y=Population)) + geom_line() + geom_point()


ggplot(data = roc_tidy, aes(x = fpr, y = tpr)) +
  geom_line(aes(color = model)) +
  geom_abline(intercept = 0, slope = 1, lty = 3) +
  ylab(perf@y.name) + xlab(perf@x.name) +
  geom_point(data = predictions_summary, size = 3,
             aes(x = test_fpr, y = test_tpr, color = model))



library(dplyr)
library(tidyr)
# From http://stackoverflow.com/questions/1181060
stocks <- tibble(
  time = as.Date('2009-01-01') + 0:9,
  X = rnorm(10, 0, 1),
  Y = rnorm(10, 0, 2),
  Z = rnorm(10, 0, 4)
)

gather(stocks, stock, price, -time)
stocks %>% gather(stock, price, -time)




print(model)

predict_final <- model %>% predict(intermediate_output_embedding)
intermediate_output_embedding %>% dim

model %>%
  fit(
    intermediate_output_embedding,
    intermediate_output_embedding,
     batch_size = 5, 
     epochs = 1000,
     callbacks = list(
       callback_early_stopping(patience = 5),
       callback_reduce_lr_on_plateau(patience = 3)
     )
  )




v1 = question1[1:2,] 
v1 %>% dim()
intermediate_output <- model %>% predict(v1)
intermediate_output %>% dim




w1 <- embedding_layer %>% get_weights()[[1]]
w1 = w1[[1]]

identical(w1[1,1:10],word_vectors_glove1[1,1:10])
word_vectors_glove1[1,1:10]
w1[1,1:10]
######################################
FLAGS <- flags(
  flag_integer("vocab_size", word_vectors_glove %>% nrow),
  flag_integer("max_len_padding", 512),
  flag_integer("embedding_size",  word_vectors_glove %>% ncol),
  flag_numeric("regularization", 0.0001),
  flag_integer("seq_embedding_size", 512)
)

input1 <- layer_input(shape = c(512))
input2 <- layer_input(shape = c(512))

embedding <- layer_embedding(
  input_dim = 94562, 
  output_dim = 100, 
  input_length = 512, 
  weights = list(word_vectors_glove),
  trainable = FALSE
)



seq_emb <- layer_lstm(
  units = FLAGS$seq_embedding_size, 
  recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
)

vector1 <- embedding(input1) %>%
  seq_emb()
vector2 <- embedding(input2) %>%
  seq_emb()

out <- layer_dot(list(vector1, vector2), axes = 1) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(list(input1, input2), out)
model %>% compile(
  optimizer = "adam", 
  loss = "binary_crossentropy", 
  metrics = list(
    acc = metric_binary_accuracy
  )
)

model %>% print()

# rm(question1)
# question1 <- c()
# start = 0
# end   = 0
# nrow_df = nrow(df )
# pkg_size = 100
# V = (nrow_df/pkg_size) %>% round() + 1
# for (i in 1:V)
# {
#   start = ( i - 1 ) * pkg_size + 1 
#   end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
#   q1 = df$question1[start:end] %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()
#   question1 <- rbind(question1,q1)
# }

question1 = df$question1 %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()
question2 = df$question2 %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()

save(question1,file="question1.RData")
save(question2,file="question2.RData")
load('df.RData')


question1_1d_1 = df$question1 %>%  prepare_questions_for_keras(word_vectors_glove_1d_1) %>% as.matrix()
question2_1d_1 = df$question2 %>%  prepare_questions_for_keras(word_vectors_glove_1d_1) %>% as.matrix()
save(question1_1d_1,file="question1_1d_1.RData")
save(question2_1d_1,file="question2_1d_1.RData")




question1_1d_2 = df$question1 %>%  prepare_questions_for_keras(word_vectors_glove_1d_2) %>% as.matrix()
question2_1d_2 = df$question2 %>%  prepare_questions_for_keras(word_vectors_glove_1d_2) %>% as.matrix()
save(question1_1d_2,file="question1_1d_2.RData")
save(question2_1d_2,file="question2_1d_2.RData")


load('question1_1d_1.RData')
load('question2_1d_1.RData')

load('question1_1d_2.RData')
load('question2_1d_2.RData')


question1_1d_1 = question1_1d_1[,31:50]
question2_2d_1 = question2_1d_1[,31:50]
question1_1d_2 = question1_1d_2[,50:31]
question2_2d_2 = question2_1d_2[,50:31]




question1_1d_1  = question1_1d_1 %>% word_vectors_glove_1d_1[.]%>% matrix(404290,20)
question2_1d_1  = question2_1d_1 %>% word_vectors_glove_1d_1[.]%>% matrix(404290,20)
question1_1d_2  = question1_1d_2 %>% word_vectors_glove_1d_2[.]%>% matrix(404290,20)
question2_1d_2  = question2_1d_2 %>% word_vectors_glove_1d_2[.]%>% matrix(404290,20)



set.seed(1817328)
val_sample <- sample.int(nrow(question1), size = 0.1*nrow(question1))
val_sample

model %>%
  fit(
    list(question1[1:20,], question2[1:20,]),
    df$is_duplicate[1:20]
    # , 
    # batch_size = 128, 
    # epochs = 30, 
    # callbacks = list(
    #   callback_early_stopping(patience = 5),
    #   callback_reduce_lr_on_plateau(patience = 3)
    # )
  )



# Words comparaison -----------------------
library(plotrix)
load("df.RData")


vocab1 <- df$question1 %>% space_tokenizer( ) %>% itoken( progressbar = FALSE ) %>%  create_vocabulary( )
vocab2 <- df$question2 %>% space_tokenizer( ) %>% itoken( progressbar = FALSE)  %>%  create_vocabulary( )

colnames(vocab2) = c("term","term_count2","doc_count2" )
colnames(vocab1) = c("term","term_count1","doc_count1" )

comoun_vocab_1_2 <- vocab1 %>% inner_join(vocab2 ) %>% arrange(term_count1 %>% desc() , term_count2 %>% desc())


set.seed(1817328)
val_sample <- sample.int(nrow(comoun_vocab_1_2), size = 20)

comoun_vocab_1_2_subset <- comoun_vocab_1_2[500:530,]

which(comoun_vocab_1_2$question2 == 0)

? pyramid.plot
pyramid.plot(comoun_vocab_1_2_subset$term_count1,comoun_vocab_1_2_subset$term_count2, labels = comoun_vocab_1_2_subset$term,
             gap = 100, space = 0.3,top.labels = c("Question1","Words","Question2") ,main = "Words in common", laxlab = NULL, raxlab = NULL, unit = NULL)

pyramid.plot(comoun_vocab_1_2_subset$term_count1,comoun_vocab_1_2_subset$term_count2, labels = comoun_vocab_1_2_subset$term,
             gap = 14, top.labels = c("Question1","Words","Question2") ,main = "Words in common", laxlab = NULL, raxlab = NULL, unit = NULL)




# World cloud
row.names(comoun_vocab_1_2) = comoun_vocab_1_2$term
comoun_vocab_1_2 <- comoun_vocab_1_2 %>% select(term_count1,term_count2)
colnames(comoun_vocab_1_2)  = c("question1", "question2")
library(RColorBrewer)
library(wordcloud)
display.brewer.all()
pal <- brewer.pal(8,"Purples")
pal <- pal[-(1:4)]

comoun_vocab_1_2 %>% commonality.cloud(max.words = 750, random.order = FALSE, colors = pal)
comoun_vocab_1_2 %>% comparison.cloud( max.words = 750, random.order = FALSE, title.size = 1.0, colors = brewer.pal( ncol(comoun_vocab_1_2) , "Dark2"))






#####################################


################### Test world vector
## http://text2vec.org/glove.html
word_vectors <- build_word_vector_qura_questions()
save(word_vectors,file="word_vectors.RData")

word_vectors_q1 <- build_word_vector_qura_question1()
save(word_vectors_q1,file="word_vectors_q1.RData")
rm(word_vectors_q1)

word_vectors_q2 <- build_word_vector_qura_question2()
save(word_vectors_q2,file="word_vectors_q2.RData")
rm(word_vectors_q2)

words_not_in_wvec <- find_words_not_in_word_vec()

words <- word_vectors %>% rownames()
head(words,1000)


qw = which(df$is_duplicate == 1, 100)

i = 100
Q = df[qw[[i]],]  %>% covariance_matrix_Q1Q2()
heatmap(Q)
qw[[i]]
display_questions(df,qw[[i]] - 1)



qw = which(df$is_duplicate == 0, 1)
i = 35
Q = df[qw[[i]],]  %>% covariance_matrix_Q1Q2()
heatmap(Q)
qw[[i]]
display_questions(df,qw[[i]] - 1)


Q %>% dim()

apply(Q, 2, min) %>% length()
apply(Q, 1, min) %>% length()

apply(Q, 2, min) %>% sum() / nrow(Q)
apply(Q, 1, min) %>% sum() / ncol(Q)

which(Q > 0.6) %>% length()
Q %>% dim()


df[54,]$is_duplicate

rownames(Q)
colnames(Q)



##################  text2vec  word embding experimentation 

library(text2vec)
text8_file = "~/Texts"
if (!file.exists(text8_file)) {
  download.file("http://mattmahoney.net/dc/text8.zip", "~/text8.zip")
  unzip ("~/text8.zip", files = "text8", exdir = "~/")
}
wiki = readLines("Texts/text8", n = 1, warn = FALSE)
rm(wiki)
# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)

vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 2L)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 2L)
save(tcm,file="tcm.RData")
rm(tcm)

glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
wv_main <- glove$fit_transform(tcm, n_iter = 20)

wv_context = glove$components
dim(wv_context)

word_vectors = wv_main + t(wv_context)
save(word_vectors, file= "word_vectors.RData")

save(vocab,file="vocab.RData")
rm("vocab")


library(tidytext)

df[1,] %>% select(question1)
documents <- df[1,] %>% select(question1)  %>% unnest_tokens(word,question1)

w1= documents[3,] %>% as.character()
x = word_vectors[w1,,drop=FALSE]
x

w1= documents[6,] %>% as.character()
w2= documents[2,] %>% as.character()
w3= documents[3,] %>% as.character()
w4= documents[4,] %>% as.character()
w5= documents[5,] %>% as.character()
w1
y = rbind( word_vectors[,w1,drop=FALSE],
           word_vectors[,w2,drop=FALSE],
           word_vectors[,w3,drop=FALSE])

y

word_vectors[,w1,drop=FALSE]


cos_sim = sim2(x = x, y = y, method = "cosine", norm = "l2")


berlin = word_vectors["berber", , drop = FALSE] - 
  word_vectors["france", , drop = FALSE] + 
  word_vectors["germany", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = berlin, method = "cosine", norm = "l2")
berlin
head(sort(cos_sim[,1],decreasing = TRUE),10)


word_vectors[,"france" ,drop=FALSE] + 
  word_vectors[,"germany" , drop = FALSE]

berlin
cos_sim = sim2(x = word_vectors, y = berlin, method = "cosine", norm = "l2")


c() %>% 
  VectorSource() %>%
  Corpus() %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map( removePunctuation) %>% 
  tm_map(removeWords, soptwords_en )
tokenQ1 <- documents[1]$content %>% tokenize_words()%>% unlist()
tokenQ1

rm(wiki)
rm
rm(tokens)

########################## Clustering 

library(apcluster)
## create two Gaussian clouds
cl1 <- cbind(rnorm(100,0.2,0.05),rnorm(100,0.8,0.06))
cl2 <- cbind(rnorm(100,0.7,0.08),rnorm(100,0.3,0.05))
x <- rbind(cl1,cl2)

## create negative distance matrix (default Euclidean)
sim1 <- negDistMat(x)
sim1
## compute similarities as squared negative distances
## (in accordance with Frey's and Dueck's demos)
sim2 <- negDistMat(x, r=2)
sim2
## compute RBF kernel
sim3 <- expSimMat(x, r=2)

sim3
## compute similarities as squared negative distances
## all samples versus a randomly chosen subset 
## of 50 samples (for leveraged AP clustering)
sel <- sort(sample(1:nrow(x), nrow(x)*0.25)) 
sim4 <- negDistMat(x, sel, r=2)


## example of leveraged AP using Minkowski distance with non-default
## parameter p
cl1 <- cbind(rnorm(150,0.2,0.05),rnorm(150,0.8,0.06))
cl2 <- cbind(rnorm(100,0.7,0.08),rnorm(100,0.3,0.05))
x <- rbind(cl1,cl2)

apres <- apclusterL(s=negDistMat(method="minkowski", p=2.5, r=2),
                    x, frac=0.2, sweeps=3, p=-0.2)
show(apres)




POS_tags(df$question1[1])

MAP_POS_tags_to_Data(df$question1[1])

MAP_POS_tags_to_Data(df$question1[2])
MAP_POS_tags_to_Data(df$question1[3])
MAP_POS_tags_to_Data(df$question1[4])


vb <- df$question1[1:10000] %>% lapply(FUN = MAP_POS_tags_to_Data) 
question1_featurs <- vb %>% rbindlist( fill = TRUE)
question1_featurs <- question1_featurs %>% mutate(id = df$id[1:10000], qid1 = df$qid1[1:10000], qid2 = df$qid2[1:10000]  )

write.csv(question1_featurs,file= "question1_featurs.csv")

vb <- df$question2[1:10000] %>% lapply(FUN = MAP_POS_tags_to_Data) 
question2_featurs <- vb %>% rbindlist( fill = TRUE)
question2_featurs <- question2_featurs %>% mutate(id = df$id[1:10000], qid1 = df$qid1[1:10000], qid2 = df$qid2[1:10000]  )
write.csv(question2_featurs,file= "question2_featurs.csv")


more_featurs <- df %>%   apply(MARGIN=1,FUN = more_features) 
more_featurs <- more_featurs %>% rbindlist( fill = TRUE)
more_featurs
more_featurs[8,]


View(Tag_sequences_Q1Q2)





######################## Experiment with traminR (Sequence)

library(TraMineR)
data(mvad)
mvad.alphab <- c("employment", "FE", "HE", "joblessness","school", "training")
mvad.seq <- seqdef(mvad, 17:86, xtstep = 6, alphabet = mvad.alphab)

mvad.om <- seqdist(mvad.seq, method = "OM", indel = 1, sm = "TRATE")
mvad.om[1,1:10]

dim(mvad.om)

library("cluster")
clusterward <- agnes(mvad.om, diss = TRUE, method = "ward")
mvad.cl4 <- cutree(clusterward, k = 4)
cl4.lab <- factor(mvad.cl4, labels = paste("Cluster", 1:4))

seqdplot(mvad.seq, group = cl4.lab, border = NA)


entropies <- seqient(mvad.seq)
lm.ent <- lm(entropies ~ male + funemp + gcse5eq, mvad)
lm.ent

mvad[1:2, 17:22]

data(mvad)
dim(mvad)
mvad.lab <- c("Employment", "Further education", "Higher education", "Joblessness", "School", "Training")
mvad.lab %>% length()
mvad.scode <- c("EM", "FE", "HE", "JL", "SC", "TR")
mvad.scode %>% length()
mvad2 = mvad[1:10,17:86]
mvad2[10,]
mvad.seq <- seqdef(mvad2, 17:86, alphabet = mvad.alphab, states = mvad.scode,labels = mvad.lab, xtstep = 6)
print(mvad.seq[1:5, ], format = "SPS")

mvad.seq <- seqdef(mvad, 17:86, alphabet = mvad.alphab, states = mvad.scode, labels = mvad.lab, weights = mvad$weight, xtstep = 6)
seqiplot(mvad.seq, border = NA, with.legend = "right")


mvad.scode <- c(1, 2, 3, 5, 9, 11,12,13)
mvad.lab <- c("L1", "L2", "L3", "L4", "L5", "L6","L7","L8")
mdat <- matrix(c(1,2,3, 11,12,1,2,3,13,1,2,3,1,2,3,5,1,2,1,2,9), 
               nrow = 7, 
               ncol = 3, 
               byrow = TRUE)

mvad.seq <- seqdef(mdat)
mvad.seq
seqLLCP(mvad.seq[3, ], mvad.seq[5, ])

1 - seqdist(mvad.seq, method = "LCP",norm = TRUE)
1 - seqdist(mvad.seq, method = "LCS",norm = TRUE)

seqLLCS(mvad.seq[4, ], mvad.seq[5, ])


couts <- seqsubm(mvad.seq, method = "TRATE")
round(couts)

library(cluster)
mvad.om <- seqdist(mvad.seq, method = "OM", indel = 3, sm = couts)
clusterward <- agnes(mvad.om, diss = TRUE, method = "ward")
plot(clusterward, which.plots = 2)

cluster3 <- cutree(clusterward, k = 3)
cluster3 <- factor(cluster3, labels = c("Type 1", "Type 2", "Type 3"))
table(cluster3)
seqfplot(mvad.seq, group = cluster3, pbarw = T)
seqmtplot(mvad.seq, group = cluster3)


data(famform)

class(famform)

famform.seq <- seqdef(famform)
famform.seq
seqmpos(famform.seq[1, ], famform.seq[2, ])

dim(famform)


k1 <-  POS_tags_seq(df$question1[8])
k2 <-  POS_tags_seq(df$question2[8])
k1
k2

k <- df[1:100,] %>% apply(MARGIN = 1,POS_tags_seq )
k %>% as.matrix()


bn <- seqdef(k)
bn

couts <- seqsubm(bn, method = "TRATE")
round(couts)

mvad.om <- seqdist(bn, method = "OM", indel = 5, sm = couts)
clusterward <- agnes(mvad.om, diss = TRUE, method = "ward")
plot(clusterward, which.plots = 2)
seqmtplot(mvad.seq, group = cluster3)


cluster3 <- cutree(clusterward, k = 5)
cluster3 <- factor(cluster3, labels = c("Type 1", "Type 2", "Type 3","Type 4","Type 5"))
table(cluster3)
seqfplot(bn, group = cluster3, pbarw = T)
seqmtplot(bn, group = cluster3)




#########################3 Markov HMM sequences

library("seqHMM")
data("biofam3c")
marr_seq <- seqdef(
  biofam3c$married, start = 15,
  alphabet = c("single", "married", "divorced"))
child_seq <- seqdef(
  biofam3c$children, start = 15,
  alphabet = c("childless", "children"))
left_seq <- seqdef(
  biofam3c$left, start = 15,
  alphabet = c("with parents", "left home"))
attr(marr_seq, "cpal") <- c("violetred2", "darkgoldenrod2",
                            "darkmagenta")
attr(child_seq, "cpal") <- c("darkseagreen1", "coral3")
attr(left_seq, "cpal") <- c("lightblue", "red3")
ssplot(
  x = list("Marriage" = marr_seq, "Parenthood" = child_seq,
           "Residence" = left_seq))

s1 <- seqdef(df[1,]["question1"] %>% tokenize_words %>% unlist())
s1
s2 <- seqdef(df[1,]["question2"] %>% tokenize_words %>% unlist())
s2
s3 <- seqdef(df[2,]["question1"] %>% tokenize_words %>% unlist())
s3 <- seqdef(df[2,]["question2"] %>% tokenize_words %>% unlist())
s3

data("hmm_biofam")
ssplot(x = hmm_biofam, plots = "both")


ssp_def <- ssp(
  hmm_biofam, plots = "both", type = "I", sortv = "mds.hidden",
  ylab.pos = c(1, 2),
  title = "Family trajectories", title.n = FALSE,
  xtlab = 15:30, xlab = "Age",
  ncol.legend = c(2, 1, 1), legend.prop = 0.37)
plot(ssp_def)


plot(hmm_biofam,
     layout = matrix(c(1, 2, 3, 4, 2,
                       1, 1, 1, 1, 0), ncol = 2),
     xlim = c(0.5, 4.5), ylim = c(-0.5, 1.5), rescale = FALSE,
     edge.curved = c(0, -0.8, 0.6, 0, 0, -0.8, 0),
     cex.edge.width = 0.8, edge.arrow.size = 1,
     legend.prop = 0.3, ncol.legend = 2,
     vertex.label.dist = 1.1, combine.slices = 0.02,
     combined.slice.label = "others (emission prob. < 0.02)")
