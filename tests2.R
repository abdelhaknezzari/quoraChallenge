load("df_more_features.RData")

df_more_features <- df_more_features %>%  
  filter( !q2_q1_cos_dist_norm    %>% is.infinite() & 
            !q1_q2_cos_dist_norm    %>% is.infinite() &
            !q1_q2_cosine_distance  %>% is.infinite() &
            !q2_q1_cos_dist_norm    %>% is.infinite() & 
            !cosin_distance_bag_pos %>% is.infinite()   )


glimpse(df_more_features)

set.seed(3645789)
n <- nrow(df_more_features)
test_idx <- sample.int(n, size = round(0.2 * n))
train <- df_more_features[-test_idx, ]
nrow(train)

test <- df_more_features[test_idx, ]
nrow(test)
library(dplyr)

library(rpart)
rpart(is_duplicate ~ words_delta_q1_q2  + words_delta_q2_q1  + q1_q2_no_stopwords +  q2_q1_no_stopwords + q1_length  , data = train)

# split <- 5095.5
# train <- train %>% mutate(hi_cap_gains = capital.gain >= split)
# ggplot(data = train, aes(x = capital.gain, y = income)) +
#   geom_count(aes(color = hi_cap_gains),
#              position = position_jitter(width = 0, height = 0.1), alpha = 0.5) +
#   geom_vline(xintercept = split, color = "dodgerblue", lty = 2) +
#   scale_x_log10(labels = scales::dollar)      

# + words_delta_q1_q2+words_delta_q2_q1+q1_q2_no_stopwords+q2_q1_no_stopwords + q1_length  + q2_length 
form <- as.formula("is_duplicate ~  traminr_seqLLCS + traminr_seqLLCP + traminr_seqmpos  + q1_q2_cos_dist_norm + q2_q1_cos_dist_norm")
mod_tree <- rpart(form, data = train)
mod_tree
plot(mod_tree)
text(mod_tree, use.n = TRUE, all = TRUE, cex = 0.7)

library(partykit)
library(mosaic)

plot(as.party(mod_tree))

# train <- train %>%
#   mutate(husband_or_wife = relationship %in% c(" Husband", " Wife"),
#          college_degree = husband_or_wife & education %in%
#            c(" Bachelors", " Doctorate", " Masters", " Prof-school"),
#          income_dtree = predict(mod_tree, type = "class"))
# cg_splits <- data.frame(husband_or_wife = c(TRUE, FALSE),
#                         vals = c(5095.5, 7073.5))
# ggplot(data = train, aes(x = capital.gain, y = income)) +
#   geom_count(aes(color = income_dtree, shape = college_degree),
#              position = position_jitter(width = 0, height = 0.1),
#              alpha = 0.5) +
#   facet_wrap(~ husband_or_wife) +
#   geom_vline(data = cg_splits, aes(xintercept = vals),
#              color = "dodgerblue", lty = 2) +
#   scale_x_log10()

printcp(mod_tree)

train <- train %>%
  mutate(income_dtree = predict(mod_tree, type = "class"))

confusion <- tally(income_dtree ~ is_duplicate, data = train, format = "count")
confusion

mod_tree2 <- rpart(form, data = train, control = rpart.control(cp = 0.002))


library(randomForest)
mod_forest <- randomForest(form, data = train, ntree = 201, mtry = 3)
mod_forest

sum(diag(mod_forest$confusion)) / nrow(train)

library(tibble)
importance(mod_forest) %>%
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(desc(MeanDecreaseGini))

library(class)
# distance metric only works with quantitative variables
train_q <- train 

income_knn <- knn(train, test = test, cl = train$is_duplicate, k = 1)
confusion <- tally(income_knn ~ is_duplicate, data = train, format = "count")
confusion


knn_error_rate <- function(x, y, numNeighbors, z = x) {
  y_hat <- knn(train = x, test = z, cl = y, k = numNeighbors)
  return(sum(y_hat != y) / nrow(x))
}
ks <- c(1:15, 20, 30, 40, 50)
train_rates <- sapply(ks, FUN = knn_error_rate, x = train_q, y = train$is_duplicate)
knn_error_rates <- data.frame(k = ks, train_rate = train_rates)
ggplot(data = knn_error_rates, aes(x = k, y = train_rate)) +
  geom_point() + geom_line() + ylab("Misclassification Rate")



library(e1071)
mod_nb <- naiveBayes(form, data = train)
income_nb <- predict(mod_nb, newdata = train)
confusion <- tally(income_nb ~ is_duplicate, data = train, format = "count")
confusion
sum(diag(confusion)) / nrow(train)


library(nnet)
mod_nn <- nnet(form, data = train, size = 5)
income_nn <- predict(mod_nn, newdata = train, type = "class")
confusion <- tally(income_nn ~ is_duplicate, data = train, format = "count")
confusion
sum(diag(confusion)) / nrow(train)


income_ensemble <- ifelse((income_knn == "1") +
                            (income_nb == "1") +
                            (income_nn == "1") >= 2, "1", "0")
confusion <- tally(income_ensemble ~ income, data = train, format = "count")
confusion
sum(diag(confusion)) / nrow(train)



income_probs <- mod_nb %>%
  predict(newdata = train, type = "raw") %>%
  as.data.frame()
head(income_probs, 3)
names(income_probs)


tally(~`1` > 0.5, data = income_probs, format = "percent")
tally(~`1` > 0.24, data = income_probs, format = "percent")


income_probs[,2] %>% class()
pred <- ROCR::prediction(income_probs[,2], train$is_duplicate)
perf <- ROCR::performance(pred, 'tpr', 'fpr')
class(perf) # can also plot(perf


perf_df <- data.frame(perf@x.values, perf@y.values)

names(perf_df) <- c("fpr", "tpr")
roc <- ggplot(data = perf_df, aes(x = fpr, y = tpr)) +
  geom_line(color="blue") + geom_abline(intercept=0, slope=1, lty=3) +
  ylab(perf@y.name) + xlab(perf@x.name)
roc
confusion <- tally(income_nb ~ income, data = train, format = "count")
confusion
sum(diag(confusion)) / nrow(train)

tpr <- confusion["1", "1"] / sum(confusion[, "1"])
fpr <- confusion["1", "0"] / sum(confusion[, "0"])
roc + geom_point(x = fpr, y = tpr, size = 3)





test_q <- test %>%
  select(age, education.num, capital.gain, capital.loss, hours.per.week)
test_rates <- sapply(ks, FUN = knn_error_rate, x = train_q,
                     y = train$income, z = test_q)
knn_error_rates <- knn_error_rates %>% mutate(test_rate = test_rates)
library(tidyr)
knn_error_rates_tidy <- knn_error_rates %>%
  gather(key = "type", value = "error_rate", -k)
ggplot(data = knn_error_rates_tidy, aes(x = k, y = error_rate)) +
  geom_point(aes(color = type)) + geom_line(aes(color = type)) +
  ylab("Misclassification Rate")
library(mosaic)
favstats(~ capital.gain, data = train)
favstats(~ capital.gain, data = test)



mod_null <- glm(is_duplicate ~ ., data = train, family = binomial)
mods <- list(mod_null, mod_tree, mod_forest, mod_nn, mod_nb)
lapply(mods, class)
predict_methods <- methods("predict")
predict_methods[grepl(pattern = "(glm|rpart|randomForest|nnet|naive)",
                      predict_methods)]


predictions_train <- data.frame(
  y = as.character(train$is_duplicate),
  type = "train",
  mod_null = predict(mod_null, type = "response"),
  # mod_tree = predict(mod_tree, type = "class"),
  mod_forest = predict(mod_forest, type = "class"),
  mod_nn = predict(mod_nn, type = "class"),
  mod_nb = predict(mod_nb, newdata = train, type = "class"))


predictions_test <- data.frame(
  y = as.character(test$is_duplicate),
  type = "test",
  mod_null = predict(mod_null, newdata = test, type = "response"),
  # mod_tree = predict(mod_tree, newdata = test, type = "class"),
  mod_forest = predict(mod_forest, newdata = test, type = "class"),
  mod_nn = predict(mod_nn, newdata = test, type = "class"),
  mod_nb = predict(mod_nb, newdata = test, type = "class"))



predictions <- bind_rows(predictions_train, predictions_test)

glimpse(predictions)

predictions_tidy <- predictions %>%
  mutate(mod_null = ifelse(mod_null < 0.5, "0", "1")) %>%
  gather(key = "model", value = "y_hat", -type, -y)
glimpse(predictions_tidy)


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
predictions_summary

outputs <- c("response", "prob", "prob", "raw", "raw")
roc_test <- mapply(predict, mods, type = outputs,
                   MoreArgs = list(newdata = test)) %>%
  as.data.frame() %>%
  select(1,3,5,6,8)
names(roc_test) <-
  c("mod_null", "mod_tree", "mod_forest", "mod_nn", "mod_nb")
glimpse(roc_test)

get_roc <- function(x, y) {
  pred <- ROCR::prediction(x$y_hat, y)
  perf <- ROCR::performance(pred, 'tpr', 'fpr')
  perf_df <- data.frame(perf@x.values, perf@y.values)
  names(perf_df) <- c("fpr", "tpr")
  return(perf_df)
}
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

library(scales)

nba <- read.csv("http://datasets.flowingdata.com/ppg2008.csv")
nba.m <- melt(nba)
nba$Name <- with(nba, reorder(Name, PTS))
nba.m <- ddply(nba.m, .(variable), transform,  rescale = rescale(value))


nba.m %>% View()
p <- ggplot(nba.m, aes(variable, Name)) + geom_tile(aes(fill = rescale),colour = "white") + scale_fill_gradient(low = "white", high = "steelblue")
base_size <- 9
p + theme_grey(base_size = base_size) + labs(x = "",y = "") + 
   scale_x_discrete(expand = c(0, 0) ) +
   scale_y_discrete(expand = c(0, 0) ) +
   theme(legend.position = "none",strip.text.x= element_text(size = base_size *0.8, angle = 290, hjust = 0, colour = "grey50"))





A <- matrix(c(2,5,2,1,0,0,0,0,1,0,0,0,0,1,3,5,6,0,0,1,0,0,0,2,0,0,1,2,7,2,4,6,2,5,1,0,0,1,0,0,0,1,0,0,3,5,4,0,0,1,0,0,1,0,0,2,0,3,5,7,3,1,4,0,1,0,0,0,0,2,0,0,0,1,3,4,6,0,0,1), byrow=T, nrow=8, ncol=10)
colnames(A) <- letters[1:10]
rownames(A) <- LETTERS[1:8]

library(reshape2)
library(ggplot2)


longData<-melt(A)
longData<-longData[longData$value!=0,]
longData




#
quora_data <- get_file(
  "quora_duplicate_questions.tsv",
  "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv" )




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

clean_appostrophes <- function(Df)
{
  patterns     <- c("it's"   ,"he's"   ,"she's"  ,"i'm" ,"'re "   ) 
  replacements <- c("it is"  ,"he is"  ,"she is" ,"i am"," are "  ) 
  
  
  patterns    <- cbind(patterns     %>% t(),"won't"   ,"ain't"  ,"n't " , "you'v")
  replacements<- cbind(replacements %>% t(),"will not","are not"," not ", "you have")
  
  
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
split_non_ascii <- function(question)
{
  ifelse( question %>% question_has_non_asccii(),split_q(question) ,question)
  
}
question_has_non_asccii <- function(question)
{
  non_ascii_q = question %>% ex_non_ascii() %>% unlist()
  
  logic_nonacii <- non_ascii_q %>% is_empty()
  if (!logic_nonacii )  logic_nonacii <- ! ( non_ascii_q %>% is.na() %>% mean() >0 )
  logic_nonacii
}

split_non_ascii_words_df <- function(Df)
{
  Df <- Df %>% apply(MARGIN = 1,FUN = split_non_ascii_words) %>% rbindlist( fill = TRUE)  
  Df
} 

split_q <- function(question)
{
  v = tibble(question = question ) 
  Q <-   v %>% select(question)  %>% unnest_tokens(word,question)
  Q <- Q %>% as.vector()  %>% t() %>% paste(collapse = ' ')    
  Q
}

stem_question <- function(Question)
{
  rt <- Question %>% c() %>% VectorSource() %>%  Corpus() %>% tm_map(stemDocument)
  rt[1]$content  
}



fix_contactions <- function(df)
{
  df$question1 = df$question1 %>% as.character() %>%  textclean::replace_contraction( )    
  df$question2 = df$question2 %>% as.character() %>%  textclean::replace_contraction( )
  
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "you'v",replacement = "you have")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "you'v",replacement = "you have")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "i'v",replacement = "i have")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "i'v",replacement = "i have")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "'v ",replacement = "have ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "'v ",replacement = "have ")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "'r ",replacement = "are ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "'r ",replacement = "are ")
  
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "haven't",replacement = "have not ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "haven't",replacement = "have not ")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "does't",replacement = "does not")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "does't",replacement = "does not")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "doesn't",replacement = "does not")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "doesn't",replacement = "does not")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "dosen't",replacement = "does not")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "dosen't",replacement = "does not")
  
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "hadn't",replacement = "had not")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "hadn't",replacement = "had not")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "had't ",replacement = "had not ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "had't ",replacement = "had not ")
  
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "wouldn't",replacement = "would not ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "wouldn't",replacement = "would not ")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "could't",replacement = "could not ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "could't",replacement = "could not ")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "should't",replacement = "should not ")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "should't",replacement = "should not ")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "gov't",replacement = "government")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "gov't",replacement = "government")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "mos't",replacement = "most")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "mos't",replacement = "most")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "wouldm't",replacement = "would not")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "wouldm't",replacement = "would not")
  
  df$question1 = df$question1 %>% as.character()  %>% str_replace_all(pattern = "did't",replacement = "did not")
  df$question2 = df$question2 %>% as.character()  %>% str_replace_all(pattern = "did't",replacement = "did not")
  df
  
}


lemmatize_question <- function(Question_)
{
  q1 <- Question_%>% as.character() %>% tibble(q1 = .) %>% 
    select(q1)  %>% unnest_tokens(word,q1) %>%  c() %>% unlist()%>% 
    lemmatize_words( ) %>% paste(collapse = ' ', sep = ' ')
  
}


lemmatize_and_stem_questions <- function(df)
{
  df$question1 <- df$question1%>% as.matrix()  %>% apply( MARGIN = 1 , FUN = lemmatize_question )
  df$question2 <- df$question2%>% as.matrix()  %>% apply( MARGIN = 1 , FUN = lemmatize_question )
  df$question1 <- df$question1 %>% as.matrix() %>% apply( MARGIN = 1 , FUN = stem_question )
  df$question2 <- df$question2 %>% as.matrix() %>% apply( MARGIN = 1 , FUN = stem_question )  
  df
}



###
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


get_tokenizer_keras <- function(Texts,MAX_NUM_WORDS) {
  # finally, vectorize the text samples into a 2D integer tensor
  tokenizer <- text_tokenizer(num_words=MAX_NUM_WORDS)
  tokenizer <- tokenizer %>% fit_text_tokenizer(Texts)
  # sequences <- texts_to_sequences(tokenizer, texts)
  tokenizer
}


create_embeding_index <- function(Path) {
  # fpath_table <- Path %>% file.path(., list.files(.))
  
  embeddings_index <- new.env(parent = emptyenv())
  fpath = "glove.6B/glove.6B.100d.txt"
  
  lines <- readLines(fpath)
  for (line in lines) {
    values <- strsplit(line, ' ', fixed = TRUE)[[1]]
    word <- values[[1]]
    coefs <- as.numeric(values[-1])
    embeddings_index[[word]] <- coefs
  }

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



get_sentence_word_sequence <- function(Sentence,word_vectors_glove)
{
  
  v_q <- Sentence %>% as.character() %>% tibble(v=.) %>% unnest_tokens(word,v) 
  v <- v_q$word %>%   sapply(function(x){ which( row.names(word_vectors_glove) == x)[[1]] } ) %>% na.omit 
  names(v) = c()
  v
}

if_word_exist <- function(x,word_vectors_glove)
{
  idx = which( row.names(word_vectors_glove) == x,useNames = T)
  ifelse(  !idx %>% is_empty(), T, F )  
}



words_not_in_vocab <- function(sentence,word_vectors_glove)
{
v_q <- Sentence %>% as.character() %>% tibble(v=.) %>% unnest_tokens(word,v) 
v_q <- v_q$word %>%  sapply(if_word_exist,word_vectors_glove) 
idx <- which(v_q == F)
vg[idx] %>% names() 
}



get_sentence_words <- function(Sentence,word_vectors_glove)
{
  v_q <- Sentence %>% as.character() %>% tibble(v=.) %>% unnest_tokens(word,v) 
  v <- v_q$word 
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

GLOVE_DIR <- 'glove.6B'
MAX_NUM_WORDS <- 2000000
EMBEDDING_DIM <- 100


df <- "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv" %>% 
      get_file("quora_duplicate_questions.tsv", . ) %>% 
      read_tsv() %>% 
      split_non_ascii_words_df() %>% 
      clean_appostrophes() %>% 
      mutate( question1 = (question1 %>% replace_contraction() %>% replace_symbol() %>% tolower()),
                     question2 = (question2 %>% replace_contraction() %>% replace_symbol() %>% tolower())) %>% 
      mutate( question1 = (question1 %>% gsub("\\?", " ?", .)),
                     question2 = (question2 %>% gsub("\\?", " ?", .) )) %>% 
      mutate( question1 = (question1 %>% gsub('\\.$', ' .',.)),
                     question2 = (question2 %>% gsub('\\.$', ' .',.) )) %>% 
      mutate( question1 = (question1 %>% gsub('\\:', ' :',.)),
              question2 = (question2 %>% gsub('\\:', ' :',.) )) %>% 
      mutate( question1 = (question1 %>% gsub('\\(', '( ',.)),
              question2 = (question2 %>% gsub('\\(', '( ',.) )) %>% 
      mutate( question1 = (question1 %>% gsub('\\)', ' )',.)),
              question2 = (question2 %>% gsub('\\)', ' )',.) )) %>% 
      mutate( question1 = (question1 %>% gsub('\\.', ' .',.)),
              question2 = (question2 %>% gsub('\\.', ' .',.) )) %>% 
      mutate( question1 = (question1 %>% gsub('\\,', ' ,',.)),
              question2 = (question2 %>% gsub('\\,', ' ,',.) ))  %>% 
      fix_contactions()
  
   # df1 <- df %>% 
   #    lemmatize_and_stem_questions()


df <- df %>% mutate(question1 = question1 %>% gsub("\\' ", " ",.),
                    question2 = question2 %>% gsub("\\' ", " ",.) )





df$question1 <- df$question1%>% as.matrix()  %>% apply( MARGIN = 1 , FUN = lemmatize_question )
df$question2 <- df$question2%>% as.matrix()  %>% apply( MARGIN = 1 , FUN = lemmatize_question )
df$question1 <- df$question1 %>% as.matrix() %>% apply( MARGIN = 1 , FUN = stem_question )
df$question2 <- df$question2 %>% as.matrix() %>% apply( MARGIN = 1 , FUN = stem_question )  

   

embeddings_index <- GLOVE_DIR %>% create_embeding_index()

load('word_vectors_glove.RData')


word_vectors_glove <- rbind(df$question1,df$question2) %>% 
  tokenize_text_() %>% 
  glove_create_vocab() %>% 
  glove_create_vectorizer() %>% 
  glove_create_tcm(15L) %>% 
  glove_get_word_vector(20,100) %>% 
  merge_with_pretrained_word_vector(embeddings_index)

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


question1_ = df$question1 %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()
question2_ = df$question2 %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()


word_vectors_glove_10d <- rbind(df$question1,df$question2) %>% 
  tokenize_text_() %>% 
  glove_create_vocab() %>% 
  glove_create_vectorizer() %>% 
  glove_create_tcm(15L) %>% 
  glove_get_word_vector(20,10)

question1_10d = df$question1 %>%  prepare_questions_for_keras(word_vectors_glove_10d) %>% as.matrix()
question2_10d = df$question2 %>%  prepare_questions_for_keras(word_vectors_glove_10d) %>% as.matrix()








###########

library(ggplot2)
library(plyr)
library(gridExtra)

## The Data
df <- data.frame(Type = sample(c('Male', 'Female', 'Female'), 1000, replace=TRUE),
                 Age = sample(18:60, 1000, replace=TRUE))

AgesFactor <- ordered(cut(df$Age, breaks = c(18,seq(20,60,5)), 
                          include.lowest = TRUE))

df$Age <- AgesFactor


df %>% View()

## Plotting
gg <- ggplot(data = df, aes(x=Age))

gg.male <- gg + 
  geom_bar( data=subset(df,Type == 'Male'), 
            aes( y = ..count../sum(..count..), fill = Age)) +
  scale_y_continuous('', labels = scales::percent) + 
  theme(legend.position = 'none',
        axis.title.y = element_blank(),
        plot.title = element_text(size = 11.5),
        plot.margin=unit(c(0.1,0.2,0.1,-.1),"cm"),
        axis.ticks.y = element_blank(), 
        axis.text.y = theme_bw()$axis.text.y) + 
  ggtitle("Male") + 
  coord_flip()    

gg.female <-  gg + 
  geom_bar( data=subset(df,Type == 'Female'), 
            aes( y = ..count../sum(..count..), fill = Age)) +
  scale_y_continuous('', labels = scales::percent, 
                     trans = 'reverse') + 
  theme(legend.position = 'none',
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), 
        plot.title = element_text(size = 11.5),
        plot.margin=unit(c(0.1,0,0.1,0.05),"cm")) + 
  ggtitle("Female") + 
  coord_flip() + 
  ylab("Age")

## Plutting it together
grid.arrange(gg.female,
             gg.male,
             widths=c(0.4,0.6),
             ncol=2
)




set.seed(321)
test <- data.frame(v=sample(1:20,1000,replace=T), g=c('M','F'))

require(ggplot2)
require(plyr)    
ggplot(data=test,aes(x=as.factor(v),fill=g)) + 
  geom_bar(subset=.(g=="F")) + 
  geom_bar(subset=.(g=="M"),aes(y=..count..*(-1))) + 
  scale_y_continuous(breaks=seq(-40,40,10),labels=abs(seq(-40,40,10))) + 
  coord_flip()





set.seed(1)
df0 <- data.frame(Age = factor(rep(x = 1:10, times = 2)), 
                  Gender = rep(x = c("Female", "Male"), each = 10),
                  Population = sample(x = 1:100, size = 20))

head(df0)

library(ggplot2)
df0 %>% head
ggplot(data = df0, 
       mapping = aes(x = term, fill = WordsCount, 
                     y = ifelse(WordsCount == "Words_in_question1", 
                                yes = -term_count, no = term_count))) +
  geom_bar(stat = "identity") +
  scale_y_continuous(limits = max(df0$term_count) * c(-1,1)) +
  labs(y = "Words Count", x = "Words") +
  
  coord_flip()


library(reshape2)

# example data frame
x = data.frame(
  id   = c(1, 1, 2, 2),
  blue = c(1, 0, 1, 0),
  red  = c(0, 1, 0, 1)
)



# collapse the data frame
melt(data = x, id.vars = "id", measure.vars = c("blue", "red"))



library(ggplot2)

# create a dataset
specie=c(rep("sorgho" , 3) , rep("poacee" , 3) , rep("banana" , 3) , rep("triticum" , 3) )
condition=rep(c("normal" , "stress" , "Nitrogen") , 4)
value=abs(rnorm(12 , 0 , 15))
data=data.frame(specie,condition,value)



condition=c("Duplicated" , "Non-duplicated" ) 
value=c(80,160)
data=data.frame(condition,value)




# Stacked
ggplot(data, aes(fill=condition, y=value, x=2)) + 
  geom_bar( stat="identity",position = "dodge")+
  coord_flip()





