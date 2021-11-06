library(readr)
library(keras)
library(purrr)

rm(list = ls(all=TRUE) )
FLAGS <- flags(
  flag_integer("vocab_size", 94561),
  flag_integer("max_len_padding", 20),
  flag_integer("embedding_size", 100),
  flag_numeric("regularization", 0.0001),
  flag_integer("seq_embedding_size", 512)
)

# Downloading Data --------------------------------------------------------
load('df.RData')



word_vectors_glove %>% dim

load('question1.RData')
load('question2.RData')
load('word_vectors_glove.RData')
question1_ = question1[,493:512]
question2_ = question2[,493:512]
question2_ %>% dim()

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

model6 <- function(FLAGS) {
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


model7 <- function(FLAGS) {
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
model8 <- function(FLAGS) {
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

model9 <- function(FLAGS) {
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


model10 <- function(FLAGS) {
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

model <- model1(FLAGS) 
print(model)
model <- model2(FLAGS) 
model <- model3(FLAGS) # 1
model <- model4(FLAGS) 
model <- model5(FLAGS) 
model <- model6(FLAGS) ## 1
model <- model7(FLAGS)
model <- model8(FLAGS)
model <- model9(FLAGS) ## 
model <- model10(FLAGS) ##

# Model Fitting -----------------------------------------------------------

train_model <- function(model,question1_,question2_)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(question1_), size = 0.1*nrow(question1_))
  
 history <- model %>%
   keras:: fit(
      list(question1_[-val_sample,], question2_[-val_sample,]),
      df$is_duplicate[-val_sample], 
      batch_size = 128, 
      epochs = 30, 
      validation_data = list(
        list(question1_[val_sample,], question2_[val_sample,]), df$is_duplicate[val_sample]
      ),
      callbacks = list(
        callback_early_stopping(patience = 5),
        callback_reduce_lr_on_plateau(patience = 3)
      )
    )  
 prediction_ = list(question1_[val_sample,], question2_[val_sample,]) %>% predict(model,.)
 
 
 list(history= history, model = model, validation = data.frame(prediction = prediction_ , val_sample = val_sample ))


}

model_hist <- FLAGS %>% model10() %>% train_model(question1_,question2_)
model_hist$model %>% save_model_hdf5('model10.hdf5',overwrite = TRUE,include_optimizer = TRUE)
saveRDS(model_hist$history,file='histmodel10.rds')
histmodel10 = readRDS('histmodel10.rds')
plot(histmodel10)



# Autoencoders Models ---------------


FLAGS <- flags(
  vocab_size = 94561,
  max_len_padding=  20,
  embedding_size =  100,
  regularization =  0.0001,
  seq_embedding_size = 512
)

question = question1_

train_autoencoder <- function(autoencoder,question ) {
  start = 0
  end   = 0
  nrow_df = nrow(question )
  pkg_size = 2000
  V = (nrow_df/pkg_size) %>% round() + 1
  
  histories = list()
  

  for( i in seq_len(10))
  {
    start = ( i - 1 ) * pkg_size + 1 
    end = ifelse( end >= ( nrow_df - pkg_size ), nrow_df, (i * pkg_size))
    nnet_data_input         = question[start:end,]
    
    random_samples <- sample.int(pkg_size,size= 0.1 * pkg_size)
    
    nnet_data_input_train    = nnet_data_input[-random_samples,]
    nnet_data_input_valid    = nnet_data_input[random_samples,]
    
    # output_embedding <- model_embeding %>% predict(nnet_data_input)
    
    data_train <- autoencoder$embeding_model %>% predict(nnet_data_input_train)
    data_valid <- autoencoder$embeding_model %>% predict(nnet_data_input_valid)
  
    
    history = autoencoder$autoencoder_model %>%
      fit(
        data_train,
        data_train,
        batch_size = 1, 
        epochs = 5,
        callbacks = list(
          callback_early_stopping(patience = 5),
          callback_reduce_lr_on_plateau(patience = 3)
          
        ),
        validation_data = list(data_valid,data_valid) 
      )
   if( exists('history_prev') ) 
    {
      history_prev$params$epochs         = history_prev$params$epochs + history$params$epochs
      history_prev$metrics$val_acc       = cbind(history_prev$metrics$val_acc  , history$metrics$val_acc)
      history_prev$metrics$acc           = cbind(history_prev$metrics$acc      , history$metrics$acc)
      history_prev$metrics$lr            = cbind(history_prev$metrics$lr       , history$metrics$lr)
      history_prev$metrics$val_loss      = cbind(history_prev$metrics$val_loss , history$metrics$val_loss)
      history_prev$metrics$loss          = cbind(history_prev$metrics$loss     , history$metrics$loss)
   } else history_prev <- history
    
  }
  history_prev %>% plot()
}





autoencoder1 <- function(FLAGS) {
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding))
  
  # embedding model
  embedding <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  preds_embedding  <- sequence_input %>% embedding
  embeding_model   <- keras_model(sequence_input, preds_embedding)
  
  
  seq_emb1 <- layer_lstm(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  seq_emb2 <- layer_lstm(
    units =100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T  )
  
  dense_100_tanh1 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_100_tanh2 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_020_relu1 <- layer_dense( units = 20, activation = 'tanh')
  dense_020_relu2 <- layer_dense( units = 20, activation = 'tanh')
  
  
  # Encoder Model
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding,FLAGS$embedding_size))
  
  encoder <- sequence_input %>% seq_emb1 %>% dense_100_tanh1 %>%  dense_020_relu1
  encoder_model <- keras_model(sequence_input, encoder)
  
  # Decoder Model
  input_decoder <- layer_input(shape = c(1,20) )
  decoder <- input_decoder %>% dense_020_relu2 %>% dense_100_tanh2 %>% seq_emb2
  decoder_model <- keras_model(input_decoder , decoder)
  
  
  # Autoencoder Model
  autoencoder <- sequence_input %>% encoder_model %>%   decoder_model
  autoencoder_model <- keras_model(sequence_input , autoencoder)
  autoencoder_model %>% 
    compile(
      optimizer='adadelta', 
      loss='mse',
      metrics = c('accuracy')  )
  list(autoencoder_model=autoencoder_model,
       decoder_model=decoder_model,
       encoder_model=encoder_model,
       embeding_model = embeding_model)
}



autoencoder2 <- function(FLAGS) {
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding))
  
  # embedding model
  embedding <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  preds_embedding  <- sequence_input %>% embedding
  embeding_model   <- keras_model(sequence_input, preds_embedding)
  
  
  seq_emb1 <- layer_lstm(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  seq_emb2 <- layer_lstm(
    units =100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T  
  )
  
  dense_100_tanh1 <- layer_dense( units = 100, activation = 'relu')
  dense_100_tanh2 <- layer_dense( units = 100, activation = 'relu')
  dense_020_relu1 <- layer_dense( units = 20, activation = 'tanh')
  dense_020_relu2 <- layer_dense( units = 20, activation = 'tanh')
  dense_001_tanh1  <- layer_dense( units = 1 , activation = 'tanh')
  dense_001_tanh2  <- layer_dense( units = 1 , activation = 'tanh')  
  # Encoder Model
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding,FLAGS$embedding_size))
  encoder <- sequence_input %>% seq_emb1 %>% dense_100_tanh1 %>%  dense_020_relu1 %>% dense_001_tanh1
  encoder_model <- keras_model(sequence_input, encoder)
  

  
  
  # Decoder Model
  input_decoder <- layer_input(shape = c(20,1) )
  decoder <- input_decoder %>% dense_001_tanh2 %>% dense_020_relu2 %>% dense_100_tanh2  %>% seq_emb2 
  decoder_model <- keras_model(input_decoder , decoder)
  
  
  # v1 = embeding_model %>% predict(question1_[1:2,])
  # v1 %>% dim
  # 
  # v2 = encoder_model %>% predict(v1)
  # v2 %>% dim
  # 
  # v2 %>% dim()
  # 
  # v3 = decoder_model %>% predict(v2)
  # v3 %>% dim()
  
  # Autoencoder Model
  autoencoder <- sequence_input %>% encoder_model %>%   decoder_model
  autoencoder_model <- keras_model(sequence_input , autoencoder)
  autoencoder_model %>% 
    compile(
      optimizer='adadelta', 
      loss='mse',
      metrics = c('accuracy')  )
  list(autoencoder_model=autoencoder_model,
       decoder_model=decoder_model,
       encoder_model=encoder_model,
       embeding_model = embeding_model)
}





autoencoder3 <- function(FLAGS) {
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding))
  
  # embedding model
  embedding <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  preds_embedding  <- sequence_input %>% embedding
  embeding_model   <- keras_model(sequence_input, preds_embedding)
  
  
  seq_emb1 <- layer_lstm(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  seq_emb2 <- layer_lstm(
    units =100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T  )
  
  
  seq_gru1 <- layer_gru(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  seq_gru2 <- layer_gru(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  
  
  dense_100_tanh1 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_100_tanh2 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_020_relu1 <- layer_dense( units = 10, activation = 'tanh')
  dense_020_relu2 <- layer_dense( units = 10, activation = 'tanh')
  
  
  # Encoder Model
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding,FLAGS$embedding_size))
  encoder <- sequence_input %>% seq_gru1 %>% dense_100_tanh1 %>%  dense_020_relu1
  encoder_model <- keras_model(sequence_input, encoder)
  
  # Decoder Model
  input_decoder <- layer_input(shape = c(1,10) )
  decoder <- input_decoder %>% dense_020_relu2 %>% dense_100_tanh2 %>% seq_gru2
  decoder_model <- keras_model(input_decoder , decoder)
  
  
  # Autoencoder Model
  autoencoder <- sequence_input %>% encoder_model  %>% decoder_model
  autoencoder_model <- keras_model(sequence_input , autoencoder)
  autoencoder_model %>% 
    compile(
      optimizer='adadelta', 
      loss='mse',
      metrics = c('accuracy')  )
  
  list(autoencoder_model=autoencoder_model,
       decoder_model=decoder_model,
       encoder_model=encoder_model,
       embeding_model = embeding_model)
}




autoencoder4 <- function(FLAGS) {
  
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding))
  
  # embedding model
  embedding <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  preds_embedding  <- sequence_input %>% embedding
  embeding_model   <- keras_model(sequence_input, preds_embedding)
  
  
  seq_emb1 <- layer_lstm(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  seq_emb2 <- layer_lstm(
    units =100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T  )
  
  
  seq_gru <- layer_gru(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  conv1d_100_1         <- layer_conv_1d(filters = 100, kernel_size =  4, activation='relu', padding='same' ) 
  conv1d_100_2         <- layer_conv_1d(filters = 100, kernel_size =  4, activation='relu', padding='same' )
  max_pooling_1d       <- layer_max_pooling_1d(pool_size = 4)
  upsampling           <- layer_upsampling_1d( size = 4)  
  
  dense_100_tanh1 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_100_tanh2 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_020_relu  <- layer_dense( units = 20, activation = 'tanh')
  
  
  # Encoder Model
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding,FLAGS$embedding_size))
  encoder        <- sequence_input %>% conv1d_100_1 %>% max_pooling_1d 
  encoder_model  <- keras_model(sequence_input, encoder)
  
  # Decoder Model
  input_decoder <- layer_input(shape = c(5,100) )
  decoder <- input_decoder %>% upsampling %>% conv1d_100_2      
  decoder_model <- keras_model(input_decoder , decoder)
  
  
  # Autoencoder Model
  autoencoder <- sequence_input %>% encoder_model %>%   decoder_model
  autoencoder_model <- sequence_input  %>% keras_model(autoencoder)
  autoencoder_model %>% 
    compile(
      optimizer='adadelta', 
      loss='mse',
      metrics = c('accuracy')  )
  
  
  # 
  #  v1 = embeding_model %>% predict(question1_[1:2,])
  #  v1 %>% dim
  # 
  #  v2 = encoder_model %>% predict(v1)
  #  v2 %>% dim
  # 
  #  v3 = decoder_model %>% predict(v2)
  #  v3 %>% dim()
  # 
  # v4 = autoencoder_model %>% predict(v1)
  # 
  # v4 %>% dim()
  
  list(autoencoder_model=autoencoder_model,
       decoder_model=decoder_model,
       encoder_model=encoder_model,
       embeding_model = embeding_model)
}


autoencoder5 <- function(FLAGS) {
  
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding))
  
  # embedding model
  embedding <- layer_embedding(
    input_dim = word_vectors_glove %>% nrow() ,
    # regularizer_l2(l = FLAGS$regularization)
    output_dim = FLAGS$embedding_size,
    weights = list(word_vectors_glove),
    input_length = FLAGS$max_len_padding,
    trainable = FALSE
  )
  
  preds_embedding  <- sequence_input %>% embedding
  embeding_model   <- keras_model(sequence_input, preds_embedding)
  
  
  seq_emb1 <- layer_lstm(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  seq_emb2 <- layer_lstm(
    units =100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T  )
  
  
  seq_gru <- layer_gru(
    units = 100, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    return_sequences=T
  )
  
  conv1d_100_1         <- layer_conv_1d(filters = 100, kernel_size =  4, activation='relu', padding='same' ) 
  conv1d_100_2         <- layer_conv_1d(filters = 100, kernel_size =  4, activation='relu', padding='same' ) 
  conv1d_100_3         <- layer_conv_1d(filters = 100, kernel_size =  4, activation='relu', padding='same' ) 
  conv1d_100_4         <- layer_conv_1d(filters = 100, kernel_size =  4, activation='relu', padding='same' ) 
  
  max_pooling_1d     <- layer_max_pooling_1d(pool_size = 4)
  upsampling         <- layer_upsampling_1d( size = 4)  
  

  
  max_pooling_1d_5     <- layer_max_pooling_1d(pool_size = 5)
  upsampling_5         <- layer_upsampling_1d( size = 5)  
  
  
  dense_100_tanh1 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_100_tanh2 <- layer_dense( units = FLAGS$embedding_size, activation = 'relu')
  dense_020_relu  <- layer_dense( units = 20, activation = 'tanh')
  
  # Encoder Model
  sequence_input <- layer_input(shape = list(FLAGS$max_len_padding,FLAGS$embedding_size))
  encoder        <- sequence_input %>% conv1d_100_1 %>% max_pooling_1d_5 %>% conv1d_100_2 %>% max_pooling_1d 
  encoder_model  <- keras_model(sequence_input, encoder)
  
  # Decoder Model
  input_decoder <- layer_input(shape = c(1,100) )
  decoder <- input_decoder%>%    upsampling_5  %>% conv1d_100_3   %>% conv1d_100_4 %>%    upsampling  
  decoder_model <- keras_model(input_decoder , decoder)
  
  
  # Autoencoder Model
  autoencoder <- sequence_input %>% encoder_model %>%   decoder_model
  autoencoder_model <- sequence_input  %>% keras_model(autoencoder)
  autoencoder_model %>% 
    compile(
      optimizer='adadelta', 
      loss='mse',
      metrics = c('accuracy')  )
  
  
  
  #  v1 = embeding_model %>% predict(question1_[1:2,])
  #  v1 %>% dim
  # 
  #  v2 = encoder_model %>% predict(v1)
  #  v2 %>% dim
  # 
  #  v3 = decoder_model %>% predict(v2)
  #  v3 %>% dim()
  # 
  # v4 = autoencoder_model %>% predict(v1)
  # v4 %>% dim()
  
  list(autoencoder_model=autoencoder_model,
       decoder_model=decoder_model,
       encoder_model=encoder_model,
       embeding_model = embeding_model)
}


# Tests ----- 
autoencoder = FLAGS %>% autoencoder1() 
FLAGS %>% autoencoder1()%>% train_autoencoder(question1_)
FLAGS %>% autoencoder2()%>% train_autoencoder(question1_)
FLAGS %>% autoencoder3()%>% train_autoencoder(question1_)
FLAGS %>% autoencoder4()%>% train_autoencoder(question1_)
FLAGS %>% autoencoder5()%>% train_autoencoder(question1_ )


autoencoder = FLAGS %>% autoencoder5() 
autoencoder$encoder_model %>% print()
autoencoder$decoder_model %>% print()
autoencoder$autoencoder_model %>% print()



# Autoencoder for wrod vektor, and replacment of embedding layer 

train_word_vec_model<- function(model)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(model$word_vectors_glove), size = 0.1*nrow(model$word_vectors_glove))
  
  history <- model$autoencoder_model  %>%
    fit(
      model$word_vectors_glove[-val_sample,],
      model$word_vectors_glove[-val_sample,], 
      batch_size = 1000, 
      epochs = 90, 
      validation_data = list(
        model$word_vectors_glove[val_sample,], model$word_vectors_glove[val_sample,]
      ),
      callbacks = list(
       callback_early_stopping(patience = 5),
        callback_reduce_lr_on_plateau(patience = 3)
      )
    ) 
  
  list(history= history, model= model)

}


word_vect_autoencoder1 <- function(word_vectors_glove,config_params)
{ 
  dim_word_vec = word_vectors_glove %>% ncol
  sequence_input <- layer_input(shape = list( dim_word_vec ))
  

  dense_500_tanh1 <- layer_dense( units = 500, activation = 'tanh')
  dense_500_tanh2 <- layer_dense( units = 500, activation = 'tanh')
  
  dense_250_tanh1 <- layer_dense( units = 250, activation = 'tanh')
  dense_250_tanh2 <- layer_dense( units = 250, activation = 'tanh')
    
  dense_100_tanh1 <- layer_dense( units = dim_word_vec, activation = 'tanh')
  dense_100_tanh2 <- layer_dense( units = dim_word_vec, activation = 'tanh')
 
  dense_050_tanh1 <- layer_dense( units = 50, activation = 'tanh')
  dense_050_tanh2 <- layer_dense( units = 50, activation = 'tanh')
  

  dense_024_tanh1 <- layer_dense( units = 24, activation = 'tanh')
  dense_024_tanh2 <- layer_dense( units = 24, activation = 'tanh')

  dense_020_tanh1 <- layer_dense( units = 20, activation = 'tanh')
  dense_020_tanh2 <- layer_dense( units = 20, activation = 'tanh')
  
  dense_012_tanh1 <- layer_dense( units = 12, activation = 'tanh')
  dense_012_tanh2 <- layer_dense( units = 12, activation = 'tanh')
  
  
  dense_010_tanh1 <- layer_dense( units = 10, activation = 'tanh')
  dense_010_tanh2 <- layer_dense( units = 10, activation = 'tanh')
  
  dense_005_tanh1 <- layer_dense( units = 5, activation = 'tanh')
  dense_005_tanh2 <- layer_dense( units = 5, activation = 'tanh')
  
  
  dense_004_tanh1 <- layer_dense( units = 4, activation = 'tanh')
  dense_004_tanh2 <- layer_dense( units = 4, activation = 'tanh')
  
  dense_002_tanh1 <- layer_dense( units = 2, activation = 'tanh')
  dense_002_tanh2 <- layer_dense( units = 2, activation = 'tanh')
  
  dense_001_tanh1 <- layer_dense( units = 1, activation = 'tanh')
  dense_001_tanh2 <- layer_dense( units = 1, activation = 'tanh')
  
  
  # Encoder Model
  encoder <- sequence_input %>%  
    dense_100_tanh1 %>% dense_500_tanh1 %>% dense_250_tanh1 %>% 
    dense_050_tanh1 %>% dense_024_tanh1 %>% 
    dense_020_tanh1 %>% dense_012_tanh1 %>% dense_010_tanh1 %>% 
    dense_005_tanh1 %>% dense_004_tanh1 %>% dense_001_tanh1
    

  encoder_model <- sequence_input %>% keras_model(encoder)
  

  # Decoder Model
  input_decoder <- layer_input(shape = 1 )
  decoder <- input_decoder %>%  
    dense_001_tanh2 %>% dense_002_tanh2 %>% dense_004_tanh2 %>% 
    dense_005_tanh2 %>% dense_010_tanh2 %>% dense_012_tanh2 %>% 
    dense_020_tanh2 %>% dense_024_tanh2 %>% dense_050_tanh2 %>% 
    dense_250_tanh2 %>% dense_500_tanh2 %>% dense_100_tanh2 
  decoder_model <- input_decoder %>%  keras_model( decoder)
  
    # Autoencoder Model
  autoencoder <- sequence_input %>% encoder_model  %>% decoder_model
  autoencoder_model <- sequence_input %>%  keras_model(autoencoder)
  autoencoder_model %>% 
    compile(
      optimizer='adadelta', 
      loss='mse',
      metrics = c('accuracy')  )
  
  list(autoencoder_model=autoencoder_model,
       decoder_model=decoder_model,
       encoder_model=encoder_model,
       word_vectors_glove = word_vectors_glove)
  
   # v1 = encoder_model %>% predict(word_vectors_glove[1:2,1:50])
   # v1 %>% dim
   # v2 = decoder_model %>% predict(v1)
   # v2 %>% dim
  
  # v3 = autoencoder %>% predict(word_vectors_glove[1:2,])
  # v3 %>% dim
}


# one dimention word vector with encoder ----
d_tsne_1 %>% word_vect_autoencoder1() %>% train_word_vec_model()

autoencoder_hist <- word_vectors_glove %>% word_vect_autoencoder1() %>% train_word_vec_model()
plot(autoencoder_hist$history)
word_vec1d_encoder = autoencoder_hist$model$encoder_model %>% predict(word_vectors_glove )
word_vec1d_encoder <- word_vec1d_encoder %>% as.data.frame %>% mutate(term = row.names(word_vectors_glove ))
colnames(word_vec1d_encoder ) = c('value','term')


serach_similar_words_1d <- function(Word_) {
  sorted_word_vecs <- word_vec1d_encoder %>% arrange( desc(value))
  word_indx = which(sorted_word_vecs$term == Word_)%>% as.numeric()
  
  sorted_word_vecs[(word_indx -10):(word_indx+ 10 ), ]
}
serach_similar_words_1d('france') %>% View()
which(duplicated(word_vec1d_encoder$term))

word_vec1d_encoder$term %>% class()



#  one decimal value word with R-TSNE --- 
library(Rtsne)
get_word_vec_1d <- function(word_vectors_glove)
{
  tsne_model_1 = word_vectors_glove[-c(94561:94562),]  %>%  as.matrix( ) %>%  Rtsne(. , check_duplicates=F, pca=TRUE, perplexity=3, theta=0.5, dims=1)
  d_tsne_1d     = data.frame(value = tsne_model_1$Y, term =row.names(word_vectors_glove[-c(94561:94562),])  ) 
  colnames(d_tsne_1d) = c('value','term')
  d_tsne_1d$term = as.character(d_tsne_1d$term)
  # row.names(d_tsne_1d) =  row.names(word_vectors_glove[-c(94561:94562),])
  save(d_tsne_1d,file='d_tsne_1d.RData')
  save(tsne_model_1, file = 'tsne_model_1.RData')
  d_tsne_1d$value = scale(d_tsne_1d$value )/max(d_tsne_1d$value )
  d_tsne_1d
}

word_vectors_glove1D <- word_vectors_glove[-c(94561:94562),]  %>% get_word_vec_1d()


get_sentence_word_sequence_1d <- function(Sentence,word_vec_1d)
{
  v_q <- Sentence %>% as.character() %>% tibble(v=.) %>% select(v)   %>% unnest_tokens(word,v) 
  v <- v_q$word %>%   sapply(function(x){ which( word_vec_1d$term == x)[[1]] } ) %>% na.omit 
  names(v) = c()
  v
}
df[1,]$question1 %>% get_sentence_word_sequence_1d(word_vec_1d)


prepare_questions_for_keras_1d <- function(Questions,word_vec_1d)
{
  n = word_vec_1d %>% nrow()
  v <- Questions %>% sapply(FUN = get_sentence_word_sequence_1d,word_vec_1d) %>% unname()
  v <- v %>% pad_sequences( maxlen = 30, value = n )
  dimv <- v %>% dim
  v   <- matrix(v %>% word_vec_1d[.,'value'] ,dimv[1],dimv[2])
  v
}

question1_1d = df$question1 %>% prepare_questions_for_keras_1d(word_vec_1d)
question2_1d = df$question2 %>% prepare_questions_for_keras_1d(word_vec_1d)


covq1q2 = question1_1d %>%  apply(1,cov(.,question2_1d))

save(question2_1d ,file='question2_1d.RData')
save(question1_1d ,file='question1_1d.RData')





plot_gama_ddensity <- function(numb = 1000000) {
  x <- rgamma(numb, shape = 3, scale = 0.2)
  den <- density(x)
  dat <- data.frame(x = den$x, y = den$y)
  # Plot density as points
  ggplot(data = dat, aes(x = x, y = y)) + 
    geom_point(size = 3) +
    theme_classic()
}

plot_gama_ddensity(numb = 50)




model1d_1 <- function(FLAGS,question1,question2)
{
  input1 <- layer_input(shape = c(30))
  input2 <- layer_input(shape = c(30))

  seq_reshape1 <- layer_reshape(target_shape = c(1,30))
  seq_reshape2 <- layer_reshape(target_shape = c(1,30))
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)      
  
  
  out <- layer_dot(list(vector1, vector2), axes = 1) %>% dense_100_relu %>% 
    layer_dropout( rate = 0.1)%>% 
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
  
  # v1 = model %>% predict(list(question1_1d[1:2,],question2_1d[1:2,]))
  # v1 %>% dim
  
}

model1d_2 <- function(FLAGS,question1,question2)
{
  input1 <- layer_input(shape = c(30))
  input2 <- layer_input(shape = c(30))
  
  seq_reshape1 <- layer_reshape(target_shape = c(30,1))
  seq_reshape2 <- layer_reshape(target_shape = c(30,1))
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)      
  
  
  out <- layer_dot(list(vector1, vector2), axes = 1) %>% dense_100_relu %>% 
    layer_dropout( rate = 0.1)%>% 
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
  
  # v1 = model %>% predict(list(question1_1d[1:2,],question2_1d[1:2,]))
  # v1 %>% dim
  
}



model1d_3 <- function(FLAGS,question1,question2)
{
  input1 <- layer_input(shape = c(30))
  input2 <- layer_input(shape = c(30))
  
  seq_reshape1 <- layer_reshape(target_shape = c(1,30))
  seq_reshape2 <- layer_reshape(target_shape = c(1,30))
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.4)
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.4)
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)      
  
  
  out <- list(vector1, vector2) %>% layer_concatenate( ) %>% dense_100_relu %>% 
    layer_dropout( rate = 0.1)%>% 
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
  
  # v1 = model %>% predict(list(question1_1d[1:2,],question2_1d[1:2,]))
  # v1 %>% dim
  
}


model1d_4 <- function(FLAGS,question1,question2)
{
  input1 <- layer_input(shape = c(30))
  input2 <- layer_input(shape = c(30))
  
  seq_reshape1 <- layer_reshape(target_shape = c(1,30))
  seq_reshape2 <- layer_reshape(target_shape = c(1,30))
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.2)
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization),
    dropout = 0.2)
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
 
  dense_100_relu1     <- layer_dense( units = 100, activation = 'tanh') 
  dense_100_relu2     <- layer_dense( units = 100, activation = 'relu')
  dense_100_relu3     <- layer_dense( units = 100, activation = 'tanh')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)      
  
  
  out <- list(vector1, vector2) %>% layer_concatenate( ) %>% dense_100_relu1 %>% 
    layer_dropout( rate = 0.2) %>% dense_100_relu2 %>% layer_dropout( rate = 0.2) %>%  dense_100_relu3 %>% 
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
  
  # v1 = model %>% predict(list(question1_1d[1:2,],question2_1d[1:2,]))
  # v1 %>% dim
  
}



model1d_5 <- function(FLAGS,question1,question2)
{
  input1 <- layer_input(shape = c(30))
  input2 <- layer_input(shape = c(30))
  
  seq_reshape1 <- layer_reshape(target_shape = c(30,1))
  seq_reshape2 <- layer_reshape(target_shape = c(30,1))
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
  
  dense_100_relu1     <- layer_dense( units = 100, activation = 'tanh') 
  dense_100_relu2     <- layer_dense( units = 100, activation = 'relu')
  dense_100_relu3     <- layer_dense( units = 100, activation = 'tanh')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)     
  
  
  out <- layer_dot(list(vector1, vector2), axes = 1) %>% dense_100_relu1 %>% 
    layer_dropout( rate = 0.1) %>% dense_100_relu2 %>% layer_dropout( rate = 0.1) %>%  dense_100_relu3 %>% 
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
  
  # v1 = model %>% predict(list(question1_1d[1:2,],question2_1d[1:2,]))
  # v1 %>% dim
  
}





question1_1d[1,] %>% length()


model <- FLAGS %>% model1d_4(question1_1d,question2_1d) %>% train_model(question1_1d,question2_1d)

x <- c(2,8,11,19)
x <- data.frame(x,1) ## 1 is your "height"
plot(seq(1:30),question1_1d[2453,], type='l')
plot(seq(1:30),question2_1d[2453,], type='l')



plot_questions_1d_project <- function(idx) {
  q1d_graph_data = data.frame(word_value= question1_1d[idx,16:30],sequence= seq(15),field = 'question1')
  q1d_graph_data = q1d_graph_data %>% rbind(data.frame(word_value= question2_1d[idx,16:30],sequence= seq(15),field = 'question2'))
  p= q1d_graph_data %>% ggplot( aes(x=sequence, y=word_value, colour=field)) + geom_line()
  p
  # q1d_graph_data
}

v1 = which(df$is_duplicate == 0 )[100:111] %>% lapply(plot_questions_1d_project)
multiplot(v1[[1]],v1[[2]],v1[[3]],v1[[4]],v1[[5]],v1[[5]],v1[[6]],v1[[7]],v1[[8]],v1[[9]],v1[[10]],v1[[11]],cols=3)



serach_similar_words_1d <- function(Word_) {
  sorted_word_vecs <- word_vec_1d %>% arrange( desc(value))
  word_indx = which(sorted_word_vecs$term == Word_)%>% as.numeric()
  
  sorted_word_vecs[(word_indx -10):(word_indx+ 10 ), ]
}

serach_similar_words_1d('safety') %>% View()

heatmap(question1_1d[idx,16:30], question2_1d[idx,16:30])

# Two Dimension word vector Use TSNE to Reduce the dimension ----

library(Rtsne)
get_word_vec_2d <- function(word_vectors_glove)
{
  tsne_model_2 = word_vectors_glove[-c(94561:94562),]  %>%  as.matrix( ) %>%  Rtsne(. , check_duplicates=F, pca=TRUE, perplexity=3, theta=0.5, dims=2)
  d_tsne_2d     = data.frame(value1 = tsne_model_1$Y[,1], value2 = tsne_model_1$Y[,2], term =row.names(word_vectors_glove[-c(94561:94562),])  ) 
  colnames(d_tsne_2d) = c('value1','value2','term')
  d_tsne_2d$term = as.character(d_tsne_2d$term)
  # row.names(d_tsne_1d) =  row.names(word_vectors_glove[-c(94561:94562),])

  d_tsne_2d$value1 = scale(d_tsne_2d$value1 )/max(d_tsne_2d$value1 )
  d_tsne_2d$value2 = scale(d_tsne_2d$value2 )/max(d_tsne_2d$value2 )
  save(d_tsne_2d,file='d_tsne_2d.RData')
  save(tsne_model_2, file = 'tsne_model_2.RData')
  d_tsne_2d
}
load('word_vectors_glove.RData')
word_vectors_glove2D <- word_vectors_glove[-c(94561:94562),]  %>% get_word_vec_2d()

get_sentence_word_sequence_2d <- function(Sentence,word_vectors_glove2D)
{
  v_q <- Sentence %>% as.character() %>% tibble(v=.)  %>% unnest_tokens(word,v) 
  v <- v_q$word %>%   sapply(function(x){ which( word_vectors_glove2D$term == x)[[1]] } ) %>% na.omit 
  names(v) = c()
  v
}

df[1,]$question1 %>% get_sentence_word_sequence_2d(word_vectors_glove2D)


prepare_questions_for_keras_2d <- function(Questions,word_vectors_glove2D)
{ 
  n_q = Questions %>% length()
  n = word_vectors_glove2D %>% nrow()
  v <- Questions %>% sapply(FUN = get_sentence_word_sequence_2d,word_vectors_glove2D) %>% unname()
  v <- v %>% pad_sequences( maxlen = 30, value = n )
  dimv <- v %>% dim
  v1 = v %>% word_vectors_glove2D[.,-3] %>% as.matrix() %>% t()

  
  dim(v1) = c(n_q,dimv[2],2)
  v1 
}

question1_2d = df$question1 %>% prepare_questions_for_keras_2d(word_vectors_glove2D)
question2_2d = df$question2 %>% prepare_questions_for_keras_2d(word_vectors_glove2D)


save(question2_2d ,file='question2_2d.RData')
save(question1_2d ,file='question1_2d.RData')

load('question2_2d.RData')
question2_2d[1,,]

load('question1.RData')
load('question2.RData')

model2d_1 <- function(FLAGS,question1,question2)
{
  input1 <- layer_input(shape = c(30,2))
  input2 <- layer_input(shape = c(30,2))
  # 
   seq_reshape1 <- layer_reshape(target_shape = c(60,1))
   seq_reshape2 <- layer_reshape(target_shape = c(60,1))
  # 
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_     <- layer_flatten(input_shape = 512)      
  
  
  out <- layer_dot(list(vector1, vector2), axes = 1) %>% dense_100_relu %>% 
    layer_dropout( rate = 0.1)%>% 
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
  
   v1 = model %>% predict(list(question1_2d[1:200,,],question2_2d[1:200,,]))
   v1 %>% dim
  
}



train_model_2d <- function(model,question1_,question2_)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(question1_), size = 0.1*nrow(question1_))
  
  history <- model %>%
    keras::fit(
      list(question1_[-val_sample,,], question2_[-val_sample,,]),
      df$is_duplicate[-val_sample], 
      batch_size = 128, 
      epochs = 30, 
      validation_data = list(
         list(question1_[val_sample,,], question2_[val_sample,,]), df$is_duplicate[val_sample]
       ) ,
       callbacks = list(
         callback_early_stopping(patience = 5),
         callback_reduce_lr_on_plateau(patience = 3)
       )
    )  
  prediction_ = list(question1_[val_sample,,], question2_[val_sample,,]) %>% predict(model,.)
  
  
  list(history= history, model = model, validation = data.frame(prediction = prediction_ , val_sample = val_sample ))
  
  
}

model <- FLAGS %>% model2d_1(question1_2d,question2_2d) %>% train_model_2d(question1_2d,question2_2d)



# Autoencoder classifier   ----- 

get_sentence_word_sequence <- function(Sentence,word_vectors_glove)
{
  
  v_q <- Sentence %>% as.character() %>% tibble(v=.)    %>% unnest_tokens(word,v) 
  v <- v_q$word %>%   sapply(function(x){ which( row.names(word_vectors_glove) == x)[[1]] } ) %>% na.omit 
  names(v) = c()
  v
  
}


idx = which(df$is_duplicate == 0 )
leng_idx = idx %>% length()
df_autoencoder1 <- cbind(question1 = df$question1[1:leng_idx],question2 = df$question1[1:leng_idx],is_duplicate = 1)
df_autoencoder <- cbind( question1 = df$question1[idx],question2 = df$question2[idx],is_duplicate = 0 )
rm(df_autoencoder1)

df_autoencoder <- rbind(df_autoencoder1,df_autoencoder) %>% as.data.frame()
load('word_vectors_glove.RData')

question1_auto_enc = df_autoencoder$question1 %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()
question2_auto_enc = df_autoencoder$question2 %>%  prepare_questions_for_keras(word_vectors_glove) %>% as.matrix()

save(question1_auto_enc,file="question1_auto_enc.RData")
save(question2_auto_enc,file="question2_auto_enc.RData")


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
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  vector1 <- input1 %>% embedding1() %>%
    seq_gru1()
  vector2 <- input2 %>% embedding2() %>%
    seq_gru2()
  
  dense_100_relu     <- layer_dense( units = 100, activation = 'relu')
  dense_25_relu      <- layer_dense( units = 25 , activation = 'relu')
  dense_001_softmax  <- layer_dense( units = 1  , activation = "sigmoid")
  
  
  out <- layer_dot(list(vector1, vector2), axes = 1) %>% 
    dense_100_relu %>% 
    dense_25_relu  %>% 
    # layer_dropout( rate = 0.5)%>% 
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
df1$is_duplicate =df1$is_duplicate %>% as.numeric()

train_model <- function(model,question1_,question2_,df)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(question1_), size = 0.1*nrow(question1_))
  
  history <- model %>%
    keras:: fit(
      list(question1_[-val_sample,], question2_[-val_sample,]),
      df$is_duplicate[-val_sample], 
      batch_size = 128, 
      epochs = 30,
      validation_data = list( list(question1_[val_sample,], question2_[val_sample,]), df$is_duplicate[val_sample]),
      callbacks = list( callback_early_stopping(patience = 5), callback_reduce_lr_on_plateau(patience = 3) )
    )  
  prediction_ = list(question1_[val_sample,], question2_[val_sample,]) %>% predict(model,.)
  list(history= history, model = model, validation = data.frame(prediction = prediction_ , val_sample = val_sample ))
}

sz_ = 100000
rm(df1)
df1 = data.frame(is_duplicate = rep.int(1,sz_) )
df1 = rbind(df1, data.frame(is_duplicate = rep.int(0,sz_)))
df1 %>% dim()

load('question1.RData')
load('question2.RData')

rm(question1_)
rm(question2_)
set.seed(1817328)
idx = which(df$is_duplicate == 0) %>%  sample(size=sz_)
question1_ = rbind(question1[1:sz_,493:512],question1[idx,493:512])
question2_ = rbind(question1[1:sz_,493:512],question2[idx,493:512])

rm(question1)
rm(question2)


model = FLAGS %>% model1() %>% train_model(question1_, question2_,df1)




# Model with sequence and reverse sequence ----
model1 <- function(FLAGS)
{
  
  input1 <- layer_input(shape = c(20))
  input2 <- layer_input(shape = c(20))
  
  input3 <- layer_input(shape = c(20))
  input4 <- layer_input(shape = c(20))
  
  
  seq_reshape1 <- layer_reshape(target_shape = c(1,20))
  seq_reshape2 <- layer_reshape(target_shape = c(1,20))
  
  seq_reshape3 <- layer_reshape(target_shape = c(1,20))
  seq_reshape4 <- layer_reshape(target_shape = c(1,20))
  
  
  seq_gru1 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru2 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
 
   seq_gru3 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  seq_gru4 <- layer_gru(
    units =  FLAGS$seq_embedding_size, 
    recurrent_regularizer = regularizer_l2(l = FLAGS$regularization))
  
  
  
  vector1 <- input1  %>% seq_reshape1 %>% 
    seq_gru1()
  vector2 <- input2  %>% seq_reshape2 %>% 
    seq_gru2()
 
   vector3 <- input3  %>% seq_reshape3 %>% 
    seq_gru3()
  vector4 <- input4  %>% seq_reshape4 %>% 
    seq_gru4()
  
  dense_100_relu       <- layer_dense( units = 100, activation = 'relu')
  dense_100_relu1      <- layer_dense( units = 100, activation = 'relu')
  dense_100_relu2      <- layer_dense( units = 100, activation = 'relu')
  dense_001_softmax  <- layer_dense(units = 1 , activation = "sigmoid")
  flatten_layer_       <- layer_flatten(input_shape = 512)      
  
  
  out1 <- layer_dot(list(vector1, vector2), axes = 1) %>% dense_100_relu1 
  
  
  out2 <- layer_dot(list(vector3, vector4), axes = 1) %>% dense_100_relu2
  
  
  out <- layer_dot(list(out1, out2), axes = 1) %>% dense_100_relu %>% 
    layer_dropout( rate = 0.1)%>% 
    layer_dense(1, activation = "sigmoid")
  
  
  model <- keras_model(list(input1, input2,input3,input4), out)
  model %>% compile(
    optimizer = "adam", 
    loss = "binary_crossentropy", 
    metrics = list(
      acc = metric_binary_accuracy
    )
  )
  model
  
  
  # v1 = model %>% predict(list(question1_1d_1[1:2,],question2_1d_1[1:2,],question1_1d_2[1:2,],question2_1d_2[1:2,]))
  # v1 %>% dim
  
}


train_model <- function(model,question1_,question2_,question3_,question4_)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(question1_), size = 0.1*nrow(question1_))
  
  history <- model %>%
    keras:: fit(
      list(question1_[-val_sample,], question2_[-val_sample,],question3_[-val_sample,], question4_[-val_sample,]),
      df$is_duplicate[-val_sample], 
      batch_size = 128, 
      epochs = 30, 
      validation_data = list(
        list(question1_[val_sample,], question2_[val_sample,],question3_[val_sample,], question4_[val_sample,]), df$is_duplicate[val_sample]
      ),
      callbacks = list(
        callback_early_stopping(patience = 5),
        callback_reduce_lr_on_plateau(patience = 3)
      )
    )  
  prediction_ = list(question1_[val_sample,], question2_[val_sample,]) %>% predict(model,.)
  
  
  list(history= history, model = model, validation = data.frame(prediction = prediction_ , val_sample = val_sample ))
  
  
}



load('df.RData')
df$is_duplicate = df$is_duplicate %>% as.numeric

model_hist <- FLAGS %>% model1() %>% train_model(question1_1d_1,question2_1d_1,question1_1d_2,question2_1d_2)




# change the prportion of negative positive labels to 50/50 ----- (improve accuracy) -----
# change the test data percentage to 20%, let's play with model1 classifier: -----
set.seed(3645789)
idx_duplicated     = which(df$is_duplicate == 1)
idx_non_duplicated = which(df$is_duplicate == 0)
idx_non_duplicated = sample(idx_non_duplicated,idx_duplicated %>% length)
idx_all = rbind(idx_non_duplicated, idx_duplicated)

df2 = df[idx_all, ]

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

train_model <- function(model,question1_,question2_,df)
{
  set.seed(1817328)
  val_sample <- sample.int(nrow(question1_), size = 0.3*nrow(question1_))
  
  history <- model %>%
    keras:: fit(
      list(question1_[-val_sample,], question2_[-val_sample,]),
      df$is_duplicate[-val_sample], 
      batch_size = 128, 
      epochs = 30, 
      validation_data = list(
        list(question1_[val_sample,], question2_[val_sample,]), df$is_duplicate[val_sample]
      ),
      callbacks = list(
        callback_early_stopping(patience = 5),
        callback_reduce_lr_on_plateau(patience = 3)
      )
    )  
  prediction_ = list(question1_[val_sample,], question2_[val_sample,]) %>% predict(model,.)
  
  
  list(history= history, model = model, validation = data.frame(prediction = prediction_ , val_sample = val_sample ))
  
  
}

load('question1.RData')
load('question2.RData')


FLAGS %>% model1() %>% train_model(question1_ = question1[idx_all,493:512],question2_ = question2[idx_all,493:512],df2)








# Compare keras models with ROC curves -----

set.seed(3645789)
n <- nrow(question1_)
test_idx <- sample.int(n, size = round(0.2 * n))
question1_train <- question1_[-test_idx, ]
question1_test <- question1_[test_idx, ]
question2_train <- question2_[-test_idx, ]
question2_test <- question2_[test_idx, ]

model10 <- load_model_hdf5('model10.hdf5')
model7 <- load_model_hdf5('model7.hdf5')
model8 <- load_model_hdf5('model8.hdf5')
model9 <- load_model_hdf5('model9.hdf5')

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


roc_keras_models <- function( ) {

 load('validation10.RData')  
 load('validation9.RData')  
 load('validation8.RData')  
 load('validation7.RData')  
 load('validation6.RData')  
 load('validation5.RData')  
 load('validation4.RData')  
 load('validation3.RData')  
 load('validation2.RData')  
 load('validation1.RData')  


 
 roc10  = get_roc_data_for_keras_model(validation10,'model10')
 roc9  = get_roc_data_for_keras_model(validation_data9,'model9')
 roc8  = get_roc_data_for_keras_model(validation_data8,'model8')
 roc7  = get_roc_data_for_keras_model(validation_data7,'model7')
 roc6  = get_roc_data_for_keras_model(validation_data6,'model6')
 roc5  = get_roc_data_for_keras_model(validation_data5,'model5')
 roc4  = get_roc_data_for_keras_model(validation_data4,'model4')
 roc3  = get_roc_data_for_keras_model(validation_data3,'model3')
 roc2  = get_roc_data_for_keras_model(validation_data2,'model2')
 roc1  = get_roc_data_for_keras_model(validation_data1,'model1')
 

 roc_data = rbind(roc1,roc2,roc3,roc4,roc5,roc6,roc7,roc8,roc9,roc10)
 
 roc <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
    geom_line(aes(color=model_name)) + geom_abline(intercept=0, slope=1, lty=3) +
    ylab(perf@y.name) + xlab(perf@x.name)
  roc
}

roc_keras_models()


multiplot(readRDS('histmodel1.rds') %>% plot(),readRDS('histmodel2.rds') %>% plot(),readRDS('histmodel3.rds') %>% plot(),
          readRDS('histmodel4.rds') %>% plot(),readRDS('histmodel5.rds') %>% plot(),readRDS('histmodel6.rds') %>% plot(),
          readRDS('histmodel7.rds') %>% plot(),readRDS('histmodel8.rds') %>% plot(),readRDS('histmodel9.rds') %>% plot(),    
          readRDS('histmodel10.rds') %>% plot()  ,cols=4)
