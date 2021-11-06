### https://cran.r-project.org/web/packages/keras/vignettes/getting_started.html

library(keras)
mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y


x_train <- x_train %>% array_reshape(dim = c( nrow(x_train), (28*28)  ) )

x_test <- x_test %>% array_reshape(dim = c( nrow(x_test), (28*28)  ) )
dim(x_test)


x_test   <- x_test / 255
x_train <- x_train / 255
summary(y_test)


y_train <- to_categorical(y_train)
y_test  <- to_categorical(y_test)



model <- keras_model_sequential()
model %>% 
  layer_dense( units = 256, activation = 'relu', input_shape = c(784) ) %>% 
  layer_dropout(  rate = 0.4) %>% 
  layer_dense( units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3 ) %>% 
  layer_dense( units = 10, activation = 'softmax')

summary(model)

model %>% 
  compile( loss = 'categorical_crossentropy' , optimizer = optimizer_rmsprop() , metrics = c('accuracy') )


history <- model %>% 
  fit( x = x_train, y = y_train,batch_size = 128, epochs = 10,validation_split = 0.2 , callbacks = callback_tensorboard('run/run_a')  )

tensorboard('log')
plot(history)


model %>% evaluate( x_test, y_test )
model %>% predict_classes(x_test)
