library(MASS)
library(ggplot2)
library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(tibble)
library(magrittr)
require(class)
library(C50)
library(rpart)
library(randomForest)
library(plotROC)
library(e1071)
library(caret)
library(scales)


# parameters that will be passed to ``stat_function``
n = 1000
mean = 10
sd = 1
binwidth = 0.05 # passed to geom_histogram and stat_function
set.seed(1)
df <- data.frame(x = rnorm(n, mean, sd))

ggplot(df, aes(x = x, mean = mean, sd = sd, binwidth = binwidth, n = n)) +
  theme_bw() +
  geom_histogram(binwidth = binwidth, 
                 colour = "white", fill = "cornflowerblue", size = 0.1) +
  stat_function(fun = function(x) dnorm(x, mean = mean, sd = sd) * n * binwidth,
                color = "darkred", size = 1)



set.seed(4776)

mvrnorm(20, mu=c(0,0), Sigma=rbind(c(5, 2),
                                  c(0, 2) ))


function1 <- function(mean,sd)
{
  
  Z = rnorm(3,mean,sd)
  A = rbind(c(1,1,0),c(1,0,1))
  print(A %*% t(A))
  X = A %*% Z
  X = matrix(0,nrow=2,ncol=1000)
  A = rbind(c(1,1,0),c(1,0,1))
  for(i in 1:1000){
    Z = rnorm(3,mean,sd)
    X[,i] = A %*% Z
  }
  X
}

getRandomVar <- function(mean,sd,n)
{
  
  Z = rnorm(2,mean,sd)
  A = rbind(c(1,0),c(-1,1))

  X = matrix(0,nrow=2,ncol=n)

  for(i in 1:n){
    Z = rnorm(2,mean,sd)
    X[,i] = A %*% Z
  }
  X
}


getTestDataset <- function()
{
  X1 = getRandomVar(0,1,100)
  X2 = getRandomVar(2,1,100)
  X_data1 = data.frame(x1=X1[1,],x2=X1[2,],class="1")
  X_data2 = data.frame(x1=X2[1,],x2=X2[2,],class="0")
  rbind(X_data1,X_data2)
  
}



decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
  
  
  
}


draw_boundary2 <- function(xy, model) {
  u <- rep(xy, times = length(xy))
  v <- rep(xy, each = length(xy))
  
  X <- cbind(x1 = u,x2= v) %>% as.data.frame()
  
  
  
  p <- predict(model, X, type = "class")
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  
  cbind(s1 = u,s2= v, class = p) %>% 
    as.data.frame 
}

draw_boundary3 <- function(xy, model,threshold) {
  u <- rep(xy, times = length(xy))
  v <- rep(xy, each = length(xy))
  
  X <- cbind(x1 = u,x2= v) %>% as.data.frame()
  
  
  pr <- predict(model, X,type = "prob" ,probability = TRUE)
  
  
  cl <- ifelse(pr[,1] >threshold, 1 , 0) 
  
  cbind(s1 = u,s2= v, class = cl) %>% 
    as.data.frame 
}

getModelClass <- function(model)
{
  modelClass <- class(model)
  
  ifelse( modelClass %>% length() >= 2, modelClass[2],modelClass[1] )
}

getPrediction <- function(model, data,threshold)
{
  class <- getModelClass(model)
  
  if(  class == "naiveBayes"  )
  {
    pr = predict(model, data,type ="raw")
  } else if( class  == "lda" ){
    
    pr = predict(model, data,type ="raw")$posterior
  } else if( class == "svm") {
    
    pr = predict(model, data, probability=TRUE) %>% attr("probabilities")
    
  } else pr = predict(model, data,type = "prob" ) 
  
  
  ifelse(pr[,1] >threshold, 1 , 0) 
}



getBoundary <- function(xy, model,threshold) {
  u <- rep(xy, times = length(xy))
  v <- rep(xy, each = length(xy))
  
  X <- cbind(x1 = u,x2= v) %>% as.data.frame()
  
  
  cl <-  model %>% getPrediction(X,threshold)
  
  cbind(s1 = u,s2= v, class = cl) %>% 
    as.data.frame 
}

plot3 <- function(x_data,model, threshold)
{
  boundary_data <- draw_boundary3( seq(from = -4, to = 4, length = 400), model,threshold )
  
  ggplot( ) +
    geom_contour( data = boundary_data, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 1, colour = "black", bins = 1 ,show.legend = TRUE) +
    #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
    geom_point( data = x_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
    coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X1 Numeric Feature") +  ylab("X2 Numeric Feature") +
    scale_colour_discrete(name  ="Data Classes",  labels=c("Class 0", "Class 1"))
  
}


plotBoundary <- function(x_data,model, threshold)
{
  boundary_data <- seq(from = -20, to = 10, length = 500) %>% getBoundary( model,threshold )
  
  ggplot( ) +
    geom_contour( data = boundary_data, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 1, colour = "black", bins = 1 ,show.legend = TRUE) +
    #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
    geom_point( data = x_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
    coord_cartesian( xlim = c(-5,5), ylim = c(-6,4) ) +  xlab("X1 Numeric Feature") +  ylab("X2 Numeric Feature") +
    scale_colour_discrete(name  ="Data Classes",  labels=c("Class 0", "Class 1"))
  
}

getConfMatrix <- function(model,data,threshold)
{
  data$class %>% as.character() %>% as.numeric() %>%  table( model %>%  getPrediction(data,threshold))
}

getConfMatrix2 <- function(model,data,threshold)
{
  data$class %>% as.character() %>% as.numeric() %>%  confusionMatrix( model %>%  getPrediction(data,threshold))
}




getRocXY <- function(model, data,threshold)
{
  confMat <- model %>% getConfMatrix(data,threshold)
  col <- confMat %>% colnames()
  
  if(col %>% length() ==1 )
  {
    
    #   Suppose we take the threshold to be 0, that is, 
    #    all emails are classified as spam. On the one hand, 
    #    this implies that no spam emails are predicted as real emails and so there are no false negatives 
    #    — the true positive rate (or recall) is 1. On the other hand, 
    
    #    this also means that no real email is classified as real, and thus there are no true negatives 
    #    — the false positive rate is also 1. This corresponds to the top-right part of the curve.
    
    #    Now suppose that the threshold is 1, that is, no email is classified as spam. Then,
    #    there are no true positives (and thus the true positive rate is 0) 
    #    and no false positives (and thus the false positive rate is 0). 
    #    This corresponds to the bottom-left of the curve.
    
    
    if("1" %in% col)
    {
      # No True negataive and no false negative
      sens <- 0
      spec <- 1
    } else {
      sens <- 1
      spec <- 0
      
    }
    
    
  } else {
    
    sens <- confMat %>% sensitivity()
    spec <- confMat %>% specificity()
  }
  
  list( TPR = sens, FPR = (1 - spec) ) %>% as.data.frame()
  
}


ggplotConfusionMatrix <- function(m) {

    ggplot(data = m ,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
    theme(legend.position = "none") 

}



X_data = getTestDataset()

ggplot(X_data, aes(x1, x2,color=class)) +
  geom_point(shape = 21,size = 2)


svmfit <- svm(class~.,data=X_data, kernal = "radial", gamma = 1, cost=1)
plot(svmfit,X_data)

svmfit <- svm(class~.,data=X_data, kernal = "linear", cost = 0.1, scale=FALSE)
plot(svmfit,X_data)





randomForest(class ~ ., data=X_data) %>% plotBoundary(X_data,.,0.7)
naiveBayes(class ~ ., data=X_data) %>% plotBoundary(X_data,.,0.7)
knn3(class ~ ., data=X_data, k = 1) %>% plotBoundary(X_data,.,0.7)
lda(class ~ ., data=X_data)  %>% plotBoundary(X_data,.,0.5)
rpart(class ~ ., data=X_data) %>% plotBoundary(X_data,.,0.5)
C5.0(class ~ ., data=X_data) %>%  plotBoundary(X_data,.,0.5)
svm(class ~ ., data=X_data, kernel = "radial", probability=TRUE)  %>%  plotBoundary(X_data,.,0.5)
svm(class ~ ., data=X_data, kernel = "linear", probability=TRUE)  %>%  plotBoundary(X_data,.,0.5)


roc1 <- seq(from = 0.1, to = 0.1, by = 0.1) %>% 
  lapply(getRocXY, model =C5.0(class ~ ., data=X_data) , data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame() %>% 
  cbind(model = 'C5.0')

roc2 <- seq(from = 0.0, to = 1, by = 0.01) %>% 
  lapply(getRocXY, model =knn3(class ~ ., data=X_data, k = 1), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>% 
  cbind(model = 'knn3' )


roc3 <- seq(from = 0.0, to = 1, by = 0.01) %>% 
  lapply(getRocXY, model =randomForest(class ~ ., data=X_data), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>% 
  cbind(model = 'randomForest' )

roc4 <- seq(from = 0.0, to = 1, by = 0.01) %>% 
  lapply(getRocXY, model =naiveBayes(class ~ ., data=X_data), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>% 
  cbind(model = 'naiveBayes' )


roc5 <- seq(from = 0.0, to = 1, by = 0.01) %>% 
  lapply(getRocXY, model =lda(class ~ ., data=X_data), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>% 
  cbind(model = 'lda' )

roc6 <- seq(from = 0.0, to = 1, by = 0.01) %>% 
  lapply(getRocXY, model =rpart(class ~ ., data=X_data), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>% 
  cbind(model = 'rpart' )

roc7 <- seq(from = 0.0, to = 1, by = 0.01) %>% 
  lapply(getRocXY, model =svm(class ~ ., data=X_data, kernel = "radial", probability=TRUE), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>% 
  cbind(model = 'svm' )



rocs <- roc4 %>%  rbind(  roc7  )  


rocs <- roc1 %>%  rbind(  roc2  )  %>% 
  rbind(  roc3  ) %>%  rbind(  roc4  ) %>%  
  rbind(  roc5  ) %>%  rbind(  roc6 ) %>%  
  rbind(  roc7 )


roc5 %>% ggplot( aes(x = FPR, y = TPR)) +
  geom_point(aes(color = model))  + 
  coord_cartesian(ylim = c(0,1), xlim = c(0, 1))+ 
  ylab('True Positive Rate') + xlab('False Positive Rate') 


seq(from = 0.0, to = 1, by = 0.1) %>% 
  lapply(getRocXY, model =lda(class ~ ., data=X_data), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame() %>% cbind(model = 'naiveBayes' ) %>%  ggplot( aes(x = FPR, y = TPR)) +
  geom_point(aes(color = model))  + 
  coord_cartesian(ylim = c(0,1), xlim = c(0, 1))+ 
  ylab('True Positive Rate') + xlab('False Positive Rate') 





seq(from = 0.9, to = .99999, by = 0.0001) %>% 
  lapply(getRocXY, model = rpart(class ~ ., data=X_data), data = X_data ) %>% 
  rbindlist( fill = TRUE) %>% 
  as.data.frame()%>%  ggplot( aes(x = FPR, y = TPR)) +
  geom_point(aes(color = "red"))  + 
  coord_cartesian(ylim = c(0,1), xlim = c(0, 1))+ 
  ylab('True Positive Rate') + xlab('False Positive Rate') +
  guides(fill=guide_legend(title=NULL)) +
  theme(legend.title=element_blank())
 


rpart(class ~ ., data=X_data)  %>% getRocXY(X_data,0.2) %>% 
  ggplot( aes(x = FPR, y = TPR)) + 
  geom_roc(stat = "identity") + style_roc(theme = theme_grey)
binorm.plot


 rpart(class ~ ., data=X_data)  %>% getRocXY(X_data,0.2) %>%  ggplot( aes(x = FPR, y = TPR)) +
  geom_line()  + 
  ylab('True Positive Rate') + xlab('False Positive Rate')

roc


randomForest(class ~ ., data=X_data) %>% getConfMatrix2(X_data,0.7) -> po


po$table %>% as.data.frame()%>% ggplotConfusionMatrix()
p1

set.seed(2529)
D.ex <- rbinom(200, size = 1, prob = .5)
D.ex <- rbinom(200, size = 1, prob = .5)
M1 <- rnorm(200, mean = D.ex, sd = .65)
M2 <- rnorm(200, mean = D.ex, sd = 1.5)

test <- data.frame(D = D.ex, D.str = c("Healthy", "Ill")[D.ex + 1], 
                   M1 = M1, M2 = M2, stringsAsFactors = FALSE)
basicplot <- ggplot(test, aes(d = D, m = M1)) + geom_roc()
basicplot


shiny_plotROC()



model <- randomForest(class ~ ., data=X_data)
draw_boundary2(seq(from = -4, to = 4, length = 1000),model)

D.ex <- test$D
M.ex <- test$M1
mu1 <- mean(M.ex[D.ex == 1])
mu0 <- mean(M.ex[D.ex == 0])
s1 <- sd(M.ex[D.ex == 1])
s0 <- sd(M.ex[D.ex == 0])
c.ex <- seq(min(M.ex), max(M.ex), length.out = 300)

binorm.roc <- data.frame(c = c.ex, 
                         FPF = pnorm((mu0 - c.ex)/s0), 
                         TPF = pnorm((mu1 - c.ex)/s1)
)

binorm.plot <- ggplot(binorm.roc, aes(x = FPF, y = TPF, label = c)) + 
  geom_roc(stat = "identity") + style_roc(theme = theme_grey)
binorm.plot
binorm.plot <- ggplot(binorm.roc, aes(x = FPF, y = TPF)) + 
  geom_roc(stat = "identity") + style_roc(theme = theme_grey)
binorm.plot


fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )
ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ,show.legend = TRUE) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y") + 
  scale_colour_discrete(name  ="Data Classes",  labels=c("Class 0", "Class 1"))

plot3(X_data,model,0.7)
randomForest(class ~ ., data=X_data) %>% plot3(X_data,.,0.7)

library(caret)
model <- knn3(class ~ ., data=X_data, k = 1)

decisionplot(model, X_data, class = "class", main = "kNN (1)")

fg <- draw_boundary2( seq(from = -4, to = 4, length = 400), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 1, colour = "black", bins = 1 ,show.legend = TRUE) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X1 Numeric Feature") +  ylab("X2 Numeric Feature") +
  scale_colour_discrete(name  ="Data Classes",  labels=c("Class 0", "Class 1"))


knn3(class ~ ., data=X_data, k = 1) %>% plot3(X_data,.,0.5)

model

model <- naiveBayes(class ~ ., data=X_data)
naiveBayes(class ~ ., data=X_data) %>% plot3(X_data,.,0.1)


fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "naive Bayes")


model <- lda(class ~ ., data=X_data)


lda(class ~ ., data=X_data) %>% plot3(X_data,.,0.1)
fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "LDA")


model <- rpart(class ~ ., data=X_data)

rpart(class ~ ., data=X_data) %>% plot3(X_data,.,0.565656)
fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "CART")


model <- rpart(class ~ ., data=X_data,
               control = rpart.control(cp = 0.001, minsplit = 1))

fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "CART (overfitting)")



model <- C5.0(class ~ ., data=X_data)

C5.0(class ~ ., data=X_data) %>% plot3(X_data,.,0.565656)

fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "C5.0")

library(randomForest)
model <- randomForest(class ~ ., data=X_data)
fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "Random Forest")



model <- svm(class ~ ., data=X_data, kernel = "radial")

svm(class ~ ., data=X_data, kernel = "radial") %>% plot3(X_data,.,0.565656)

fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "SVD (radial)")



model <- svm(class ~ ., data=X_data, kernel="linear")
fg <- draw_boundary2( seq(from = -4, to = 4, length = 50), model )

ggplot( ) +
  geom_contour( data = fg, mapping= aes( x = s1, y = s2, z = class, colour = class  ), size = 0.5, colour = "black", bins = 1 ) +
  #geom_point( data = fg, mapping =   aes( x = s1, y = s2, colour = class ), size = 2 ) +
  geom_point( data = X_data, mapping =   aes( x = x1, y = x2, colour = class ), size = 4 ) +
  coord_cartesian( xlim = c(-3,4), ylim = c(-4,4) ) +  xlab("X") +  ylab("Y")

decisionplot(model, X_data, class = "class", main = "SVD (linear)")



data <- rbind(iris3[1:25,1:2,1],
               iris3[1:25,1:2,2],
               iris3[1:25,1:2,3]) %>% cbind( class= factor(c(rep("s",25), rep("c",25), rep("v",25))))

 data = X_data
model <- svm(class ~ ., data=X_data, kernel = "radial")

boundary_decision <- function(model, data, class = NULL, predict_type = "class",
                              resolution = 100, showgrid = TRUE, ...)
{
  
  test <- expand.grid(x=seq(min(data[,1]-1), max(data[,1]+1),
                            by=0.2),
                      y=seq(min(data[,2]-1), max(data[,2]+1), 
                            by=0.2))
  

  r <- sapply(data[,1:2], range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  g <- cbind(g, p)
  
  
  ggplot() +
    geom_point(aes(x=x1, y=x2, col=p),
               data = g,
               size=1.2) + 
    geom_contour(aes(x=x1, y=x2, z=p, group=p, color=p),
                 bins=2,
                 data=g) +
    geom_point(aes(x=x1, y=x2, col=p),
               size=3,
               data=data)
  
  
  
}





head(train)

train <- rbind(iris3[1:25,1:2,1],
               iris3[1:25,1:2,2],
               iris3[1:25,1:2,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))




test <- expand.grid(x=seq(min(train[,1]-1), max(train[,1]+1),
                          by=0.1),
                    y=seq(min(train[,2]-1), max(train[,2]+1), 
                          by=0.1))
test




classif <- train %>% knn( test, cl, k = 3, prob=TRUE)
prob <- classif %>% attr( "prob")
prob



dataf <- bind_rows(mutate(test,
                          prob=prob,
                          cls="c",
                          prob_cls=ifelse(classif==cls,
                                          1, 0)),
                   mutate(test,
                          prob=prob,
                          cls="v",
                          prob_cls=ifelse(classif==cls,
                                          1, 0)),
                   mutate(test,
                          prob=prob,
                          cls="s",
                          prob_cls=ifelse(classif==cls,
                                          1, 0)))

ggplot(dataf) +
  geom_point(aes(x=x, y=y, col=cls),
             data = mutate(test, cls=classif),
             size=1.2) + 
  geom_contour(aes(x=x, y=y, z=prob_cls, group=cls, color=cls),
               bins=2,
               data=dataf) +
  geom_point(aes(x=x, y=y, col=cls),
             size=3,
             data=data.frame(x=train[,1], y=train[,2], cls=cl))









# Load dplyr to make life easier


# Set random seed to fix the answer

set.seed(1337)

m <- 10
x <- runif(n = m, min = 0, max = 500) %>%
  round

x
a <- 1
b <- 10

y <- (x * a) + b

y
X <- matrix(
  cbind( 1, x ), 
  ncol = 2
)
X

beta <- matrix(
  cbind( b, a ), 
  nrow = 2
)


Y <- X %*% beta 

Y

identical(y,Y[,1])

coef_ne <- solve(t(X) %*% X) %*% (t(X) %*% y)

coef_ne
qrx <- qr(X)
Q <- qr.Q(qrx, complete = TRUE)
R <- qr.R(qrx)


solve.qr(qrx, y)




# Set a seed for reproducibility

set.seed(1337)

# This function will create bivariate normal distributions about two means with
# a singular deviation

dummy_group <- function(
  x = 30, 
  mean1 = 10, 
  mean2 = 10, 
  sd = 0.45
) {
  
  cbind(
    rnorm(x, mean1, sd),
    rnorm(x, mean2, sd)
  )
  
}

# Generate 10 bivariate distributions using normal distributions to generate the
# means for each of the two variables. Bind this all together into a dataframe, 
# and label this for training examples. Note that I draw the distinctions
# between 0s and 1s, pretty much by eye - there was not magic to this.

dummy_data <- data_frame(
  mean1 = rnorm(10),
  mean2 = rnorm(10)
) %>%
  pmap(dummy_group) %>%
  map(as.data.frame) %>%
  rbind_all %>%
  mutate(
    group = rep(1:10, each = 30),
    group = factor(group),
    group_bit = ifelse(group %in% c(2,3,5,10), 0, 1),
    group_bit = factor(group_bit)
  ) %>%
  select(
    X = V1,
    Y = V2,
    group,
    group_bit
  )


library(ggplot2)

p <- dummy_data %>%
  ggplot +
  aes(
    x = X,
    y = Y,
    colour = group_bit
  ) +
  geom_point(
    size = 3
  )

p


G <- dummy_data$group_bit %>%
  #as.character %>% 
  as.integer

X <- dummy_data[,c("X","Y")] %>%
  as.matrix

beta <- solve(t(X) %*% X) %*% (t(X) %*% G)
beta_qr <- qr.solve(X, G)

beta


all.equal(
  as.vector(beta),
  as.vector(beta_qr)
)


Y <- X %*% beta
Y <- ifelse(Y > 0.5, 1, 0)

table(G, Y)






draw_boundary <- function(xy, beta) {
  
  u <- rep(xy, times = length(xy))
  v <- rep(xy, each = length(xy))
  
  X <- cbind(x1 = u,x2= v)
  
  Y <- X %*% beta

    cbind(X, Y) %>% 
    as.data.frame %>%
    mutate(
      actual = ifelse(Y > 0.5, 1, 0)
    ) 
}


bound <- draw_boundary(
  seq(from = -4, to = 4, length = 1000),
  beta
)


bound %>% 
  ggplot +
  aes(
    x = u,
    y = v,
    z = actual,
    colour = actual
  ) +
  geom_contour(
    size = 0.4,
    colour = "black",
    bins = 1
  ) +
  geom_point(
    data = dummy_data %>% cbind(prediction = Y) %>%
      mutate(
        prediction = factor(prediction),
        actual = factor(group_bit)
      ),
    aes(
      x = X,
      y = Y,
      colour = actual,
      shape = prediction
    ),
    size = 3
  ) +
  coord_cartesian(
    xlim = c(-2.2,2.8),
    ylim = c(-3,3.5)
  ) +
  xlab("X") +
  ylab("Y")

