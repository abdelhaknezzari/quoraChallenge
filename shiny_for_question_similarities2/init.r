#
# Example R code to install packages
# See http://cran.r-project.org/doc/manuals/R-admin.html#Installing-packages for details
#

###########################################################
# Update this line with the R packages to install:
#my_packages = c("shiny","shinydashboard","shinyjs","ggplot2","wordcloud","ggrepel","keras","RColorBrewer","class","MASS","randomForest","e1071",
#"nnet",
#"ROCR",
#"xgboost",
#"neuralnet",
#"tm",
#"tidytext")


my_packages = c("shiny",
"lsa",
"Rtsne",
"tokenizers",
"TraMineR",
"plot3D",
"textstem",
"textclean",
"NLP",
"openNLP",
"openNLPdata",
"shinydashboard",
"shinyjs",
"grid",
"SnowballC",
"text2vec",
"RColorBrewer",
"ggrepel",
"class",
"MASS",
"randomForest",
"e1071",
"nnet",
"ROCR",
"xgboost",
"neuralnet",
"tm",
"wordcloud",
"keras",
"dplyr",
"ggplot2",
"cales",
"readr",
"purrr",
"tidytext",
"tidyr",
"caret",
"corrplot",
"pryr",
"rgl",
"car")


###########################################################

install_if_missing = function(p) {
  if (p %in% rownames(installed.packages()) == FALSE) {
    install.packages(p, dependencies = TRUE)
  }
  else {
    cat(paste("Skipping already installed package:", p, "\n"))
  }
}
invisible(sapply(my_packages, install_if_missing))
