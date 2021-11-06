library(wordVectors)
install.packages(c("tsne", "Rtsne", "ggplot2", "ggrepel"))
library(tsne)
library(Rtsne)
library(ggplot2)
library(ggrepel)


prep_word2vec("Texts/Pepys.txt", "Pepys_processed.txt", lowercase = T)
pepys <- train_word2vec("Pepys_processed.txt",
                        output = "Pepys_model.bin", threads = 1,
                        vectors = 100, window = 12)


class(pepys)

pepys2 <- read.vectors("Pepys_model.bin")
pepys %>% closest_to('france')


plot(pepys)
w2v_plot <- function(model, word, path, ref_name) {
  
  # Identify the nearest 10 words to the average vector of search terms
  ten <- nearest_to(model, model[[word]])
  
  # Identify the nearest 500 words to the average vector of search terms and
  # save as a .txt file
  main <- nearest_to(model, model[[word]], 500)
  wordlist <- names(main)
  filepath <- paste0(path, ref_name)
  write(wordlist, paste0(filepath, ".txt"))
  
  # Create a subset vector space model
  new_model <- model[[wordlist, average = F]]
  
  # Run Rtsne to reduce new Word Embedding Model to 2D (Barnes-Hut)
  reduction <- Rtsne(as.matrix(new_model), dims = 2, initial_dims = 50,
                     perplexity = 30, theta = 0.5, check_duplicates = F,
                     pca = F, max_iter = 1000, verbose = F,
                     is_distance = F, Y_init = NULL)
  
  # Extract Y (positions for plot) as a dataframe and add row names
  df <- as.data.frame(reduction$Y)
  rows <- rownames(new_model)
  rownames(df) <- rows
  
  # Create t-SNE plot and save as jpeg
  ggplot(df) +
    geom_point(aes(x = V1, y = V2), color = "red") +
    geom_text_repel(aes(x = V1, y = V2, label = rownames(df))) +
    xlab("Dimension 1") +
    ylab("Dimension 2 ") +
    # geom_text(fontface = 2, alpha = .8) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none") +
    ggtitle(paste0("2D reduction of Word Embedding Model ", ref_name," using t_SNE"))
  
  ggsave(paste0(ref_name, ".jpeg"), path = path, width = 24,
         height = 18, dpi = 100)
  
  new_list <- list("Ten nearest" = ten, "Status" = "Analysis Complete")
  return(new_list)
  
}


w2v_plot(pepys, "king", "Results/", "king")



set.seed(40)


centers = 100


clusters <- pepys %>% kmeans(centers = centers, iter.max = 40)



sapply(sample(1:centers,10),function(n) {
  names(clusters$cluster[clusters$cluster==n][1:10])})


