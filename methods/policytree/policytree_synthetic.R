library(policytree)
library(grf)
library(tictoc)
library(glue)
library(ggplot2)
library(dplyr)

setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/data/processed/synthetic/')

run = function(df_train, df_test, depth) {
  get_scores = function(df_train) {
    df_train = df_train %>% mutate(y_pred = ifelse(t == 0, linear0, 
                                                   ifelse(t==1, linear1, NA))) %>%
      transform(y_pred = as.numeric(y_pred))
    bias <- (df_train$y - df_train$y_pred)/df_train$prob_t_pred_tree
    scores <- df_train[, c('linear0', 'linear1')]
    scores <- scores %>% rename('0'='linear0',
                                '1'='linear1')
    for(i in 1:ncol(scores)) {
      scores[, i] <- scores[, i] + bias
    }
    return(scores)
  }
  
  scores <- get_scores(df_train)
  X_train <- data.matrix(df_train[ , names(df_train) %in% c('V1', 'V2')])
  y_train <- data.matrix(df_train[c('y')])
  t_train <- factor(df_train[, 't'], labels=c(0, 1))
  
  X_test <- data.matrix(df_test[ , names(df_test) %in% c('V1', 'V2')])
  y_test <- data.matrix(df_test[c('y')])
  t_test <- factor(df_test[, 't'], labels=c(0, 1))
  
  # tic('policytree')
  start <- Sys.time()
  tree <- policy_tree(X_train, scores, depth)
  # policy_tree(X_train, scores, 2)
  # toc()
  time_elapsed <- Sys.time() - start
  
  node.id <- predict(tree, X_test, type='action.id')
  node.id <- node.id-1
  df_test$treatment <- node.id
  df_test = df_test %>% mutate(t_opt = ifelse(y0 > y1, 0, 1))
  df_test = df_test %>% mutate(y_opt = ifelse(y0 > y1, y0, y1))
  
  df_test = df_test %>% mutate(realized_outcome = ifelse(treatment == 0, y0, 
                                                         ifelse(treatment == 1, y1,
                                                                ifelse(treatment == 2, y2, NA))))
  
  regret <- df_test$y_opt - df_test$realized_outcome
  
  eval_policy <- df_test$treatment == df_test$t_opt
  # return(mean(df_test$realized_outcome))
  list_return <- list('oos_regret' = sum(regret),
                      'oosp' = sum(eval_policy)/length(eval_policy), 'time' = time_elapsed)
  return(list_return)
}


seed = '1'
split = '1'
dataset_type <- '0.1'
df_train <- read.csv(glue('data_train_{dataset_type}_{split}.csv'))
df_test <- read.csv(glue('data_test_{dataset_type}_{split}.csv'))
run(df_train, df_test, 1)

splits <- c('1', '2', '3', '4', '5')
seeds <- c('1')
dataset_type <- '0.1'
oosp_0.1 <- c()
time_0.1 <- c()
oosr_0.1 <- c()
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 1)
    
    oosp_0.1 <- append(oosp_0.1, res$oosp)
    time_0.1 <- append(time_0.1, res$time)
    oosr_0.1 <- append(oosr_0.1, res$oos_regret)
    
    # oosp_0.1 <- append(oosp_0.1, run(df_train, df_test, 2))
  }
}

dataset_type <- '0.25'
oosp_0.25 <- c()
time_0.25 <- c()
oosr_0.25 <- c()
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 1)
    
    oosp_0.25 <- append(oosp_0.25, res$oosp)
    time_0.25 <- append(time_0.25, res$time)
    oosr_0.25 <- append(oosr_0.25, res$oos_regret)
    
    # oosp_0.25 <- append(oosp_0.25, run(df_train, df_test, 2))
  }
}

dataset_type <- '0.5'
oosp_0.5 <- c()
time_0.5 <- c()
oosr_0.5 <- c()
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 1)
    
    oosp_0.5 <- append(oosp_0.5, res$oosp)
    time_0.5 <- append(time_0.5, res$time)
    oosr_0.5 <- append(oosr_0.5, res$oos_regret)
    
    # oosp_0.5 <- append(oosp_0.5, run(df_train, df_test, 1))
  }
}

dataset_type <- '0.75'
oosp_0.75 <- c()
time_0.75 <- c()
oosr_0.75 <- c()
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 1)
    
    oosp_0.75 <- append(oosp_0.75, res$oosp)
    time_0.75 <- append(time_0.75, res$time)
    oosr_0.75 <- append(oosr_0.75, res$oos_regret)
    
    # oosp_0.75 <- append(oosp_0.75, run(df_train, df_test, 1))
  }
}

dataset_type <- '0.9'
oosp_0.9 <- c()
time_0.9 <- c()
oosr_0.9 <- c()
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 1)
    
    oosp_0.9 <- append(oosp_0.9, res$oosp)
    time_0.9 <- append(time_0.9, res$time)
    oosr_0.9 <- append(oosr_0.9, res$oos_regret)
    
    # oosp_0.9 <- append(oosp_0.9, run(df_train, df_test, 1))
  }
}


results_synthetic = data.frame(p0.1=oosp_0.1, p0.25=oosp_0.25, p0.5=oosp_0.5, p0.75=oosp_0.75,
                     p0.9=oosp_0.9, oosr_0.1=oosr_0.1, oosr_0.25=oosr_0.25, oosr_0.5=oosr_0.5, oosr_0.75=oosr_0.75,
                     oosr_0.9=oosr_0.9, time_0.1=time_0.1, time_0.25=time_0.25, time_0.5=time_0.5,
                     time_0.75=time_0.75, time_0.9=time_0.9)
colMeans(results_synthetic)
setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/results/synthetic/compiled/policytree')
write.csv(results_synthetic, glue('raw.csv'), row.names = FALSE)
