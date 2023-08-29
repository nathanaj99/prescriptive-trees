library(policytree)
library(grf)
library(tictoc)
library(glue)
library(ggplot2)
library(dplyr)

setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/data/processed/warfarin/')

run = function(df_train, df_test, depth, dataset_type) {
  get_scores = function(df_train) {
    df_train = df_train %>% mutate(y_pred = ifelse(t == 0, ml0, 
                                                   ifelse(t==1, ml1, ifelse(t==2, ml2, 'NA')))) %>%
      transform(y_pred = as.numeric(y_pred))
    bias <- (df_train$y - df_train$y_pred)/df_train$prob_t_pred_tree
    scores <- df_train[, c('ml0', 'ml1', 'ml2')]
    scores <- scores %>% rename('0'='ml0',
                                '1'="ml1",
                                '2'="ml2")
    for(i in 1:ncol(scores)) {
      scores[, i] <- scores[, i] + bias
    }
    return(scores)
  }
  
  get_scores_not33 = function(df_train) {
    df_train = df_train %>% mutate(y_pred = ifelse(t == 0, lrrf0, 
                                                   ifelse(t==1, lrrf1, ifelse(t==2, lrrf2, 'NA')))) %>%
      transform(y_pred = as.numeric(y_pred))
    bias <- (df_train$y - df_train$y_pred)/df_train$prob_t_pred_tree
    # scores <- df_train[, c('lrrf0', 'lrrf1', 'lrrf2')]
    # scores <- scores %>% rename('0'='lrrf0',
    #                             '1'="lrrf1",
    #                             '2'="lrrf2")
    
    scores <- df_train[, c('lrrf0', 'lrrf1')]
    scores <- scores %>% rename('0'='lrrf0',
                                '1'="lrrf1")
    for(i in 1:ncol(scores)) {
      scores[, i] <- scores[, i] + bias
    }
    return(scores)
  }
  
  if(dataset_type == '0.33') {
    scores <- get_scores(df_train)
  }
  else{
    scores <- get_scores_not33(df_train)
  }
  
  X_train <- data.matrix(df_train[ , !(names(df_train) %in% c('y', 't', 'y0',
                                                              'y1', 'y2', 'prob_t_pred_tree',
                                                              'ml0', 'ml1', 'ml2', 'y_pred',
                                                              'White', 'lrrf0', 'lrrf1', 'lrrf2',
                                                              'lr0', 'lr1', 'lr2'))])
  y_train <- data.matrix(df_train[c('y')])
  t_train <- factor(df_train[, 't'], labels=c(0, 1, 2))
  
  X_test <- data.matrix(df_test[ , !(names(df_test) %in% c('y', 't', 'y0',
                                                           'y1', 'y2', 'prob_t_pred_tree',
                                                           'ml0', 'ml1', 'ml2', 'y_pred',
                                                           'White', 'lrrf0', 'lrrf1', 'lrrf2',
                                                           'lr0', 'lr1', 'lr2'))])
  y_test <- data.matrix(df_test[c('y')])
  t_test <- factor(df_test[, 't'], labels=c(0, 1, 2))
  
  # tic('policytree')
  start <- Sys.time()
  tree <- policy_tree(X_train, scores, depth)
  # toc()
  time_elapsed <- Sys.time() - start
  
  node.id <- predict(tree, X_test, type='action.id')
  node.id <- node.id-1

  df_test = df_test %>% mutate(t_opt = ifelse(y0 == 1, 0,
                                                 ifelse(y1 == 1, 1, ifelse(y2 == 1, 2, 'NA'))))

  eval_policy <- node.id == df_test$t_opt
  
  
  # node.id <- predict(tree, X_train, type='action.id')
  # node.id <- node.id-1
  # 
  # df_train = df_train %>% mutate(t_opt = ifelse(y0 == 1, 0, 
  #                                             ifelse(y1 == 1, 1, ifelse(y2 == 1, 2, 'NA'))))
  # 
  # eval_policy <- node.id == df_train$t_opt
  list_return <- list('oos_regret'=length(eval_policy)-sum(eval_policy), 
                      'oosp' = sum(eval_policy)/length(eval_policy), 'time' = time_elapsed)
  return(list_return)
}


seeds <- c('1', '2', '3', '4', '5')
splits <- c('1', '2', '3', '4', '5')
dataset_type <- 'r0.11'

seed = '1'
split = '4'
df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
run(df_train, df_test, 2, dataset_type)


oosp_random <- c()
oosp_r0.06 <- c()
oosp_r0.11 <- c()
oosr_random <- c()
oosr_r0.06 <- c()
oosr_r0.11 <- c()
time_random <- c()
time_r0.06 <- c()
time_r0.11 <- c()

dataset_type <- '0.33'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    res <- run(df_train, df_test, 2, dataset_type)
    
    oosp_random <- append(oosp_random, res$oosp)
    time_random <- append(time_random, res$time)
    oosr_random <- append(oosr_random, res$oos_regret)
  }
}

dataset_type <- 'r0.06'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 2, dataset_type)
    
    oosp_r0.06 <- append(oosp_r0.06, res$oosp)
    time_r0.06 <- append(time_r0.06, res$time)
    oosr_r0.06 <- append(oosr_r0.06, res$oos_regret)
  }
}

dataset_type <- 'r0.11'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    res <- run(df_train, df_test, 2, dataset_type)
    
    oosp_r0.11 <- append(oosp_r0.11, res$oosp)
    time_r0.11 <- append(time_r0.11, res$time)
    oosr_r0.11 <- append(oosr_r0.11, res$oos_regret)
  }
}


results = data.frame(random=oosp_random, r0.06=oosp_r0.06, r0.11=oosp_r0.11,
                     random_oos_regret=oosr_random, r0.06_oos_regret=oosr_r0.06, r0.11_oos_regret=oosr_r0.11,
                     random_time=time_random, r0.06_time=time_r0.06,
                     r0.11_time=time_r0.11)
# colMeans(results)
transformed_results = data.frame()
for(i in 1:ncol(results)) {
  print(names(results)[i])
  buffer = data.frame(exp_design=rep(names(results)[i], 25), oosp=results[, i])
  transformed_results <- rbind(transformed_results, buffer)
}

ggplot(transformed_results, aes(x=exp_design, y=oosp)) + 
  geom_boxplot()

setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/results/warfarin/compiled/CF')

write.csv(results, glue('raw_proba.csv'), row.names = FALSE)
# ggsave(glue("oosp_boxplot_proba.pdf"))
