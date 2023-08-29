library(policytree)
library(grf)
library(tictoc)
library(glue)
library(ggplot2)
library(dplyr)
library(plyr)

setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/data/Warfarin_v2/rf_balance_proba_white')

run = function(df_train, df_test, from_vector) {
  X_train <- data.matrix(df_train[ , !(names(df_train) %in% c('y', 't', 'y0',
                                                              'y1', 'y2', 'prob_t_pred_tree',
                                                              'ml0', 'ml1', 'ml2', 'y_pred',
                                                              'White', 'lrrf0', 'lrrf1', 'lrrf2',
                                                              'lr0', 'lr1', 'lr2'))])
  y_train <- data.matrix(df_train[c('y')])
  w_hat <- data.matrix(df_train[c('prob_t_pred_tree')])
  
  X_test <- data.matrix(df_test[ , !(names(df_test) %in% c('y', 't', 'y0',
                                                           'y1', 'y2', 'prob_t_pred_tree',
                                                           'ml0', 'ml1', 'ml2', 'y_pred',
                                                           'White', 'lrrf0', 'lrrf1', 'lrrf2',
                                                           'lr0', 'lr1', 'lr2'))])
  y_test <- data.matrix(df_test[c('y')])
  
  df_train$t_buffer <- df_train$t
  df_train$t_buffer <- mapvalues(df_train$t_buffer, 
                                 from=from_vector, 
                                 to=c(1, 0, 0))
  
  t_train <- data.matrix(df_train[c('t_buffer')])
  
  df_test$t_buffer <- df_test$t
  df_test$t_buffer <- mapvalues(df_test$t_buffer, 
                                from=from_vector, 
                                to=c(1, 0, 0))
  
  t_test <- data.matrix(df_test[c('t_buffer')])
  Y.forest = regression_forest(X_train, y_train)
  y_hat = predict(Y.forest)$predictions
  forest <- causal_forest(X_train, y_train, t_train, Y.hat=y_hat, W.hat=w_hat)
  pred <- predict(forest, X_test)
  
  return(pred)
}

driver_1va = function(df_train, df_test) {

  # t = 0 v all
  t0 = run(df_train, df_test, c(0, 1, 2))
  
  # t = 1 v all
  t1 = run(df_train, df_test, c(1, 0, 2))
  
  # t = 2 v all
  t2 = run(df_train, df_test, c(2, 0, 1))
  
  results = data.frame(t0=t0, t1=t1, t2=t2)
  df_results <- data.frame(var = c(rep('t0', nrow(df_test)), rep('t1', nrow(df_test)), rep('t2', nrow(df_test))),
                           value = c(t0[['predictions']],
                                     t1[['predictions']], t2[['predictions']]))
  h <- ggplot(df_results, aes(x=value, fill=var)) +
    geom_histogram(alpha=0.6, position='identity')
  print(h)
  
  assign_t <- c()
  for (row in 1:nrow(results)) {
    t0 <- results[row, 1]
    t1 <- results[row, 2]
    t2 <- results[row, 3]
    
    if(t0 > t1 && t0 > t2) {
      assign_t <- append(assign_t, 0)
    }
    else if(t1 > t0 && t1 > t2) {
      assign_t <- append(assign_t, 1)
    }
    else {
      assign_t <- append(assign_t, 2)
    }
  }
  
  df_test = df_test %>% mutate(t_opt = ifelse(y0 == 1, 0, 
                                              ifelse(y1 == 1, 1, ifelse(y2 == 1, 2, 'NA'))))
  
  eval_policy <- assign_t == df_test$t_opt
  return(sum(eval_policy)/length(eval_policy))
}

seeds <- c('1', '2', '3', '4', '5')
splits <- c('1', '2', '3', '4', '5')

seed <- '1'
dataset_type <- 'r0.06'
split <- '1'
df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))

driver_1va(df_train, df_test)

oosp_random <- c()
dataset_type <- '0.33'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    oosp_random <- append(oosp_random, driver_1va(df_train, df_test))
  }
}

oosp_r0.06 <- c()
dataset_type <- 'r0.06'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    oosp_r0.06 <- append(oosp_r0.06, driver_1va(df_train, df_test))
  }
}

oosp_r0.11 <- c()
dataset_type <- 'r0.11'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    oosp_r0.11 <- append(oosp_r0.11, driver_1va(df_train, df_test))
  }
}

results = data.frame(random=oosp_random, r0.06=oosp_r0.06, r0.11=oosp_r0.11)
colMeans(results)
transformed_results = data.frame()
for(i in 1:ncol(results)) {
  print(names(results)[i])
  buffer = data.frame(exp_design=rep(names(results)[i], 25), oosp=results[, i])
  transformed_results <- rbind(transformed_results, buffer)
}

ggplot(transformed_results, aes(x=exp_design, y=oosp)) + 
  geom_boxplot()

save_name = 'cf_1va_yhat_what'
setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/other_methods/results/CF/warfarin')
write.csv(results, glue('{save_name}_raw.csv'), row.names = FALSE)
ggsave(glue("{save_name}_oosp_boxplot.pdf"))
