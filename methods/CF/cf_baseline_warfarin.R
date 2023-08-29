library(policytree)
library(grf)
library(tictoc)
library(glue)
library(ggplot2)
library(dplyr)
library(plyr)

setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/data/processed/warfarin/')

run = function(df_train, df_test, from_vector, dataset_type, seed) {
  seed1 <- 16
  set.seed(seed1)
  X_train <- data.matrix(df_train[ , !(names(df_train) %in% c('y', 't', 'y0',
                                                              'y1', 'y2', 'prob_t_pred_tree',
                                                              'ml0', 'ml1', 'ml2', 'y_pred',
                                                              'White', 'lrrf0', 'lrrf1', 'lrrf2',
                                                              'lr0', 'lr1', 'lr2'))])
  y_train <- data.matrix(df_train[c('y')])
  y_hat <- data.matrix(df_train[c('ml0', 'ml1', 'ml2')])
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
                                 to=c(1, 0))
  
  t_train <- data.matrix(df_train[c('t_buffer')])
  
  Y.forest = regression_forest(X_train, y_train)
  y_hat = predict(Y.forest)$predictions
  
  # --- Comment for causal tree ---
  forest <- causal_forest(X_train, y_train, t_train, Y.hat=y_hat, W.hat=w_hat)
  
  if(dataset_type == '0.33'){
    forest <- causal_forest(X_train, y_train, t_train, Y.hat=y_hat, W.hat=w_hat)
  }
  else if(dataset_type == 'r0.06'){
    df_train = df_train %>% mutate(weights = ifelse((t == 0 & y == 0)|(t==2 & y == 0), 15, 1))
    forest <- causal_forest(X_train, y_train, t_train, Y.hat=y_hat, W.hat=w_hat, sample.weights=df_train[['weights']])
  }
  else if(dataset_type == 'r0.11'){
    if((seed == '2')){
      df_train = df_train %>% mutate(weights = ifelse((t == 0 & y == 0), 100, 1))
    }
    else if (seed == '4'){
      df_train = df_train %>% mutate(weights = ifelse((t == 0 & y == 0), 1000, 1))
    }
    else{
      df_train = df_train %>% mutate(weights = ifelse((t == 0 & y == 0)|(t==2 & y == 0), 15, 1))
    }
    forest <- causal_forest(X_train, y_train, t_train, Y.hat=y_hat, W.hat=w_hat, sample.weights=df_train[['weights']])
  }
  
  # --- Uncomment for causal tree ---
  # forest <- causal_forest(X_train, y_train, t_train, Y.hat=y_hat, W.hat=w_hat, sample.weights=df_train[['weights']], num.trees=1)

  pred <- predict(forest, X_test)
  
  return(pred)
}



driver_baseline = function(df_train, df_test, dataset_type, seed) {
  h0 <- df_train[df_train$t == 0, ]
  # print(as.data.frame(table(h0[['y']])))
  
  h1 <- df_train[df_train$t == 1, ]
  # print(as.data.frame(table(h1[['y']])))
  
  h2 <- df_train[df_train$t == 2, ]
  # print(as.data.frame(table(h2[['y']])))
  
  start <- Sys.time()
  # t = 1 vs 0
  df_train_1 <- df_train[df_train$t != 2, ]
  t1 = run(df_train_1, df_test, c(1, 0), dataset_type, seed)

  # t = 2 vs 0
  df_train_1 <- df_train[df_train$t != 1, ]
  t2 = run(df_train_1, df_test, c(2, 0), dataset_type, seed)

  results = data.frame(t0=rep(0, nrow(df_test)), t1=t1, t2=t2)

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
  
  time_elapsed <- Sys.time() - start
  
  df_test = df_test %>% mutate(t_opt = ifelse(y0 == 1, 0, 
                                              ifelse(y1 == 1, 1, ifelse(y2 == 1, 2, 'NA'))))
  
  
  df_test$treatment <- assign_t
  df_test = df_test %>% mutate(realized_outcome = ifelse(treatment == 0, y0, 
                                                         ifelse(treatment == 1, y1,
                                                                ifelse(treatment == 2, y2, NA))))
  
  df_test = df_test %>% mutate(y_opt = ifelse((y0 > y1 && y0 > y2), y0, 
                                              ifelse((y1 > y0 && y1 > y2), y1, ifelse((y2 > y0 && y2 > y1), y2, 'NA'))))
  regret <- df_test$y_opt - df_test$realized_outcome
  
  # print(as.data.frame(table(assign_t)))
  # print(as.data.frame(table(df_test[['t_opt']])))
  eval_policy <- assign_t == df_test$t_opt
  list_return <- list('oos_regret' = sum(regret),
                      'oosp' = sum(eval_policy)/length(eval_policy), 'time' = time_elapsed)
  return(list_return)
}

seeds <- c('1', '2', '3', '4', '5')
splits <- c('1', '2', '3', '4', '5')

seed <- '2'
dataset_type <- 'r0.06'
split <- '1'
df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))

driver_baseline(df_train, df_test, dataset_type, seed)

oosp_random <- c()
time_random <- c()
oosr_random <- c()
dataset_type <- '0.33'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    res <- driver_baseline(df_train, df_test, dataset_type, seed)
    
    oosp_random <- append(oosp_random, res$oosp)
    time_random <- append(time_random, res$time)
    oosr_random <- append(oosr_random, res$oos_regret)
    
    # oosp_random <- append(oosp_random, driver_baseline(df_train, df_test, dataset_type, seed))
  }
}

oosp_r0.06 <- c()
time_r0.06 <- c()
oosr_r0.06 <- c()
dataset_type <- 'r0.06'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    res <- driver_baseline(df_train, df_test, dataset_type, seed)
    
    oosp_r0.06 <- append(oosp_r0.06, res$oosp)
    time_r0.06 <- append(time_r0.06, res$time)
    oosr_r0.06 <- append(oosr_r0.06, res$oos_regret)
    
    # oosp_r0.06 <- append(oosp_r0.06, driver_baseline(df_train, df_test, dataset_type, seed))
  }
}
mean(oosp_r0.06)

oosp_r0.11 <- c()
time_r0.11 <- c()
oosr_r0.11 <- c()
dataset_type <- 'r0.11'
for (seed in seeds) {
  for (split in splits) {
    df_train <- read.csv(glue('seed{seed}/data_train_{dataset_type}_{split}.csv'))
    df_test <- read.csv(glue('seed{seed}/data_test_{dataset_type}_{split}.csv'))
    
    res <- driver_baseline(df_train, df_test, dataset_type, seed)
    
    oosp_r0.11 <- append(oosp_r0.11, res$oosp)
    time_r0.11 <- append(time_r0.11, res$time)
    oosr_r0.11 <- append(oosr_r0.11, res$oos_regret)
    
    # oosp_r0.11 <- append(oosp_r0.11, driver_baseline(df_train, df_test, dataset_type, seed))
  }
}

results = data.frame(random=oosp_random, r0.06=oosp_r0.06, r0.11=oosp_r0.11,
                     random_oos_regret=oosr_random, 
                     r0.06_oos_regret=oosr_r0.06, r0.11_oos_regret=oosr_r0.11,
                     time_random=time_random, time_r0.06=time_r0.06, time_r0.11=time_r0.11)

# colMeans(results)
transformed_results = data.frame()
for(i in 1:ncol(results)) {
  print(names(results)[i])
  buffer = data.frame(exp_design=rep(names(results)[i], 25), oosp=results[, i])
  transformed_results <- rbind(transformed_results, buffer)
}

ggplot(transformed_results, aes(x=exp_design, y=oosp)) + 
  geom_boxplot()

save_name = 'cf_baseline_untuned'
setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/results/warfarin/compiled/CF')
write.csv(results, glue('{save_name}_raw.csv'), row.names = FALSE)
ggsave(glue("{save_name}_oosp_boxplot.pdf"))
