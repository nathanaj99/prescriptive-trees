library(policytree)
library(grf)
library(tictoc)
library(glue)
library(ggplot2)
library(dplyr)

setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/data/processed/synthetic/')

driver = function(df_train, df_test) {
  set.seed(16)
  X_train <- data.matrix(df_train[ , names(df_train) %in% c('V1', 'V2')])
  y_train <- data.matrix(df_train[c('y')])
  t_train <- data.matrix(df_train[c('t')])
  
  X_test <- data.matrix(df_test[ , names(df_test) %in% c('V1', 'V2')])
  y_test <- data.matrix(df_test[c('y')])
  t_test <- data.matrix(df_test[c('t')])
  
  start <- Sys.time()
  # ---- Uncomment for causal trees ----
  # forest <- causal_forest(X_train, y_train, t_train, num.trees=1)
  forest <- causal_forest(X_train, y_train, t_train)
  pred <- predict(forest, X_test)
  
  assign_t <- c()

  for (row in 1:nrow(pred)) {
    val <- pred[row, 1]
    # print(val)
    if(val > 0) {
      assign_t <- append(assign_t, 1)
    }
    else {
      assign_t <- append(assign_t, 0)
    }
  }
  
  df_test$treatment <- assign_t
  time_elapsed <- Sys.time() - start
  
  df_test = df_test %>% mutate(t_opt = ifelse(y0 > y1, 0, 1))
  df_test = df_test %>% mutate(y_opt = ifelse(y0 > y1, y0, y1))
  df_test = df_test %>% mutate(realized_outcome = ifelse(treatment == 0, y0, 
                                                        ifelse(treatment == 1, y1,
                                                               ifelse(treatment == 2, y2, NA))))
  regret <- df_test$y_opt - df_test$realized_outcome
  
  eval_policy <- assign_t == df_test$t_opt
  list_return <- list('oos_regret' = sum(regret),
                      'oosp' = sum(eval_policy)/length(eval_policy), 'time' = time_elapsed)
  return(list_return)
  # return(mean(df_test$realized_outcome))
}

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
    
    res <- driver(df_train, df_test)
    
    oosp_0.1 <- append(oosp_0.1, res$oosp)
    time_0.1 <- append(time_0.1, res$time)
    oosr_0.1 <- append(oosr_0.1, res$oos_regret)
    
    # oosp_0.1 <- append(oosp_0.1, driver(df_train, df_test))
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
    
    res <- driver(df_train, df_test)
    
    oosp_0.25 <- append(oosp_0.25, res$oosp)
    time_0.25 <- append(time_0.25, res$time)
    oosr_0.25 <- append(oosr_0.25, res$oos_regret)
    
    # oosp_0.25 <- append(oosp_0.25, driver(df_train, df_test))
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
    
    res <- driver(df_train, df_test)
    
    oosp_0.5 <- append(oosp_0.5, res$oosp)
    time_0.5 <- append(time_0.5, res$time)
    oosr_0.5 <- append(oosr_0.5, res$oos_regret)
    
    # oosp_0.5 <- append(oosp_0.5, driver(df_train, df_test))
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
    
    res <- driver(df_train, df_test)
    
    oosp_0.75 <- append(oosp_0.75, res$oosp)
    time_0.75 <- append(time_0.75, res$time)
    oosr_0.75 <- append(oosr_0.75, res$oos_regret)
    
    # oosp_0.75 <- append(oosp_0.75, driver(df_train, df_test))
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
    
    res <- driver(df_train, df_test)
    
    oosp_0.9 <- append(oosp_0.9, res$oosp)
    time_0.9 <- append(time_0.9, res$time)
    oosr_0.9 <- append(oosr_0.9, res$oos_regret)
    
    # oosp_0.9 <- append(oosp_0.9, driver(df_train, df_test))
  }
}

results_synthetic = data.frame(p0.1=oosp_0.1, p0.25=oosp_0.25, p0.5=oosp_0.5, p0.75=oosp_0.75,
                               p0.9=oosp_0.9, oosr_0.1=oosr_0.1, oosr_0.25=oosr_0.25, oosr_0.5=oosr_0.5, oosr_0.75=oosr_0.75,
                               oosr_0.9=oosr_0.9, time_0.1=time_0.1, time_0.25=time_0.25, time_0.5=time_0.5,
                               time_0.75=time_0.75, time_0.9=time_0.9)
setwd('/Users/nathanjo/Documents/Github/prescriptive-trees/results/synthetic/compiled/CF')
write.csv(results_synthetic, glue('cf_raw.csv'), row.names = FALSE)
