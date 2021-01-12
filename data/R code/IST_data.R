# library(data.table)
# library(Publish)
# library(caret)
# library(sigmoid)
# library(rpart)

rm(list=ls())
graphics.off()
setwd("/Users/sina/Documents/GitHub/prescriptive-trees/data/IST_5000/")


##########################################################################################################
# Parameters
##########################################################################################################
# Choose the seeds
seeds = c(123,156,67,1,43)

training_portion = 0.75
# Run = 1
# set.seed(seeds[Run])


##########################################################################################################
# read data
##########################################################################################################
data <- read.csv("/Users/sina/Documents/GitHub/prescriptive-trees/Direct_Approach/cleaned_IST_enc.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
data$t <- as.factor(data$t)


for(Run in c(1,2,3,4,5)){
  set.seed(seeds[Run])
  ##########################################################################################################
  # Splitting data into training and test 
  ##########################################################################################################
  ## 75% of the sample size
  # smp_size <- floor(training_portion * nrow(data))
  smp_size = 5000
  
  ## set the seed to make your partition reproducible
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  
  data_train <- data[train_ind, ]
  data_test <- data[-train_ind, ]
  
  
  ##########################################################################################################
  # Learning propensity score P(t|x) for each entry using decision tree
  ##########################################################################################################
  t_train_data = data_train[,!(names(data_train) %in% c("y","y0","y1","y2","y3","y4","y5"))]
  t_test_data = data_test[,!(names(data_test) %in% c("y","y0","y1","y2","y3","y4","y5"))]
  
  
  train_control<- trainControl(method="repeatedcv", number=10, repeats = 3)
  model.cv <- train(t ~ ., 
                    data = t_train_data,
                    method = "rpart",
                    trControl = train_control)
  
  model <- model.cv$finalModel
  
  data_train$prob_t_pred_tree <- NA
  data_test$prob_t_pred_tree <- NA
  for(t in levels(data$t)){
    index <- data_train$t == t
    data_train$prob_t_pred_tree[index]  <- predict(model, t_train_data, type = "prob")[index,t]
    
    index <- data_test$t == t
    data_test$prob_t_pred_tree[index]  <- predict(model, t_test_data, type = "prob")[index,t]
  }
  
  rm(t_train_data,t_test_data)
  
  # par(xpd = TRUE)
  # plot(model, compress = TRUE)
  # text(model, use.n = TRUE)
  
  rm(model,model.cv)
  
  ##########################################################################################################
  # Save the files
  ##########################################################################################################
  
  # Save files
  # write.csv(data_train,paste("data_train_",toString(Run),".csv",sep=''),row.names = FALSE)
  # write.csv(data_test,paste("data_test_",toString(Run),".csv",sep=''),row.names = FALSE)
}

st = ""
for(s in names(data)){
  s = paste('\'',s,'\'',sep = "")
  st = paste(st,",",s,sep = "")
}
