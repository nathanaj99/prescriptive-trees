# library(data.table)
# library(Publish)
# library(caret)
# library(sigmoid)
# library(rpart)

rm(list=ls())
graphics.off()



##########################################################################################################
# Parameters
##########################################################################################################
# Choose the seeds
seeds = c(123,156,67,1,43)



for(data_seed in c(1,2,3,4,5)){
  for(r in c(1,2,3)){
    ##########################################################################################################
    # read data
    ##########################################################################################################
    data_list = c("warfarin_0.33.csv","warfarin_r0.06.csv","warfarin_r0.11.csv")
    data_enc_list = c("warfarin_enc_0.33.csv","warfarin_enc_r0.06.csv","warfarin_enc_r0.11.csv")
    threshold_list = c("0.33","r0.06","r0.11")
    
    path = paste("/Users/sina/Documents/GitHub/prescriptive-trees/data/Warfarin_v2/seed",toString(data_seed),"/",sep = "")
    data_path = paste("/Users/sina/Documents/GitHub/prescriptive-trees/Direct_Approach/seed",toString(data_seed),"/",data_list[r],sep = "")
    data_enc_path = paste("/Users/sina/Documents/GitHub/prescriptive-trees/Direct_Approach/seed",toString(data_seed),"/",data_enc_list[r],sep = "")
    
    setwd(path)
    data <- read.csv(data_path, header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
    data_enc <- read.csv(data_enc_path, header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
    data$t <- as.factor(data$t)
    data_enc$t <- as.factor(data_enc$t)
    threshold = threshold_list[r]
    
    for(Run in c(1,2,3,4,5)){
      ## set the seed to make your partition reproducible
      set.seed(seeds[Run])
      ##########################################################################################################
      # Splitting data into training and test
      ##########################################################################################################
      smp_size = 3000
      
      if((data_seed==1 | data_seed == 3) & r==3){
        rare_index <- (data$t == 2 & data$y ==0)
        prob = rep(1/nrow(data), nrow(data))
        prob[rare_index]=1
        train_ind <- sample(seq_len(nrow(data)), size = smp_size, prob = prob)
      }else{
        train_ind <- sample(seq_len(nrow(data)), size = smp_size)
      }
      

      data_train <- data[train_ind, ]
      data_test <- data[-train_ind, ]


      data_train_enc <- data_enc[train_ind, ]
      data_test_enc <- data_enc[-train_ind, ]
      
    
      
      ##########################################################################################################
      # Learning propensity score P(t|x) for each entry using decision tree
      ##########################################################################################################
      t_train_data = data_train[,!(names(data_train) %in% c("y","y0","y1","y2"))]
      t_test_data = data_test[,!(names(data_test) %in% c("y","y0","y1","y2"))]


      train_control<- trainControl(method="repeatedcv", number=10, repeats = 3)
      model.cv <- train(t ~ .,
                        data = t_train_data,
                        method = "rpart",
                        trControl = train_control)

      model <- model.cv$finalModel

      data_train_enc$prob_t_pred_tree <- NA
      data_test_enc$prob_t_pred_tree <- NA
      for(t in levels(data$t)){
        index <- data_train$t == t
        data_train_enc$prob_t_pred_tree[index]  <- predict(model, t_train_data, type = "prob")[index,t]
        data_train$prob_t_pred_tree[index]  <- predict(model, t_train_data, type = "prob")[index,t]

        index <- data_test$t == t
        data_test_enc$prob_t_pred_tree[index]  <- predict(model, t_test_data, type = "prob")[index,t]
        data_test$prob_t_pred_tree[index]  <- predict(model, t_test_data, type = "prob")[index,t]
      }

      rm(t_train_data,t_test_data)

      # par(xpd = TRUE)
      # plot(model, compress = TRUE)
      # text(model, use.n = TRUE)

      rm(model,model.cv,train_control)

      ##########################################################################################################
      # Save the files
      ##########################################################################################################

      # Save files
      write.csv(data_train_enc,paste("data_train_enc_",toString(threshold),"_",toString(Run),".csv",sep=''),row.names = FALSE)
      write.csv(data_test_enc,paste("data_test_enc_",toString(threshold),"_",toString(Run),".csv",sep=''),row.names = FALSE)
      write.csv(data_train,paste("data_train_",toString(threshold),"_",toString(Run),".csv",sep=''),row.names = FALSE)
      write.csv(data_test,paste("data_test_",toString(threshold),"_",toString(Run),".csv",sep=''),row.names = FALSE)
    }
  }
}

# data <- read.csv("data_train_enc_1.csv", header = TRUE, sep = ",",na.strings = "",stringsAsFactors = TRUE)
# st = ""
# for(s in names(data)){
#   s = paste('\'',s,'\'',sep = "")
#   st = paste(st,",",s,sep = "")
# }