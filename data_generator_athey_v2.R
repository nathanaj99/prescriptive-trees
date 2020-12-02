# library(data.table)
# library(Publish)
# library(caret)
# library(sigmoid)
# library(rpart)

rm(list=ls())
graphics.off()
setwd("/Users/sina/Documents/GitHub/prescriptive-trees/data/Athey_v2_4000/")


##########################################################################################################
# Functions
##########################################################################################################
baseline <- function (x) {
  x = as.numeric(x)
  alpha_1 = 1/2
  alpha_2 = 1/2
  alpha_3 = 1
  
  value =  alpha_1*x[1] + alpha_2*x[2] + alpha_3*x[3]
  value
}



effect <- function (x) {
  x = as.numeric(x)
  alpha_4 = 0.5
  alpha_5 = 0.5
  value =  alpha_4 * x[1] + alpha_5 * x[2]
  
  value 
}




scalefunc <- function(x){
  y0_min_val = min(x$y0)
  y0_max_val = max(x$y0)
  x$y0 <- (x$y0 - y0_min_val)/(y0_max_val-y0_min_val)
  
  y1_min_val = min(x$y1)
  y1_max_val = max(x$y1)
  x$y1 <- (x$y1 - y1_min_val)/(y1_max_val-y1_min_val)
  
  x
}


##########################################################################################################
# Parameters
##########################################################################################################
# Choose the seeds
seeds = c(123,156,67,1,43)

N_training = 4000
N_test = 10000
N  = N_training + N_test
d = 3 #dimension of the data

# threshold = 0.6
# Run = 1

for(threshold in c(0.1,0.5,0.6,0.75,0.9)){
  for(Run in c(1,2,3,4,5)){
    set.seed(seeds[Run])
    ##########################################################################################################
    # Generating the data
    ##########################################################################################################
    #Generating odd and even columns from normal and bernoulli distribution respectiveley
    mat <- matrix(data = rbinom((N)*d, size=1, prob=0.5), nrow = N, ncol = d)*2 - 1
    
    data <- as.data.frame(mat)
    rm(mat)
    
    #Generating the true outcomes under treatment zero and one
    data$y0  =  apply(data, 1, function(x) baseline(x) - 0.5* effect(x) )
    data$y1  =  apply(data, 1, function(x) baseline(x) + 0.5* effect(x) )
    
    
    ###########################################################
    # Genreating the treatment each person receives
    ###########################################################
    # Finding the best possible treatment for each person
    data$best_treatment <- 1
    index <- data$y0 > data$y1
    data$best_treatment[index] <- 0
    # if prob_best <= threshold then that person would receive the best treatment
    data$prob_best <- runif(n=N,min=0,max=1)
    data$threshold <- threshold
    # assigning the treatment using prob_best and best_treatment
    data$t =  1 - data$best_treatment
    index <- data$prob_best <= threshold
    data$t[index] =  data$best_treatment[index]
    
    ###########################################################
    # generating the true prob_t P(t|x)
    ###########################################################
    data$prob_t = (1-abs(data$best_treatment - data$t))*threshold + abs(data$best_treatment - data$t)*(1-threshold)
    
    data$best_treatment <- NULL
    data$prob_best <- NULL
    data$threshold <- NULL
    
    
    ##########################################################################################################
    # Adding the noise to the  data
    ##########################################################################################################
    # Adding noise to the data y0 and y1
    data$y0= data$y0 + rnorm(N,mean = 0 , sd = 0.1)
    data$y1 = data$y1 + rnorm(N,mean = 0 , sd = 0.1)
    
    
    # cat("Let's see how often treatment is a better choice")
    # nrow(subset(data, data$y1 > data$y0))/nrow(data)*100
    # 
    # cat("Let's see how often treatment is a worse choice")
    # nrow(subset(data, data$y1 < data$y0))/nrow(data)*100
    # 
    # cat("Let's see how often both result in the same outcome")
    # nrow(subset(data, data$y1 == data$y0))/nrow(data)*100
    
    
    ##########################################################################################################
    # Splitting data into training and test and save the files
    ##########################################################################################################
    
    # Splitting training and test data
    data_train = data[1:N_training,]
    data_test = data[(N_training+1):N,]
    
    
    
    ##########################################################################################################
    # Scaling y0 and y1 to [0,1]
    ##########################################################################################################
    data_train <- scalefunc(data_train)
    data_test <- scalefunc(data_test)
    
    ##########################################################################################################
    # Generating the outcome column y
    ##########################################################################################################
    data_train$y = data_train$t*data_train$y1 + (1-data_train$t)*data_train$y0 
    data_test$y = data_test$t*data_test$y1 + (1-data_test$t)*data_test$y0 
    
    # Some minor steps
    data_train$t <- as.factor(as.character(data_train$t))
    data_test$t <- as.factor(as.character(data_test$t))
    
    ##########################################################################################################
    # Learning propensity score P(t=1|x) for each entry
    ##########################################################################################################
    t_train_data = data_train[,!(names(data_train) %in% c("y0","y1","y","prob_t"))]
    t_test_data = data_test[,!(names(data_test) %in% c("y0","y1","y","prob_t"))]
    
    glm.fit <- glm(t ~ ., data = t_train_data, family = "binomial")
    
    data_train$prop_score_1 <- predict(glm.fit,newdata = t_train_data,type = "response")
    data_test$prop_score_1 <- predict(glm.fit,newdata = t_test_data,type = "response")
    rm(t_train_data,t_test_data)
    
    data_train$prob_t_pred_log <- as.numeric(as.character(data_train$t))*data_train$prop_score_1 + (1-as.numeric(as.character(data_train$t)))*(1-data_train$prop_score_1)
    data_train$prop_score_1 <- NULL
    
    
    data_test$prob_t_pred_log <- as.numeric(as.character(data_test$t))*data_test$prop_score_1 + (1-as.numeric(as.character(data_test$t)))*(1-data_test$prop_score_1)
    data_test$prop_score_1 <- NULL
    
    rm(glm.fit)
    
    ##########################################################################################################
    # Learning propensity score P(t=1|x) for each entry using decision tree
    ##########################################################################################################
    t_train_data = data_train[,!(names(data_train) %in% c("y0","y1","y","prob_t","prob_t_pred_log"))]
    t_test_data = data_test[,!(names(data_test) %in% c("y0","y1","y","prob_t","prob_t_pred_log"))]
    
    #model <- rpart(t ~ ., data = t_train_data, method = "class", control = rpart.control(maxdepth = 4, minsplit = 20, cp=0.01))
    train_control<- trainControl(method="repeatedcv", number=10, repeats = 3)
    model.cv <- train(t ~ ., 
                      data = t_train_data,
                      method = "rpart",
                      trControl = train_control)
    
    model <- model.cv$finalModel
    
    data_train$prop_score_1  <- predict(model, t_train_data, type = "prob")[,2]
    data_test$prop_score_1  <- predict(model, t_test_data, type = "prob")[,2]
    rm(t_train_data,t_test_data)
    
    data_train$prob_t_pred_tree <- as.numeric(as.character(data_train$t))*data_train$prop_score_1 + (1-as.numeric(as.character(data_train$t)))*(1-data_train$prop_score_1)
    data_train$prop_score_1 <- NULL
    
    
    data_test$prob_t_pred_tree <- as.numeric(as.character(data_test$t))*data_test$prop_score_1 + (1-as.numeric(as.character(data_test$t)))*(1-data_test$prop_score_1)
    data_test$prop_score_1 <- NULL
    
    # par(xpd = TRUE)
    # plot(model, compress = TRUE)
    # text(model, use.n = TRUE)
    # 
    # summary(data_train$prob_t_pred_log)
    # summary(data_train$prob_t_pred_tree)
    # summary(data_train$prob_t_pred_log - data_train$prob_t)
    # summary(data_train$prob_t_pred_tree - data_train$prob_t)
    rm(model)
    
    ##########################################################################################################
    # Save the filess
    ##########################################################################################################
    
    # Save files
    write.csv(data_train,paste("data_train_",toString(threshold),'_',toString(Run),".csv",sep=''),row.names = FALSE)
    write.csv(data_test,paste("data_test_",toString(threshold),'_',toString(Run),".csv",sep=''),row.names = FALSE)

  }
}
