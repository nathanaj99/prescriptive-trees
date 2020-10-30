# library(data.table)
# library(Publish)
# library(caret)
# library(sigmoid)

rm(list=ls())
graphics.off()
setwd("/Users/sina/Documents/GitHub/prescriptive-trees/data/")

##########################################################################################################
# Parameters
##########################################################################################################
# Choose the seeds
seeds = c(123,156,67,1,43)
N_train_set = c(500,500,500,500,500)
Run = 5
set.seed(seeds[Run])
threshold = 0.9

N_training = N_train_set[Run]
N_test = 10000
N  = N_training + N_test
d = 2 #dimension of the data


##########################################################################################################
# Functions
##########################################################################################################
baseline <- function (x) {
  x = as.numeric(x)
  value =  0.5*x[1] + x[2] 
  value
}



effect <- function (x) {
  x = as.numeric(x)
  value =  x[1]*0.5
  
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


binarize <- function(data_enc, data_names, d){
  data_enc$t <- as.numeric(as.character(data_enc$t))
  for(i in seq(1, 2, 1)){
    x = data_names[i]
    data_enc[[x]] = cut(data_enc[[x]],
                        c(-Inf,qnorm(0.1,0,1),qnorm(0.2,0,1),qnorm(0.3,0,1),qnorm(0.4,0,1),qnorm(0.5,0,1),qnorm(0.6,0,1),qnorm(0.7,0,1),qnorm(0.8,0,1),qnorm(0.9,0,1), Inf),
                        labels=c(1,2,3,4,5,6,7,8,9,10))
  }
  
  
  #Now we tuurn all categorical  features into one-hot vectors
  dmy <- dummyVars(" ~ .", data = data_enc)
  data_enc <- data.frame(predict(dmy, newdata = data_enc))
  
  # Taking care of  the integer columns : If x_ij = 1 then x_i(j+1) should be one as well  for odd i's
  features = data_names[seq(1,d,1)]
  for(v in features){
    for(i in seq(2,10,1)){
      a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
      b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
      data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
      
    }
  }
  
  data_enc
}


##########################################################################################################
# Generating the data
##########################################################################################################
#Generating odd and even columns from normal and bernoulli distribution respectiveley
mat <- matrix(data = rnorm((N)*d, mean=0, sd=1), nrow = N, ncol = d)

data <- as.data.frame(mat)
rm(mat)

#Generating the true outcomes under treatment zero and one
data$y0  =  apply(data, 1, function(x) baseline(x) - 0.5* effect(x) )
data$y1  =  apply(data, 1, function(x) baseline(x) + 0.5* effect(x) )


##########################################################################################################
# Adding the noise to the  data
##########################################################################################################
# Adding noise to the data y0 and y1
data$y0= data$y0 + rnorm(N,mean = 0 , sd = 0.1)
data$y1 = data$y1 + rnorm(N,mean = 0 , sd = 0.1)


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
t_train_data = data_train[,!(names(data_train) %in% c("y0","y1","y","prop_score_t"))]
t_test_data = data_test[,!(names(data_test) %in% c("y0","y1","y","prop_score_t"))]

glm.fit <- glm(t ~ ., data = t_train_data, family = "binomial")

data_train$prop_score_1 <- predict(glm.fit,newdata = t_train_data,type = "response")
data_test$prop_score_1 <- predict(glm.fit,newdata = t_test_data,type = "response")
rm(t_train_data,t_test_data)

data_train$prob_t_pred <- as.numeric(as.character(data_train$t))*data_train$prop_score_1 + (1-as.numeric(as.character(data_train$t)))*(1-data_train$prop_score_1)
data_train$prop_score_1 <- NULL


data_test$prob_t_pred <- as.numeric(as.character(data_test$t))*data_test$prop_score_1 + (1-as.numeric(as.character(data_test$t)))*(1-data_test$prop_score_1)
data_test$prop_score_1 <- NULL



##########################################################################################################
# Binarization of the columns
##########################################################################################################
#First we turn each continues column into a categorical feature with 4 different levels depending on the quantile that value falls in
data_train_enc <- binarize(data_train, names(data_train), d)
data_test_enc <- binarize(data_test, names(data_test), d)
##########################################################################################################
# Save the filess
##########################################################################################################

# Save files
write.csv(data_train,paste("data_train_",toString(threshold),'_',toString(Run),".csv",sep=''),row.names = FALSE)
write.csv(data_test,paste("data_test_",toString(threshold),'_',toString(Run),".csv",sep=''),row.names = FALSE)


write.csv(data_train_enc,paste("data_train_enc_",toString(threshold),'_',toString(Run),".csv",sep=''),row.names = FALSE)
write.csv(data_test_enc,paste("data_test_enc_",toString(threshold),'_',toString(Run),".csv",sep=''),row.names = FALSE)


