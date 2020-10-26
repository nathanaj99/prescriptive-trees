library(data.table)
library(Publish)
library(caret)
library(sigmoid)

rm(list=ls())
graphics.off()
setwd("/Users/sina/Documents/GitHub/prescriptive-trees/data/")

##########################################################################################################
# Parameters
##########################################################################################################
# Choose the seeds
seeds = c(123,156,67,1,43)
N_train_set = c(100,100,100,100,100)
Run = 5
set.seed(seeds[Run])


N_training = N_train_set[Run]
N_test = 60000
N  = N_training + N_test
d = 10 #dimension of the data


##########################################################################################################
# Functions
##########################################################################################################
baseline <- function (x) {
  x = as.numeric(x)
  value =  x[1] + x[3] + x[5] + x[7] + x[8] + x[9] -2
  value
}



effect <- function (x) {
  x = as.numeric(x)
  value =  (x[1] >= 1)*5 - 5
  
  value 
}


setTreatment <- function (x) {
  x = as.numeric(x)
  treatment = rbinom(1,size =1, prob = sigmoid(x))
  
  treatment
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
# Generating the data
##########################################################################################################
#Generating odd and even columns from normal and bernoulli distribution respectiveley
odd_cols <- matrix(data = rbinom((N)*d/2,size =1, prob = 0.5), nrow = N, ncol = d/2) * 2 -1
even_cols <- matrix(data = rbinom((N)*d/2,size =1, prob = 0.5), nrow = N, ncol = d/2)


mat <- matrix(NA, nrow=N, ncol=d)
mat[, seq(1, d, 2)] <- odd_cols
mat[, seq(2, d, 2)] <- even_cols

data <- as.data.frame(mat)
rm(odd_cols,even_cols,mat)

#Generating the true outcomes under treatment zero and one
data$y0  =  apply(data, 1, function(x) baseline(x) - 0.5* effect(x) )
data$y1  =  apply(data, 1, function(x) baseline(x) + 0.5* effect(x) )

# Genreating the treatment each person receives
data$t =  apply(data,1, function(x) setTreatment(x[which(colnames(data)=="y0")]))


data$prop_score_t = data$t*sigmoid(data$y0) + (1-data$t)*(1-sigmoid(data$y0))


##########################################################################################################
# Adding the noise to the  data
##########################################################################################################
# Adding noise to the data y0 and y1
data$y0= data$y0 + rnorm(N,mean = 0 , sd = 0.1)
data$y1 = data$y1 + rnorm(N,mean = 0 , sd = 0.1)


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
for(i in seq(2, d, 2)){
  data_train[,i] <- as.factor(as.character(data_train[,i]))
  data_test[,i] <- as.factor(as.character(data_test[,i]))
}




##########################################################################################################
# Learning propensity score P(t=1|x) for each entry
##########################################################################################################
t_train_data = data_train[,!(names(data_train) %in% c("y0","y1","y","prop_score_t"))]
t_test_data = data_test[,!(names(data_test) %in% c("y0","y1","y","prop_score_t"))]

glm.fit <- glm(t ~ ., data = t_train_data, family = "binomial")

data_train$prop_score_1 <- predict(glm.fit,newdata = t_train_data,type = "response")
data_test$prop_score_1 <- predict(glm.fit,newdata = t_test_data,type = "response")
rm(t_train_data,t_test_data)

data_train$prop_score_t_pred <- as.numeric(as.character(data_train$t))*data_train$prop_score_1 + (1-as.numeric(as.character(data_train$t)))*(1-data_train$prop_score_1)
data_train$prop_score_1 <- NULL


data_test$prop_score_t_pred <- as.numeric(as.character(data_test$t))*data_test$prop_score_1 + (1-as.numeric(as.character(data_test$t)))*(1-data_test$prop_score_1)
data_test$prop_score_1 <- NULL



##########################################################################################################
# Save the filess
##########################################################################################################

# Save files
write.csv(data_train,paste("data_train",toString(Run),"N",toString(N_training),".csv",sep='_'),row.names = FALSE)
write.csv(data_test,paste("data_test",toString(Run),"N",toString(N_test),".csv",sep='_'),row.names = FALSE)



