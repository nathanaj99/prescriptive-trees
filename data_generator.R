# I generate the data through following sections:
# 
# Section 1
# In this section we generate synthetic data in the following fashion:
# 1- We pick a seed number
# 2- we genereate the data with n= 1000 (training) + 60000(test) datapoints
# # x_ij ~ N(0,1) if j is odd
# # x_ij ~ Bernoulli(0.5) if j is even
# # j \in {1,...,20}
# 3- y_0(x) = baseline(x) - 0.5 effect(x) #This is the true outcome under treatment 0
# 4- y_1(x) = baseline(x) + 0.5 effect(x) #This is the true outcome under treatment 1
# 5- if P(t=1|x) = [1+exp(-y_0(x))]^-1 > 0.5 then t=1 else t=0 # This is the treatment that each datapoint receives
# 6- For the training set we add noise eps_i ~ N(mean = 0  , sd = 0.1) to the outcome
# 
# 
# Section 2
# For the training data, we fit a logisitc regression model to learn the propensity score P(t=1|x) for each entry; we fit the model only on the training data. Column  prop_score shows the predicted propensity score. The true propensity score is [1+exp(-y_0(x))]^-1
# 
# 
# Section 3
# 
# In this section we categorize the odd columns which are derived from standard normal distribution:
# (q_25 = qnrom(0.25,mean=0,sd=1) )
# 1- first for each column x we create 4 binary columns x_1, x_2, x_3 and x_4
# if x <= q_25 then x_1 = x_2 = x_3 = x_4 = 1
# else x <= q_50 then x_2 = x_3 = x_4 = 1
# else x <= q_75 then x_3 = x_4 = 1
# else x_4 = 1 
# 
# I have  generated 5 set of (training,test) datasets. For each set I have also included the original data where I haven’t binarized the continues columns.


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
Run = 5
set.seed(seeds[Run])


N_training = 1000
N_test = 60000
N  = N_training + N_test
d = 20 #dimension of the data


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
  value =  (x[1] > 1)*5 - 5
  
  value 
}


##########################################################################################################
# Generating the data
##########################################################################################################
#Generating odd and even columns from normal and bernoulli distribution respectiveley
odd_cols <- matrix(data = rnorm((N_training + N_test)*d/2, mean=0, sd=1), nrow = N, ncol = d/2)
even_cols <- matrix(data = rbinom((N_training + N_test)*d/2,size =1, prob = 0.5), nrow = N, ncol = d/2)


mat <- matrix(NA, nrow=N, ncol=d)
mat[, seq(1, d, 2)] <- odd_cols
mat[, seq(2, d, 2)] <- even_cols

data <- as.data.frame(mat)
rm(odd_cols,even_cols,mat)

#Generating the true outcomes under treatment zero and one
data$y0  =  apply(data, 1, function(x) baseline(x) - 0.5* effect(x) )
data$y1  =  apply(data, 1, function(x) baseline(x) + 0.5* effect(x) )

# Genreating the treatment each person receives
data$t = 0
index <- sigmoid(data$y0) > 0.5
data$t[index] <- 1


# Generating the outcome column y
data$y = data$t*data$y1 + (1-data$t)*data$y0 

# Some minor steps
data$t <- as.factor(as.character(data$t))
for(i in seq(2, d, 2)){
  data[,i] <- as.factor(as.character(data[,i]))
}



# Adding noise to the training data y (the first 1000 indices)
index = 1:N_training
data$y[index] = data$y[index] + rnorm(N_training,mean = 0 , sd = 0.1)


##########################################################################################################
# Learning propensity score P(t=1|x) for each entry
##########################################################################################################
t_train_data = data[1:N_training,!(names(data) %in% c("y0","y1","y"))]
t_test_data = data[,!(names(data) %in% c("y0","y1","y"))]

glm.fit <- glm(t ~ ., data = t_train_data, family = "binomial")
data$prop_score <- predict(glm.fit,newdata = t_test_data,type = "response")
rm(t_train_data,t_test_data)



##########################################################################################################
# Binarized the odd columns
##########################################################################################################
#First we turn each continues column into a categorical feature with 4 different levels depending on the quantile that value falls in
data_enc = data
for(i in seq(1, d, 2)){
  x = names(data)[i]
  data_enc[[x]] = cut(data_enc[[x]],c(-Inf,qnorm(0.25,0,1),qnorm(0.5,0,1),qnorm(0.75,0,1),Inf),labels=c(1,2,3,4))
}

#Now we tuurn all categorical  features into one-hot vectors
dmy <- dummyVars(" ~ .", data = data_enc)
data_enc <- data.frame(predict(dmy, newdata = data_enc))

#Because of the previous  steps even the binary columns got affected and now we have some unnecessary column  that we should remove
#if a feature has only two levels we should only keep one column
#As our convention, we always keep the second one
cols = c()
tmp <- gsub("\\..*","",names( data_enc ))
for(name in names(data)){
  a = tmp == name
  if(sum(a)==2){
    cols <- append(cols, max(which(a == TRUE)))
  }else{
    cols <- append(cols, which(a == TRUE))
  }
}

data_enc <- data_enc[,cols]
for(i in seq(1,d/2*4 + d/2,1)){
  data_enc[,i] <- as.factor(data_enc[,i])
}

# Taking care of  the integer columns : If x_ij = 1 then x_i(j+1) should be one as well  for odd i's

odd_features = names(data)[seq(1,d,2)]
for(v in odd_features){
  for(i in seq(2,4,1)){
    a =  as.numeric(as.character(data_enc[[paste(v,toString(i),sep = ".")]]))
    b =  as.numeric(as.character(data_enc[[paste(v,toString(i-1),sep = ".")]]))
    data_enc[[paste(v,toString(i),sep = ".")]] =  as.numeric(a|b)
    
  }
}



##########################################################################################################
# Splitting data into training and test and save the files
##########################################################################################################

# Splitting training and test data
data_train = data[1:N_training,]
data_test = data[(N_training+1):N,]

data_train_enc= data_enc[1:N_training,]
data_test_enc = data_enc[(N_training+1):N,]


# Save files
write.csv(data_train,paste("data_train_",toString(Run),".csv",sep=''),row.names = FALSE)
write.csv(data_test,paste("data_test_",toString(Run),".csv",sep=''),row.names = FALSE)

write.csv(data_train_enc,paste("data_train_enc_",toString(Run),".csv",sep=''),row.names = FALSE)
write.csv(data_test_enc,paste("data_test_enc_",toString(Run),".csv",sep=''),row.names = FALSE)





