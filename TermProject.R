getwd()
setwd("C:/Users/sahba/Desktop/SFU/Spring2023/CMPT 318/Term Project")

library(plyr)
library(dplyr)
library(lubridate)
library(devtools)
remotes::install_github("vqv/ggbiplot")
library(zoo)
require(ggbiplot)
library(depmixS4)
library(ggplot2)


###############
## functions ##
###############

# takes the dataframe and it will
# extract the chosen time window.
extract_time_window <- function(dataframe) {
  weekday = "Monday"

  # Convert the Date column to readable dates
  dataframe$Date <- as.Date(dataframe$Date, format = "%d/%m/%Y")
  
  # add a new column for the day of the week
  dataframe$DayOfWeek <- weekdays(as.Date(dataframe$Date))
  
  # Fixing the format of Time column
  dataframe$Time <- as.POSIXct(dataframe$Time, format = '%H:%M:%S')
  
  # Replace NA values by interpolation
  dataframe <- dataframe %>% mutate(across(c('Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'), ~ na.approx(.x, na.rm = FALSE, rule=2)))
  
  # Define the minimum and maximum time window duration in minutes
  start_time <- as.POSIXct("13:00:00", format = "%H:%M:%OS")
  end_time <- as.POSIXct("16:00:00", format = "%H:%M:%OS")
  
  # # Extracting the time window
  time_window <- dataframe %>%
    filter(DayOfWeek == weekday, Time >= start_time, Time <= end_time)
  time_window <- time_window[c('Time', 'Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3')]
  
  return(time_window)
}

# Prepare the new dataset for PCA by taking the mean of the 
# entire dataset w.r.t. each value in the time window
get_resp_PCA <- function(time_window) {

  mean_time <- aggregate(.~Time, data=time_window, FUN=function(x) c(mean=mean(x)))
  mean_time <- subset(mean_time, select = -c(Time))
  
  # Standardize the data set
  data_resp <- mean_time %>% mutate_at(c('Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'), ~(scale(.) %>% as.vector))
  
  return(data_resp)
}

# Select the features and rename them for later uses
get_resp <- function(time_window, feature_choices) {
  
  data_resp <- time_window[feature_choices]
  
  # Standardize the data set
  data_resp <- data_resp %>% mutate_at(feature_choices, ~(scale(.) %>% as.vector))
  
  colnames(data_resp)[1] <- 'feat1'
  colnames(data_resp)[2] <- 'feat2'
  
  return(data_resp)
}

# Function to calculate pca and variance percentage
compute_pca_variance <- function(data_resp){
  
  
  # PCA calculation
  pca <- prcomp(data_resp, scale=TRUE)
  
  # Principal components summary
  summary(pca)
  
  # Calculate the percentage of variance
  var_per <-pca$sdev^2
  var_per <-round(var_per/sum(var_per)*100,1)
  
  result <- list ("pca" = pca, "var_per" = var_per)
  return (result)
}
  


#-----------------------------------------------
# 2. HMM Training and Testing

# Function to train various multivariate HMMs on the train data (data_resp) with different
# numbers of states (num_states).
# This function returns the best models
find_best_models <- function(data_resp, num_states, ntime) {
  # Calculate the number of time windows (number of observations)
  sequence_num = nrow(data_resp) %/% ntime
  
  # Create a list of observation lengths
  ntimes <- rep(ntime, sequence_num)
  
  # create a list to save the models, loglikes, and BIC values
  train_results <- list()
  
  # looping through the number of states
  for (i in num_states) {
    model <- depmix(list(feat1~1, feat2~1), data=data_resp, nstates=i,
                    family= list(gaussian(),gaussian()), ntimes=ntimes)
    
    # fit the model by calling fit
    fmodel <- fit(model)
    
    # avoiding NULL models
    if (!is.null(fmodel)) {
      loglik <- logLik(fmodel) / sequence_num
      bic <- BIC(fmodel)
      
      # Save the results
      train_results[[i]] <- list(model=fmodel, loglik=loglik, bic=bic, nstates=i)
    }
  }
  
  # Select the best performing models based on BIC and overall fit on train data
  valid_models <- Filter(function(x) !is.null(x$bic), train_results)
  
  # Choosing the best minimum BIC as the best BIC
  best_bic <- min(sapply(valid_models, function(x) x$bic))
  
  # Find models with minimum BIC
  best_models <- lapply(valid_models, function(x) if (x$bic == best_bic && !is.null(x$model)) x)
  best_models <- Filter(Negate(is.null), best_models)
  
  return(best_models)
}

# Calculate the log likelihood of the data
calculate_likelihood <- function(model, data_resp, ntime) {
  sequence_num = nrow(data_resp) %/% ntime
  ntimes <- rep(ntime, sequence_num)
  
  infer_model <- depmix(list(feat1~1, feat2~1), data=data_resp, nstates=model$nstates,
                        family= list(gaussian(),gaussian()), ntimes=ntimes)
  infer_model <- setpars(infer_model, getpars(model$model))
  
  # forwardbackward and calculating the loglike
  fb <- forwardbackward(infer_model)
  return(fb$logLike / sequence_num)
}

# A function to calculate the log-likelihood of the test data for best
# models to decide on the best one among these candidates
# This function returns the best model
get_best_model <- function(best_models, data_resp, ntime) {
  # Make a vector to save the loglikes
  log_likelihoods <- vector("list", length = length(best_models))

  for (i in seq_along(best_models)) {
    log_likelihoods[[i]] <- calculate_likelihood(best_models[[i]], data_resp, ntime)
  }

  # choosing the maximum loglike as the best one
  max_log_likelihood <- max(unlist(log_likelihoods))
  
  # finding the model with maximum loglike
  best_model <- best_models[[which.max(unlist(log_likelihoods))]]
  logLikRet <- list(testLogLik = max_log_likelihood, trainLogLik = best_model$loglik)

  return(list(model=best_model, logLikRet=logLikRet))
}


#####################
## Using functions ##
#####################
#-------
# part 1

df <- read.table("Term_Project_Dataset.txt", header = T, sep = ",")
time_window <- extract_time_window(df)

pca_resp <- get_resp_PCA(time_window)

# PCA calculation
pca_var <- compute_pca_variance (pca_resp)
pca <- pca_var$pca
p_var <-pca_var$var_per
pca
summary (pca)
p_var

# Plot the scree plot to figure out which PC works better
barplot(p_var,main='Scree Plot', xlab="Principal Component", ylab="Percent Variation")

ggbiplot(pca, obs.scale = 1, var.scale = 1)

loading_scores <- pca$rotation[,1]
resp_rank <- sort(abs(loading_scores), decreasing = TRUE)
pca$rotation[names(resp_rank), 1]

feature_choices = c('Voltage', 'Global_intensity')

#-------
# part 2

# Plot the mean features
ggplot(pca_resp[feature_choices], aes(x = 1:nrow(pca_resp), y = Voltage)) +
  geom_line() +
  labs(x = "Minutes counted from the start of the time frame", y = feature_choices[1])

ggplot(pca_resp[feature_choices], aes(x = 1:nrow(pca_resp), y = Global_intensity)) +
  geom_line() +
  labs(x = "Minutes counted from the start of the time frame", y = feature_choices[2])

# Choose the range of nmber of states for the models
num_states <- 6:10

train_test_resp <- get_resp (time_window, feature_choices)

# Get the length of each time window (length of each observation)
window_sample_num <- nrow(pca_resp)

# Get the total length of the new dataset and find the point to divide 
# train (80%) and test (20%) sets
all_sample_num <- nrow(train_test_resp)
train_range <- ((all_sample_num * 8) %/% 10) %/% window_sample_num * window_sample_num

# Divide the dataset into train and test sets
train_resp <- train_test_resp[1:train_range, ]
test_resp <- train_test_resp[(train_range + 1):all_sample_num, ]

# Display the data distributions to select the distribution families for the HMMs
hist(train_resp$feat1)
hist(train_resp$feat2)

# Calling the function to find the best models
train_best_models <- find_best_models(train_resp, num_states, window_sample_num)

# Calling the function to find the best model among our candidates
test_ret <- get_best_model (train_best_models, test_resp, window_sample_num)

# Best model
test_best_model <- test_ret$model

# Train and test log likelihood
test_ret$logLikRet


#-------------------------------------
# 3. Anomaly Detection

# Calculate the likelihood of dataset 1
df1 <- read.table("Dataset_with_Anomalies_1.txt", header = T, sep = ",")
time_window1 <- extract_time_window(df1)
df1_resp <- get_resp (time_window1, feature_choices)
likelihood1 <- calculate_likelihood(test_best_model, df1_resp, window_sample_num)
likelihood1

# Calculate the likelihood of dataset 2
df2 <- read.table("Dataset_with_Anomalies_2.txt", header = T, sep = ",")
time_window2 <- extract_time_window(df2)
df2_resp <- get_resp (time_window2, feature_choices)
likelihood2 <- calculate_likelihood(test_best_model, df2_resp, window_sample_num)
likelihood2

# Calculate the likelihood of dataset 3
df3 <- read.table("Dataset_with_Anomalies_3.txt", header = T, sep = ",")
time_window3 <- extract_time_window(df3)
df3_resp <- get_resp (time_window3, feature_choices)
likelihood3 <- calculate_likelihood(test_best_model, df3_resp, window_sample_num)
likelihood3


#plotting the loglike of datasets with anomaly to have a better understanding of the result
library(ggplot2)

# Store the log-likelihood values in a data frame
df <- data.frame(Dataset = c("1", "2", "3"), LogLikelihood = c(likelihood1, likelihood2, likelihood3))

# Create the bar chart
ggplot(df, aes(x = Dataset, y = LogLikelihood, fill = Dataset)) + 
  geom_bar(stat = "identity") +
  ggtitle("Log-Likelihood of Datasets") +
  xlab("Dataset") +
  ylab("Log-Likelihood") +
  scale_fill_manual(values = c("deeppink4", "aquamarine3", "deepskyblue4"))

