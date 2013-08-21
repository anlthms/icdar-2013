# ICDAR2013 - Gender Prediction from Handwriting 
# Author: Anil Thomas
#

library(gbm)

logit <- function(pred) {
# Compute log-odds.
#
# Args:
#   pred    : a vector containing probability values in [0, 1). 
#
# Returns:
#   A vector containing log-odds.

    stopifnot(length(which(pred == 1)) == 0)
    return(log(pred / (1 - pred)))    
}

logistic <- function(logit, A, B) {
# Compute probabilities from log-odds. 
#
# Args:
#   logit   : a vector containing log-odds.
#   A,B     : parameters from Platt calibration. 
#
# Returns:
#   A vector containing probability values. 

    return(1 / (1 + exp(A * logit + B)))
}

calibrate <- function(pred) {
# Calibrate probabilities to minimize log loss.
#
# Args:
#   pred    : a vector containing probability values.
#
# Returns:
#   A vector containing calibrated probability values. 

    if (exists("useplatt")) {
        # NOTE: This is turned off by default.
        # The parameters A and B may be determined by using
        # gradient descent on CV data to optimize log loss.
        A <- -0.998 
        B <- -0.018 
    } else {
        A <- -1
        B <- 0
    }
    
    pred <- logistic(logit(pred), A, B)
    upper <- 0.9
    lower <- 0.1
    pred[pred > upper] <- (pred[pred > upper] + upper) / 2
    pred[pred < lower] <- (pred[pred < lower] + lower) / 2
    return(pred)
}

model <- function(train, target, test) {
# Train and make predictions
#
# Args:
#   train   : feature matrix for the training samples.
#   target  : target vector for the training samples.
#   test    : feature matrix for the test samples.
#
# Returns:
#   A vector containing predictions for each writer in the test set. 

    n.trees <- 80000
    gbm.obj <- gbm.fit(train, target, n.trees=n.trees,
                       interaction.depth=5, verbose=F)
    pred <- predict.gbm(gbm.obj, test, type="response", n.trees=n.trees)
    pred <- matrix(pred, nrow=2)

    # Average the predictions for the two samples (same text / not same text).
    return(colMeans(pred))
}

main <- function(testfile) {
# Load the data files, process them and save the predictions. 
#
# Args:
#   testfile    : name of the file containing test data. 

    train <- read.csv("train.csv", header=T)
    answers <- read.csv("train_answers.csv", header=T)
    test <- read.csv(testfile, header=T)
    train <- merge(answers,train)
    writers <- test$writer
    target <- train$male

    features <- scan("features.csv", what=character(), quiet=T)
    train <- train[, features]
    test <- test[, features]

    # There are 4 samples for each writer.
    stopifnot((nrow(test) / 4) == (max(writers) - min(writers) + 1))
    predsum <- rep(0, nrow(test) / 4)

    # Train on and predict Arabic and English samples separately. 
    for (j in c(1, 3)) {
        # This is to select rows 1,2,5,6,9,10... or 3,4,7,8,11,12...
        trainrows <- as.vector(matrix(seq(1:nrow(train)), nrow=4)[j:(j+1),])
        testrows <- as.vector(matrix(seq(1:nrow(test)), nrow=4)[j:(j+1),])

        pred <- model(train[trainrows,], target[trainrows], test[testrows,]) 
        predsum <- predsum + pred
    }

    # Average the predictions from Arabic and English samples.
    pred <- predsum / 2
    pred <- calibrate(pred)

    subm <- matrix(NA, (nrow(test) / 4), 2) 
    subm[,1] <- seq(min(writers), max(writers)) 
    subm[,2] <- round(pred, digits=4)
    write.table(subm, file="subm.csv", row.names=F, col.names=F, sep=",")
}

set.seed(1)
main("test.csv")
print("Done")
