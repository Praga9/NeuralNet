# This script implements neural network functions developed in MatLab for the
# Stanford ML course in R.

# *****TEMPORARY: Clear Workspace and Load Test Data***********
rm(list = ls())
source("loadTestData.R")
# *************************************************************

# Sigmoid Function
# Computes J, the sigmoid of z
sigmoid <- function(z)  1 / (1 + exp(-z))


# NNPredict 
# Outputs a prediction based on input values x. Theta1 and Theta2
# are vectors of the trained weights of a hidden layer and the output layer.
# The prediction is a matrix in which the first column is the prediction and
# the second is the percentage.
NNPredict <- function(x, Theta1, Theta2) {
    x <- as.matrix(x)
    Theta1 <- as.matrix(Theta1)
    Theta2 <- as.matrix(Theta2)
    x  <- cbind(rep(1, nrow(x)), x)
    a2 <- sigmoid(x %*% t(Theta1))
    a2 <- cbind(rep(1, nrow(a2)), a2)
    a3 <- a2 %*% t(Theta2)
    h  <- sigmoid(a3)
    cbind(apply(h, 1, which.max), apply(h, 1, max))
}
#NOTES: To be done: generalize for layers > 2.


# NNCost
# Calculates the regularized cost J for a two-layer
# neural network. Input values are y, a vector of m class labels; x, an m x n
# matrix of predictor training data; k is the number of classes; and
# lambda, the regularization parameter.
NNCost <- function(y, x, k, Theta1, Theta2, lambda = 0){
    
    # Calculate prediction values
    x <- data.matrix(x)
    Theta1 <- data.matrix(Theta1)
    Theta2 <- data.matrix(Theta2)
    x  <- cbind(rep(1, nrow(x)), x)
    a2 <- sigmoid(x %*% t(Theta1))
    a2 <- cbind(rep(1, nrow(a2)), a2)
    z <- a2 %*% t(Theta2)
    a3  <- sigmoid(z)
    
    # Reshape y and prediction into a k x m matrix
    if(class(y) == "dataframe") y <- data.matrix(y)
    y_matrix <- matrix(0, k, length(y))
    index <- cbind(y, 1:length(y))
    y_matrix[index] <- 1
    pred <- matrix(0, k, length(y))
    x <- x[, 2:ncol(x)]
    index <- cbind(NNPredict(x, Theta1, Theta2)[,1], 1:length(y))
    pred[index] <- 1
    
    # Calculate regularization term
    reg <- (lambda / (2 * length(y))) * (sum(sum(Theta1[, 2:ncol(Theta1)]^2)) +
                                             sum(sum(Theta2[, 2:ncol(Theta2)]^2)))
    # Return J
    reg + (sum(sum((-y_matrix %*% log(a3)) - 
                       ((1 - y_matrix) %*% log(1 - a3))))) / length(y)
}
