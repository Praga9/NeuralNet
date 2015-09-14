# This file loads test data from csv files to validate the R implementation of
# the neural net functions.

setwd("~/Personal/Projects/NeuralNet")
Theta2 <- read.csv("T2.csv", header = FALSE)
Theta1 <- read.csv("T1.csv", header = FALSE)
y <- as.numeric(readLines(con = "y_data.csv"))
X <- read.csv("X_data.csv", header = FALSE)
