# ANLY 534 - April 3rd, 2018

# Implementing a Shallow Neural Network from Scratch

# load data
library(keras)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# reformat train and test labels for binary classification problem:
train_labels <- ifelse(train_labels == 1, 1, 0)
test_labels <- ifelse(test_labels == 1, 1, 0)

# vectorize image training/test data
v_train_images <- matrix(ncol = dim(train_images)[1], nrow = 28*28) # empty matrix
dim(v_train_images) # matrix will store our vectorized image data
for(i in 1:dim(train_images)[1]){
        v_train_images[,i] <- c(train_images[i,,])
}

# vectorize image training/test data
v_test_images <- matrix(ncol = dim(test_images)[1], nrow = 28*28) # empty matrix
dim(v_test_images) # matrix will store our vectorized image data
for(i in 1:dim(test_images)[1]){
        v_test_images[,i] <- c(test_images[i,,])
}

# scale
X <- v_train_images/255
X_test <- v_test_images/255
Y <- matrix(train_labels, nrow = 1)
Y_test <- matrix(test_labels, nrow = 1)

# gut check:
train_images[1,14,14] # what is the analogous vectorized value?
v_train_images[378,1]

# Now that the data has been properly reshaped, we turn our attention to implementing
# the neural network...

# for a neural network model we want to know y-hat given x (input data)...

# we need to estimate 2 parameters: w and b

# recall that instead of the linear function y-hat = wx + b, in log. reg. we 
# actually want to to apply a sigmoid transformation to the linear function.

# for our neural network we need to build the structure of the network. We will
# implement a NN with 1 hidden layer (tanh activation) and an output layer with
# a sigmoid activation function.

# so, y-hat = sigmoid(wx + b) will actually give the proper output, because
# the sigmoid transforms the linear y-hat so it is between 0 and 1.

# recall the sigmoid function: 1/(1+e^-z) where z is the linear function above for log. reg.

# define sigmoid (activation for output layer) function
sigmoid <- function(z){
        s <- as.double(1/(1+exp(-z)), digits = 50)
        return(s)
}
sigmoid(-3)

# define tanh (activation for hidden layer) function
tanh <- function(z){
        s <- as.double((exp(z)-exp(-z))/(exp(z)+exp(-z)))
        return(s)
}
tanh(-3)

# now that we have our sigmoid and tanh functions we need a few more helper functions:

# 1. Neural Network Model Structure (input units, hidden units, etc.)
# 2. Initialize parameters
# 3. Cost Function
# 4. Forward Propagation Algorithm
# 5. Backward Propagation Algorithm
# 6. Optimizer (Gradient Descent Algorithm)

# layer sizes
layer_sizes <- function(X, Y){
        n_x <- dim(X)[1]
        n_h <- 25
        n_y <- dim(Y)[1]
        return(list(n_x, n_h, n_y))
}
layer_sizes(X, Y)

# initialize parameters
initialize_parameters <- function(n_x, n_h, n_y){
        set.seed(20)
        W1 <- matrix(runif(n_h*n_x)/100, nrow = n_h, ncol = n_x)
        b1 <- rep(0,n_h)
        W2 <- matrix(runif(n_y*n_h)/100, nrow = n_y, ncol = n_h)
        b2 <- rep(0, n_y)
        
        return(list(W1, b1, W2, b2))
}


# forward propagation function
forward_propagation <- function(X, parameters){
        
        # get params from parameters list
        W1 <- parameters[[1]]
        b1 <- parameters[[2]]
        W2 <- parameters[[3]]
        b2 <- parameters[[4]]
        
        # forward propagation
        Z1 <- ((W1 %*% X) + b1)
        A1 <- matrix(tanh(Z1), nrow = dim(Z1)[1], ncol = dim(Z1)[2])
        Z2 <- ((W2 %*% A1) + b2)
        A2 <- sigmoid(Z2)
        
        cache <- list(Z1, A1, Z2, A2)
        return(cache)
        
}

# Compute cost function
compute_cost <- function(A2, Y, parameters){
        
        # get m (number of examples)
        m <- dim(Y)[2]
        
        # Compute Cross-entropy Cost
        logprobs <- (log(A2)*Y) + (log(1-A2)*(1-Y))
        cost <- -1/m*sum(logprobs) # calculate loss function
        
        return(cost)
}

# backward propagation function
backward_propagation <- function(parameters, cache, X, Y){
        
        m <- dim(X)[2]
        
        # get params from parameters
        W1 <- parameters[[1]]
        W2 <- parameters[[3]]
        
        # get A1 and A2 from cache
        A1 <- cache[[2]]
        A2 <- cache[[4]]
        
        # calculate derivatives (dW1, db1, dW2, db2)
        dZ2 <- A2 - Y
        dW2 <- (dZ2 %*% t(A1))/m
        db2 <- sum(dZ2)/m
        dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)
        dW1 <- (dZ1 %*% t(X))/m
        db1 <- sum(dZ1)/m
        
        grads <- list(dW1, db1, dW2, db2)
        return(grads)
        
}

# optimizer (gradient descent)
update_parameters <- function(parameters, grads, learning_rate = 0.5){
        
        # Get parameters
        W1 <- parameters[[1]]
        b1 <- parameters[[2]]
        W2 <- parameters[[3]]
        b2 <- parameters[[4]]
        
        # Get gradients
        dW1 <- grads[[1]]
        db1 <- grads[[2]]
        dW2 <- grads[[3]]
        db2 <- grads[[4]]
        
        # Update rule for each parameter
        W1 <- W1 - learning_rate*dW1
        b1 <- b1 - learning_rate*db1
        W2 <- W2 - learning_rate*dW2
        b2 <- b2 - learning_rate*db2
        
        # return updated params
        params <- list(W1, b1, W2, b2)
        return(params)
        
}

# integrate all parts into nn_model function
nn_model <- function(X, Y, n_h, num_iterations = 100){
        
        set.seed(20)
        n_x <- unlist(layer_sizes(X, Y)[1])
        n_y <- unlist(layer_sizes(X, Y)[3])
        
        # initialize params
        parameters <- initialize_parameters(n_x, n_h, n_y)
        W1 <- parameters[[1]]
        b1 <- parameters[[2]]
        W2 <- parameters[[3]]
        b2 <- parameters[[4]]
        
        # Loop gradient descent
        for(i in 1:num_iterations){
                
                # forward prop
                cache <- forward_propagation(X, parameters)
                
                # compute cost
                cost <- compute_cost(cache[[4]], Y, parameters)
                
                # backward propagation
                grads <- backward_propagation(parameters, cache, X, Y)
                
                # update parameters via gradient descent
                parameters <- update_parameters(parameters, grads)
                
                # update every 10 iterations
                if(i %% 10 == 0){
                        print(paste0("Cost after iteration ", i, " = ", cost))
                } else {}
        }
        
        return(parameters)
        
}


# custom prediction function using model object
prediction <- function(parameters, X){
        
        # predict via forward propagation
        A2 <- forward_propagation(X, parameters)[[4]]
        
        # binary classification
        preds <- ifelse(A2 > 0.5, 1, 0)
        
        return(preds)
        
}

# run model
nn <- nn_model(X, Y, n_h = 25, num_iterations = 100)

# predict test/train examples
Y_pred_test <- prediction(nn, X_test)
Y_pred_train <- prediction(nn, X)

print(paste0("Train Accuracy was ", (100 - mean(abs(Y_pred_train - Y)) * 100), "%"))
print(paste0("Test Accuracy was ", (100 - mean(abs(Y_pred_test - Y_test)) * 100), "%"))
