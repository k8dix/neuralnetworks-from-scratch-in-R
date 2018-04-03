# ANLY 534 - March 27th, 2018

# Neural Network Approach to Logistic Regression (Gradient Descent Intuition)

# Note: as a general approach, we want to avoid for loops to make the code
# as efficient as possible, so we will use linear algebra and vectorization
# to perform most computations here.

# load data
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# reformat train and test labels for binary classification problem:
train_labels <- ifelse(train_labels == 1, 1, 0)
test_labels <- ifelse(test_labels == 1, 1, 0)

        # basically, we are going to train a logistic regression model to 
        # classify the digit images as a 1 or not a 1

# Last week we saw that we could represent a 28x28 image as a column vector with 
# 28x28 = 784 values. In this example, we have 60k training images. If we want to
# "vectorize" the 3D tensor of image data (60,000 x 28 x 28 array), we will end up
# with a 784 x 60,000 matrix, where each image is stored as a column vector.

# vectorize image training/test data
v_train_images <- matrix(ncol = dim(train_images)[1], nrow = 28*28) # empty matrix
dim(v_train_images) # matrix will store our vectorized image data
for(i in 1:dim(train_images)[1]){
        v_train_images[,i] <- c(train_images[i,,])
}

        # gut check:
        train_images[1,14,14] # what is the analogous vectorized value?
        v_train_images[378,1]
        
# Now that the data has been properly reshaped, we turn our attention to implementing
# logistic regression...
        
# for logistic reg. we want to know y-hat given x (input data)...

# we need to estimate 2 parameters: w and b
        
# recall that instead of the linear function y-hat = wx + b, in log. reg. we 
# actually want to to apply a sigmoid transformation to the linear function.
        
# so, y-hat = sigmoid(wx + b) will actually give the proper log. reg. output, because
# the sigmoid transforms the linear y-hat so it is between 0 and 1.
        
# recall the sigmoid function: 1/(1+e^-z) where z is the linear function above for log. reg.

# define sigmoid (activation) function
sigmoid <- function(z){
        s <- as.double(1/(1+exp(-z)), digits = 50)
        return(s)
}
sigmoid(-59343)

# now that we have our sigmoid function we need a few more helper functions:

        # 1. Cost Function
        # 2. Forward/Backward Propagation Algorithm
        # 3. Optimizer (Gradient Descent Algorithm)

# initialize parameters
initialize_with_zeros <- function(dim){
        w <- matrix(rep(0, dim), nrow = dim, ncol = 1)
        b <- 0
        return(list(w, b))
}

# initialize objects for logistic regression
w <- initialize_with_zeros(dim(v_train_images)[1])[[1]]
b <- initialize_with_zeros(dim(v_train_images)[1])[[2]]
X <- v_train_images/255 # scaled
Y <- matrix(train_labels, nrow = 1)
dim(w)
dim(b)
dim(X)
dim(Y)

# forward and backward propagation function
propagate <- function(w, b, X, Y){
        
        # forward propagation (compute cost)
        m <- dim(X)[2] # number of training examples
        A <- sigmoid((t(w) %*% X) + b) # predict values based on weights and training data
        cost <- -1/m*sum((Y*log(A))+((1-Y)*log(1-A))) # calculate loss function
        
        # backward propagation (compute derivatives)
        dw <- (1/m)*(X %*% t(A-Y))
        db <- (1/m)*sum(A-Y)
        return(list(dw, db, cost))
        
}
propagate(w, b, X, Y)

# optimizer (gradient descent)
optimize <- function(w, b, X, Y, iterations, learning_rate){
        
        # empty costs vector
        costs <- NULL
        
        # loop through iterations to optimize against cost
        for(i in 1:iterations){
                # Cost and gradient calculation
                p <- propagate(w, b, X, Y)
                
                # get derivatives from p
                dw <- p[[1]]
                db <- p[[2]]
                cost <- p[[3]]
                
                # Update Rule:
                w <- w - (learning_rate*dw)
                b <- b - (learning_rate*db)
                
                # Record Costs
                if(i %% 10 == 0){
                        
                        # record every 10 iterations
                        costs[i/10] <- cost
                        
                        # print every 10 iterations
                        print(paste0("Cost after iteration ", i, " is ", cost))
                } else {
                        
                }
                
        }
        
        params <- list(w, b)
        output <- list(params, costs)
        return(output)
        
}

model <- optimize(w, b, X, Y, 120, .01)

w <- model[[1]][[1]]
b <- model[[1]][[2]]
dim(w)

# custom prediction function using model object
prediction <- function(w, b, X){
        
        m <- dim(X)[2]
        Y_pred <- matrix(rep(0, m), nrow = 1, ncol = m)
        
        A <- sigmoid((t(w) %*% X) + b)
        
        Y_pred <- ifelse(A <= 0.5, 0, 1)
        
        return(Y_pred)
        
}

preds <- prediction(w, b, X)

# combine all into model function
log_reg_model <- function(X_train, Y_train, X_test, Y_test, num_iterations = 150, learning_rate = 0.5){
        
        # initialize weights
        w <- initialize_with_zeros(dim(X_train)[1])[[1]]
        b <- initialize_with_zeros(dim(X_train)[1])[[2]]
        
        # gradient descent
        mod <- optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
        
        # store final parameter estimates
        w <- model[[1]][[1]]
        b <- model[[1]][[2]]
        
        # predict test/train examples
        Y_pred_test <- prediction(w, b, X_test)
        Y_pred_train <- prediction(w, b, X_train)
        
        print(paste0("Train Accuracy was ", (100 - mean(abs(Y_pred_train - Y_train)) * 100), "%"))
        print(paste0("Test Accuracy was ", (100 - mean(abs(Y_pred_test - Y_test)) * 100), "%"))
        
}

# run model

# vectorize image training data
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

X <- v_train_images/255
X_test <- v_test_images/255
Y <- matrix(train_labels, nrow = 1)
Y_test <- matrix(test_labels, nrow = 1)

log_reg_model(X, Y, X_test, Y_test)

