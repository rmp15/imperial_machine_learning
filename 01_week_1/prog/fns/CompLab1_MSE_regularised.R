CompLab1.MSE.ridge <- function(x, y, PolyOrder) {
    
    # x and y are vectors containing the inputs and targets respectively
    # PolyOrder is the order of polynomial model used for fitting
    
    NumOfDataPairs <- length(x)
    
    # First construct design matrix of given order
    X      <- rep(1, NumOfDataPairs)
    dim(X) <- c(NumOfDataPairs, 1)
    
    for (n in 1:PolyOrder){
        X = cbind(X, x^n)
    }
    
    # Initialise CV variable for storing results
    CV = matrix(nrow=1, ncol=1)
    
    # Create training design matrix and target data, leaving one out each time
    Train_X <- X
    Train_y <- y
    
    # Create testing design matrix and target data
    Test_X <- X
    Test_y <- y
    
    # Create ridge regularisation matrix
    epsilon = 0.1
    R <- epsilon * diag(dim((t(Train_X) %*% Train_X))[1])
    
    # Learn the optimal paramerers using MSE loss
    Paras_hat <- solve( ((t(Train_X) %*% Train_X) + R) , t(Train_X) %*% Train_y)
    Pred_y    <- Test_X %*% Paras_hat;
    
    # Calculate the MSE of prediction using training data
    CV     <- (Pred_y - Test_y)^2
    
    # Result is MSE
    Results <- mean(CV)
    
    plot(x,Test_y); points(x,Pred_y,col='red')
    
    return(Results)
}

#temp <- ((t(Train_X) %*% Train_X) +diag(eigen(t(Train_X) %*% Train_X)$values))
#print(temp)
#Paras_hat <- solve( temp , t(Train_X) %*% Train_y)
