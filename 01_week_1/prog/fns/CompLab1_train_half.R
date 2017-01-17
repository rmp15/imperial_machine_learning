CompLab1.train_half <- function(x, y, PolyOrder) {
    
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
    
    # choose half of the dataset randomly
    set.seed(1231)
    n = sample(1:NumOfDataPairs, round(NumOfDataPairs/2))
    
    # Create training design matrix and target data, leaving half out each time
    Train_X <- X[-n,]
    Train_y <- y[-n]
    
    # Create testing design matrix and target data
    Test_X <- X[n, ]
    Test_y <- y[n]
    
    # Learn the optimal paramerers using MSE loss
    Paras_hat <- solve( t(Train_X) %*% Train_X , t(Train_X) %*% Train_y)
    Pred_y_in    <- Train_X %*% Paras_hat;
    Pred_y_out    <- Test_X %*% Paras_hat;
    
    # Calculate the MSE of prediction using training data
    CV_in     <- (Pred_y_in - Train_y)^2
    CV_out     <- (Pred_y_out - Test_y)^2

    # Result is MSE
    Results_in <- mean(CV_in)
    Results_out <- mean(CV_out)

    Results <- cbind(Results_in, Results_out)

plot(x,y); points(x[-n],Pred_y_in,col='green'); points(x[n],Pred_y_out,col='red')

    return(Results)
}
