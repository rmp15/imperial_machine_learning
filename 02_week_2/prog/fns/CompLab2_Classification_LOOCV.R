CompLab2.Classification_LOOCV <- function(x, y, PolyOrder) {

# x and y are column vectors containing the training inputs and targets respectively
# xt and yt are column vectors containing the testing inputs and targets respectively
# PolyOrder is the order of polynomial model used for fitting

NumOfDataPairs <- length(x)

for (m in 1:NumOfDataPairs){

    # First construct design matrix of given order
    X       <- rep(1, NumOfDataPairs-1)
    dim(X)  <- c(NumOfDataPairs-1, 1)
    Xt      <- rep(1, 1)
    dim(Xt) <- c(1, 1)

    for (n in 1:PolyOrder){
        X  = cbind(X, x[-m]^n)
        Xt = cbind(Xt, x[m]^n)
    }

    NumOfParas = ncol(X)

    # Set the prior precision
    alpha = 10

    # Newton routine to find MAP values of Theta
    # Fix number of steps to 10 and initial estimate to Theta = 0
    N_Steps = 10;
    Theta   = matrix(0, nrow=NumOfParas, ncol=1);
    
    for (n in 1:N_Steps){
        # Newton Step
        P     = 1./(1 + exp(-X %*% Theta))
        A     = diag(P[,1]*(1-P[,1]))
        H     = t(X) %*% A %*% X + diag(NumOfParas)/alpha
        Theta = solve(H, t(X)%*%(A%*%X%*%Theta + y[-m] - P) ) # this line updates theta using newton's method (pg 30)
        
        # Compute new likelihood and unnormalised posterior values for training data
        f             = X%*%Theta # train
        logPrior      = sum( dnorm(Theta, rep(0, NumOfParas), rep(alpha, NumOfParas), log=TRUE) )
        logLikelihood = t(f)%*%y[-m] - sum(log(1+exp(f))) # training likelihood
        logJointProb  = logLikelihood + logPrior

        cat("Iteration ", n, ": Log joint probability is ", logJointProb, "\n")
    }

    cat("\nFinal parameters are: ", Theta, "\n")

    # Compute new likelihood and unormalised posterior values for testing data
    ft                 = Xt%*%Theta # test
    logPrior_Test      = sum( dnorm(Theta, rep(0, NumOfParas), rep(alpha, NumOfParas), log=TRUE) )
    logLikelihood_Test = t(ft)%*%y[m] - sum(log(1+exp(ft))) # testing likelihood
    logJointProb_Test  = logLikelihood_Test + logPrior_Test

    # Total log-likelihood
    logJointProb_Total[m] = logJointProb + logJointProb_Test

# Calculate percentage of training errors
#Train_Error = 100 - 100*sum( (1/(1+exp(-X%*%Theta)) > 0.5) == y)/NumOfDataPairs_Train
#Train_Error = 100 - 100*sum( (1/(1+exp(-X%*%Theta)) > 0.5))/NumOfDataPairs

# Calculate percentage of training errors
#Test_Error = 100 - 100*sum( (1/(1+exp(-Xt%*%Theta)) > 0.5) == yt)/NumOfDataPairs_Test
#Test_Error[m] = 100 - 100*sum( (1/(1+exp(-Xt%*%Theta)) > 0.5))/NumOfDataPairs

#cat("Percentage training error: ", Train_Error, "\n")
#cat("Percentage testing error: ", Test_Error, "\n")

}

logJointProb_Mean = mean(logJointProb_Total)
logJointProb_sd = sd(logJointProb_Total)

return(c(logJointProb_Mean,logJointProb_sd))

}
