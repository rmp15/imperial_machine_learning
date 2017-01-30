CompLab2.Classification_LOOCV <- function(x, y, PolyOrder) {

# x and y are column vectors containing the training inputs and targets respectively
# xt and yt are column vectors containing the testing inputs and targets respectively
# PolyOrder is the order of polynomial model used for fitting
 
NumOfDataPairs_Train <- nrow(x)

logJointProb <- matrix(NA, nrow=NumOfDataPairs_Train,ncol=1)
logJointProb_Test <- matrix(NA, nrow=NumOfDataPairs_Train,ncol=1)
logJointProb_Total <- matrix(NA, nrow=NumOfDataPairs_Train,ncol=1)


#for (m in 1:NumOfDataPairs_Train){
    for (m in 1:NumOfDataPairs_Train){


    # First construct design matrix of given order
    X       <- rep(1, NumOfDataPairs_Train-1)
    X       <- rep(1, NumOfDataPairs_Train-1)
    Xt      <- rep(1, 1)
    dim(Xt) <- c(1, 1)

    for (n in 1:PolyOrder){
        X  = cbind(X, x[-m,]^n)
        Xt = cbind(Xt, x[m,]^n)
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
        Theta = solve(H, t(X)%*%(A%*%X%*%Theta + y[-m] - P) )
        
        # Compute new likelihood and unormalised posterior values for training data
        f             = X%*%Theta # train
        logPrior      = sum( dnorm(Theta, rep(0, NumOfParas), rep(alpha, NumOfParas), log=TRUE) )
        logLikelihood = t(f)%*%y[-m] - sum(log(1+exp(f))) # training likelihood
        logJointProb[m]  = logLikelihood #+ logPrior
        
        #cat("Iteration ", n, ": Log joint probability is ", logJointProb[m], "\n")

    }
    cat("Polynomial order ",PolyOrder," run ", m, " out of ", NumOfDataPairs_Train,"\n")
    cat("Final parameters are: ", Theta, "\n")


    # Compute new likelihood and unormalised posterior values for testing data
    ft                 = Xt%*%Theta # test
    logPrior_Test      = sum( dnorm(Theta, rep(0, NumOfParas), rep(alpha, NumOfParas), log=TRUE) )
    logLikelihood_Test = t(ft)%*%y[m] - sum(log(1+exp(ft))) # testing likelihood
    logJointProb_Test[m]  = logLikelihood_Test# + logPrior_Test
    
    # Total likelihood of both training and test
    logJointProb_Total[m] <- logJointProb[m] + logJointProb_Test[m]
    
}

logJointProb_Mean = mean(logJointProb)
logJointProb_sd = sd(logJointProb)
logJointProb_Mean_Test = mean(logJointProb_Test)
logJointProb_sd_Test = sd(logJointProb_Test)
logJointProb_Total_Mean = mean(logJointProb_Total)
logJointProb_Total_sd = sd(logJointProb_Total)

return(c(logJointProb_Mean,logJointProb_sd,logJointProb_Mean_Test,logJointProb_sd_Test,logJointProb_Total_Mean,logJointProb_Total_sd))


}
