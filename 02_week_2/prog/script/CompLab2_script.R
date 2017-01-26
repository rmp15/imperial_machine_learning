rm(list=ls())

# load dataset
path <- '~/git/tutorials/imperial_machine_learning/02_week_2/'
dat.train <- read.table(paste0(path,'data/data_train.txt'))
dat.test <- read.table(paste0(path,'data/data_test.txt'))

# upper limit of polynomial test
poly.max <- 20

# Q1

# part A

# load functions
source(paste0(path,'prog/fns/CompLab2_Classification.R'))

results.mse <- data.frame(order=numeric(0),error.train=numeric(0),error.test=numeric(0),logJointProb=numeric(0),logJointProb_Test=numeric(0))
for(i in c(1:poly.max)){
    results.mse[i,] <- c(i,CompLab2.Classification(dat.train[,1],dat.train[,2],dat.test[,1],dat.test[,2],i))
}

# plot results
pdf(paste0(path,'output/order_against_error.pdf'))
plot(results.mse$order,results.mse$error.train,t='l')
lines(results.mse$order,results.mse$error.test,col='red')
dev.off()

pdf(paste0(path,'output/order_against_loglikelihood.pdf'))
#plot(results.mse$order,results.mse$logJointProb,t='l')
plot(results.mse$order,results.mse$logJointProb_Test,t='l')
dev.off()

# part B

# load functions
path <- '~/git/tutorials/imperial_machine_learning/02_week_2/'
source(paste0(path,'prog/fns/CompLab2_Classification_LOOCV.R'))

results.loocv <- data.frame(order=numeric(0),mean.ll=numeric(0),sd.ll=numeric(0))
for(i in c(1:poly.max)){
    results.loocv[i,] <- c(i,CompLab2.Classification_LOOCV(dat.test[,1],dat.test[,2],i))
}

pdf(paste0(path,'output/order_against_loocvll.pdf'))
plot(results.loocv$order,results.loocv$mean.ll,t='l')
dev.off()
