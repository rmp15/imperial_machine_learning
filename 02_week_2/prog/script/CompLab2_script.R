trm(list=ls())

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
    results.mse[i,] <- c(i,CompLab2.Classification(dat.train[,c(1:2)],dat.train[,3],dat.test[,c(1:2)],dat.test[,3],i))
}

# plot results
pdf(paste0(path,'output/order_against_error.pdf'))
#plot(results.mse$order,results.mse$error.train,t='l')
plot(results.mse$order,results.mse$error.test,col='red',t='l')
dev.off()

pdf(paste0(path,'output/order_against_loglikelihood.pdf'))
#plot(results.mse$order,results.mse$logJointProb,t='l')
plot(results.mse$order,results.mse$logJointProb_Test,t='l')
dev.off()

# part B

# load functions
source(paste0(path,'prog/fns/CompLab2_Classification_LOOCV.R'))

results.loocv <- data.frame(order=numeric(0),mean.ll=numeric(0),sd.ll=numeric(0),meantr.ll=numeric(0),sdtr.ll=numeric(0),meantot.ll=numeric(0),sdtot.ll=numeric(0))
for(i in c(1:poly.max)){
    results.loocv[i,] <- c(i,CompLab2.Classification_LOOCV(dat.test[,c(1:2)],dat.test[,3],i))
}

# save output (because it takes quite a long time to run)
saveRDS(results.loocv,paste0(path,'output/order_against_loocvll'))

pdf(paste0(path,'output/order_against_loocvll.pdf'))
plot(results.loocv$order,results.loocv$meantr.ll,t='l')
dev.off()
 
