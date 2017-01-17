# load functions
path <- '~/git/tutorials/imperial_machine_learning/01_week_1/'
source(paste0(path,'prog/fns/CompLab1_Generate_Data.R'))

# create dataset
dat <- CompLab1.Generate_Data()

# upper limit of polynomial test
poly.max <- 10

# Q1

# part A
source(paste0(path,'prog/fns/CompLab1_MSE.R'))

results.mse <- data.frame(order=numeric(0),MSE=numeric(0))
for(i in c(1:poly.max)){
    results.mse[i,] <- c(i,CompLab1.MSE(dat[,1],dat[,2],i))
    print(c(i,CompLab1.MSE(dat[,1],dat[,2],i)))
}

# part B
source(paste0(path,'prog/fns/CompLab1_train_half.R'))

results.train_half <- data.frame(order=numeric(0),MSE=numeric(0))
for(i in c(1:poly.max)){
    results.train_half[i,] <- c(i,CompLab1.train_half(dat[,1],dat[,2],i))
    print(c(i,CompLab1.train_half(dat[,1],dat[,2],i)))
}

# Q2
source(paste0(path,'prog/fns/CompLab1_LOOCV.R'))

results.LOOCV <- data.frame(order=numeric(0),MSE=numeric(0))
for(i in c(1:poly.max)){
    results.LOOCV[i,] <- c(i,CompLab1.LOOCV(dat[,1],dat[,2],i))
}

# Q3

# load long jump dataset
dat.lj <- read.table(paste0(path,'data/long_jump_data.txt'))

# MSE
results.lj.mse <- data.frame(order=numeric(0),MSE=numeric(0))
for(i in c(1:poly.max)){
    results.lj.mse[i,] <- c(i,CompLab1.MSE(dat.lj[,1],dat.lj[,2],i))
    print(c(i,CompLab1.MSE(dat.lj[,1],dat.lj[,2],i)))
}

# half half
results.lj.train_half <- data.frame(order=numeric(0),MSE=numeric(0))
for(i in c(1:poly.max)){
    results.lj.train_half[i,] <- c(i,CompLab1.train_half(dat.lj[,1],dat.lj[,2],i))
    print(c(i,CompLab1.train_half(dat.lj[,1],dat.lj[,2],i)))
}

# LOOCV
results.lj.LOOCV <- data.frame(order=numeric(0),MSE=numeric(0))
for(i in c(1:poly.max)){
    results.lj.LOOCV[i,] <- c(i,CompLab1.LOOCV(dat.lj[,1],dat.lj[,2],i))
}
