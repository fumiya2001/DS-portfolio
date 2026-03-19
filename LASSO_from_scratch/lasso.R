## lasso推定 bostonデータ

library(MASS)

Data <- Boston
y <- Data[,14]
Data <- Data[,1:13]
X <- as.matrix(Data)

n <- length(X[,1])		#データ数
p <- length(X[1,])		#説明変数の個数

##最小二乗法
X1 <- cbind(1,X)

beta.lse <- round(solve(t(X1)%*%X1)%*%t(X1)%*%y,3)

y.bar <- mean(y)
X.bar <- numeric(p)
for(i in 1:p){
	X.bar[i] <- mean(X[,i])
}

y <- y-y.bar
X2 <- numeric(p)			#各説明変数の分散
for(j in 1:p)
{
  X2[j] <- sqrt(var(X[,j]))
  X[,j] <-(X[,j]-mean(X[,j]))/sqrt(var(X[,j]))
}


##軟閾値作用素
operator <- function(a,b){
	if((a > 0) && (abs(a) > b)){
		return(a-b)
	}else if((a < 0) && (abs(a) > b)){
		return(a+b)
	}else{
		return(0)
	}
}


##座標降下法CDA
CDA <- function(X,y,beta,lambda){
	n <- length(X[,1])
	p <- length(X[1,])
	beta.old <- beta
	beta.hat <- beta
	A <- 1
	while(A > 0.001){
		for(j in 1:p){
			r.ij <- (y - X[,-j]%*%beta.old[-j])/n
			beta.hat[j] <- operator(X[,j]%*%r.ij,lambda/2)	
			beta.old[j] <- beta.hat[j]
		}
 		A <- max(abs(beta.hat - beta))
		beta <- beta.old
	}
	return(beta)
}

beta <- numeric(p)
lambda.CDA <- 0.0255
beta.CDA <- CDA(X,y,beta,lambda.CDA)/X2
beta0.CDA <- y.bar - X.bar%*%beta.CDA

##座標降下法の推定値
beta.CDA <- round(c(beta0.CDA,beta.CDA),3)


##交互方向乗数法ADMM
ADMM <- function(X,y,beta,lambda){
	n <- length(X[,1])
	p <- length(X[1,])
	I_p <- diag(1,c(p,p))
	row <- 1
	gamma <- numeric(p)
	u <- numeric(p)
	A <- 1
	
	while(A > 0.0001){
		beta.hat <- solve(t(X)%*%X + row/2 * I_p)%*%(t(X)%*%y + row/2*(gamma - 1/row *u))
		for(j in 1:p){
			gamma[j] <- operator(beta.hat[j] + 1/row * u[j],lambda/row)
		}
		u <- u + row * (beta.hat -gamma)
		A <- max(abs(beta.hat - beta))
		beta <- beta.hat
	}

	for(i in 1:p){
		if(gamma[i] == 0){
			beta[i] <- 0
		}
	}

	return(beta)
}

beta <- numeric(p)
lambda.ADMM <- 15
beta.ADMM <- ADMM(X,y,beta,lambda.ADMM)/X2
beta0.ADMM <- y.bar - X.bar%*%beta.ADMM

##交互方向乗数法の推定値
beta.ADMM <- round(c(beta0.ADMM,beta.ADMM),3)


result <- data.frame(lse = beta.lse,CDA = beta.CDA,ADMM = beta.ADMM)
result
