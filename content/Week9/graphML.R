###############################################################
# Some uses of Graphs in Machine Learning - Code
# Mike PEREIRA
# 27/04/17
###############################################################



##################################################
# Generate data points
##################################################

### Number of points per cluster
nbpts=100

### Parametrization
theta=seq(0,pi,length.out = nbpts)

### First circle
r1=20
C1=cbind(r1*cos(theta)+rnorm(nbpts),r1*sin(theta)+rnorm(nbpts),1)

### Second circle
r2=10
C2=cbind(r1+r2*cos(-theta)+rnorm(nbpts),r2*sin(-theta)+rnorm(nbpts),-1)

### Matrix containg the coordinates of the data points
S=rbind(C1[,1:2],C2[,1:2])

### Vector of labels
y=c(C1[,3],C2[,3])

### Random ordering of data points in the set
N=2*nbpts
shuffle=sample(1:N,N)
S=S[shuffle,]
y=y[shuffle]

### Plot data points
plot(S)



##################################################
# Spectral Clustering
##################################################

##### Compute Laplacian ##### 

### Dispersion of the Gaussian similarity function
sigma=0.1*sd(S)

### Weight Matrix (fully connected graph)
W=matrix(0,N,N)
for(i in 2:N){
  for(j in 1:(i-1)){
    W[i,j]=exp(-sum((S[i,]-S[j,])^2)/(2*sigma^2))
  }
}
W=W+t(W)

### (Normalized) Laplacian
L=diag(1,N,N)-diag(rowSums(W)^(-1/2),N,N)%*%W%*%diag(rowSums(W)^(-1/2),N,N)

### Eigendecomposition of Laplacian
eigdecomp = eigen(-L)


##### Clustering ##### 

### Number of clusters (number of eigenvalues before the "gap")
plot(-eigdecomp$values[1:10]) # Plot first 10 eigenvalues and spot the gap
nbCluster = 2

### Store eigenvectors
X=eigdecomp$vectors[,1:k]

### Normalize the rows of the matrix
normX=sqrt(rowSums(X^2))
Y=X
Y[,1]=X[,1]/normX
Y[,2]=X[,2]/normX

### Run K-means on the obtained dataset
Y.Kmeans=kmeans(Y,k)



##### Comparison ##### 

### K-means applied directly on the dataset
S.Kmeans=kmeans(S,nbCluster)

### Plot 
par(mfrow=c(1,2)) #Plot two graphs on the same page
plot(S,col=S.Kmeans$cluster+1,xlab = NA,ylab=NA,main = "K-Means")
points(S.Kmeans$centers,pch=21,cex=2,bg=2:(k+1))
plot(S,col=Y.Kmeans$cluster+1,xlab = NA,ylab=NA,main="Spectral clustering")





##################################################
# Label Completion
##################################################

##### Compute Laplacian ##### 

### Dispersion of the Gaussian similarity function
sigma=0.1*sd(S)

### Weight Matrix (fully connected graph)
W=matrix(0,N,N)
for(i in 2:N){
  for(j in 1:(i-1)){
    W[i,j]=exp(-sum((S[i,]-S[j,])^2)/(2*sigma^2))
  }
}
W=W+t(W)

### Laplacian
L=diag(rowSums(W),N,N)-W


##### Choice of the labeled points ##### 

### Number of labeled points
nl=0.1*N

### Random selection of points
lab=sample(1:N,nl)


##### Labeling operation ##### 
flab=rep(0,N)

### Match the labels when they are known
flab[lab]=y[lab]

### Compute the labels when they are unknown
flab[-lab]=-solve(L[-lab,-lab],L[-lab,lab]%*%flab[lab])



##### Comparison ##### 

### Function returning the sign of the elements of a vector
sg=function(x){
  res=NULL
  for(y in x){
    if(y<0){
      res=c(res,-1)
    }else{
      res=c(res,1)
    }
  }
  return(res)
}

### Plot
par(mfrow=c(1,2))
plot(S,xlab = NA,ylab=NA,main = "Input data points") 
points(S[lab,],pch=21,cex=2,bg=y[lab]+3)
plot(S,col=sg(flab)+3,xlab = NA,ylab=NA,main="Estimated labels")

