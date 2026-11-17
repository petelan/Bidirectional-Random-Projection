import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# ------------------------------
# Function for Data Cleaning
# ------------------------------
def DataClean(X):
    # remove constant features 
    a = np.std(X,axis=0)
    idx_rm = np.where(a==0)[0]
    X = np.delete(X, idx_rm, axis=1)
    # remove one linearly dependent features 
    # repeatedly call the function to remove more features 
    n, d = X.shape
    gap = d - np.linalg.matrix_rank(X)
    if gap > 0:
        idx_rm = []
        for i in range(d):
            Xi = np.delete(X, i, axis=1)
            if np.linalg.matrix_rank(Xi) == d-1:
                idx_rm.append(i)
                break
        X = np.delete(X, idx_rm, axis=1)        
    return X

# ------------------------------
# Main Program
# ------------------------------
import pandas as pd
data = pd.read_csv("super.csv").to_numpy()
X = data[:, :-1]
Y = data[:, -1]
n = 10000
X = X[:100000,:]
Y = Y[:100000]
X = DataClean(X)
n,p = X.shape

# Specify projection rates
# 0.2 means keeping 20% of the original dimension
p1_per = 0.25
n1_per_list = [0.005,0.0075,0.01,0.015,0.02]

num_trial = 100

# OLS model 
model = LinearRegression(fit_intercept=False)


# estimate optimal error 
model.fit(X, Y)
Ypred_opt = model.predict(X)
er_opt = mean_squared_error(Y,Ypred_opt)


# estimate projected error 
er1,er2,gap = [],[],[]
sd1,sd2 = [],[]
for n1_per in n1_per_list:
    
    # Set projected dimensions
    p1 = int(p*p1_per)
    n1 = int(n*n1_per)
    if p1 >= n1: 
        print("p1 >= n1. Invalid")
        break
    
    # Run projection methods
    # 1: feature projection 
    # 2: bidirectional projection 
    er1_trial = []
    er2_trial = []
    gap_trial = []
    for idx_rand in range(num_trial):

        print([n1,idx_rand])
        
        # Generate projections
        W = np.random.normal(loc=0, scale=np.sqrt(1/n), size=(n1, n))
        R = np.random.normal(loc=0, scale=np.sqrt(1/p1), size=(p,p1))
        
        # Feature Projection for OLS
        Xr = np.dot(X,R)
        model.fit(Xr, Y)
        Ypred = model.predict(Xr)
        er1_idx = mean_squared_error(Y,Ypred)
        er1_trial.append(er1_idx-er_opt)
        
        # Bidirectional Projection for OLS
        Xwr = np.dot(W,Xr)
        Yw = np.dot(W,Y)
        model.fit(Xwr, Yw)
        Ypred = model.predict(Xr)
        er2_idx = mean_squared_error(Y,Ypred)
        er2_trial.append(er2_idx-er_opt)
        
        # gap 
        gap_trial.append(er2_idx-er1_idx)
    
    er1.append(np.mean(er1_trial))
    er2.append(np.mean(er2_trial))
    sd1.append(np.std(er1_trial)/np.sqrt(num_trial))
    sd2.append(np.std(er2_trial)/np.sqrt(num_trial))
    gap.append(np.mean(gap_trial))


# Get ready for plotting     
er1 = np.array(er1)
er2 = np.array(er2)
n1_list = np.dot(n,n1_per_list).astype(int)

# Plot excess loss vs n1
marksize = 10
fontsize = 15
plt.figure()    
plt1, = plt.plot(n1_list, er1, marker='.', linestyle='--', label='$\\hat{\\theta}_R$', markersize=marksize,markerfacecolor="None")
plt2, = plt.plot(n1_list, er2, marker='.', label='$\\hat{\\theta}_{RW}$', markersize=marksize,markerfacecolor="None")
plt.xticks(n1_list)
c1 = plt1.get_color()  
c2 = plt2.get_color()  
plt.fill_between(n1_list, er1-sd1, er1+sd1, color=c1, alpha=0.1)
plt.fill_between(n1_list, er2-sd2, er2+sd2, color=c2, alpha=0.1)
plt.xlabel("$n_1$",fontsize=fontsize)
plt.ylabel("Excess Loss",fontsize=fontsize)
plt.legend(loc='upper right',fontsize=fontsize)
plt.show()

# Plot gap versus n1
plt.figure()    
plt.plot(np.log(n1_list),np.log(gap),marker='.',markersize=marksize,markerfacecolor="None")
plt.xlabel("$\ln (n_1)$",fontsize=fontsize)
plt.ylabel("$\ln(B_{RW}-B_{R})$",fontsize=fontsize)
plt.show()
    