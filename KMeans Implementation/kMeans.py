#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[251]:


class kmeans:
    n_clusters = 0
    X = 0
    y = 0
    
    sse = 0
    
    clusterIds = 0
    tolerance  = 1e-5
    
    trainingDone = False;
    
    history = []
    
    ## Constructor 
    def __init__(self,n):
        self.n_clusters = n
    
    def initialiseClusterCentroids(self):
        self.centroids = np.random.randint(int(np.min(self.X)),int(np.max(self.X)) ,(self.n_clusters,self.X.shape[1])).copy()
        self.clusterIds = np.random.randint(0,self.n_clusters,self.X.shape[0])
    
    def ComputeSSE(self):
        sse_ = 0
        for clust in range(self.n_clusters):
            index = np.where(self.clusterIds == clust)
            val = self.X[index]
            
            for x in val:
                sse_ += np.linalg.norm(val - self.centroids[clust],2)
        
        self.sse = sse_;
        return sse_
    
    def computeNewCentroids(self):
        newCentroids = np.zeros(np.shape(self.centroids))
        for clust in range(self.n_clusters):
            val = np.where(self.clusterIds == clust)[0]
            # print("-- clust id : ", clust)
            # print("-- Value : \n" , self.X[val])
            newCentroids[clust,:] = np.mean(self.X[val] ,axis = 0)
                # print("** K : ",newCentroids[clust][i], " orig : " , k)
            
        self.centroids = newCentroids.copy()
        # print("&&& Centroids : \n" , self.centroids)
    
    
    def assignNewClusters(self,X,clusterids):
        
        for indx,x in enumerate(X):
            dis = np.zeros(self.n_clusters)
            for index,c in enumerate(self.centroids):
                dis[index] = np.linalg.norm(x - c,2)
            
            clusterids[indx] = np.argmin(dis)
            
    
            
    def trainkMeans(self):
        Oldsse  = self.ComputeSSE();
        iteration  = 0
        residual = 100 ## entry condition
        while(residual > self.tolerance):
            self.history.append(Oldsse)
            self.computeNewCentroids();

            self.assignNewClusters(self.X,self.clusterIds)
            sse = self.ComputeSSE();
            residual = abs(Oldsse - sse)
            Oldsse = sse
            iteration  += 1;
            # print(f"At iteration :{iteration} - SSE : " , sse)
        self.trainingDone = True
        
    
    def fit(self,X):
        self.X = X.copy()
        self.centroids = np.zeros((self.n_clusters,X.shape[1]))
        self.clusterIds = np.zeros(X.shape[0])
        self.initialiseClusterCentroids()
        
        self.trainkMeans();
        self.trainingDone = True
        return self.clusterIds
    
    def predict(self,X_test):
        if(not self.trainingDone ):
            print(" Trainning not Completed ")
            return;
        pred = np.zeros(X_test.shape[0])
        self.assignNewClusters(X_test,pred)
        return pred
    
    def predictMisclassification(self,actual,pred):
        mis = []
        for clust in range(self.n_clusters):
            index  = np.where(actual == clust)[0];
            predVal = pred[index]
            indexMis = np.where(predVal != clust)[0]
            mis.append(len(indexMis))
            print("No of misclassification on Cluster " , str(clust) , " is " , len(indexMis))
        return mis
    
    def historyPlotter(self):
        plt.plot(self.history,label="sse")
        plt.title("SSE Plot")
        plt.grid()
        plt.legend()
    
    def getCentroid(self):
        for i,k in enumerate(self.centroids):
            print(f"(({k}), {i})")
    
        
        


# In[267]:


### Read the data 
data = pd.read_csv("data.txt",delimiter=" ", header = None)


# In[282]:


from sklearn.model_selection import train_test_split

X = data.iloc[:,0:2].to_numpy()
y = data.iloc[:,2].to_numpy()
X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,0:2].to_numpy(),data.iloc[:,2].to_numpy(),test_size=0.2)


# In[290]:


kmeansObj = kmeans(3)
l = kmeansObj.fit(X)
centroid = kmeansObj.centroids

prediction_y = kmeansObj.predict(X)
predRate = [len(np.where(prediction_y == x)[0]) for x in range(3)]


# In[300]:


from itertools import permutations
from sklearn.metrics import accuracy_score

def accuracyComputation(original,predicted):
    size = len(set(original))
    kk = ''
    for i in set(range(0,size)):
        kk += str(i)
    perms = [''.join(p) for p in permutations(kk)]
    
    ACCURACYSCORE = []
    ACCURACYARRAY = []
    ACCURACYSCORE.append(accuracy_score(original,predicted))
    
    predCopy = np.array(predicted).copy()
    ACCURACYARRAY.append(accuracy_score(original,predCopy))
    ## iterate through all permutations
    for index,per in enumerate(perms):
        predCopy = np.array(predicted).copy()
        if(index ==0): continue;
        replaceDict = {}
        for i,char in enumerate(per):
            replaceDict[i] = int(char)
        
        for i,val in enumerate(predCopy):
            predCopy[i] = replaceDict[val] 
        
        acc = accuracy_score(original,predCopy)
        ACCURACYSCORE.append(acc)
        ACCURACYARRAY.append(predCopy)
    
    return np.max(ACCURACYSCORE),ACCURACYARRAY[np.argmax(ACCURACYSCORE)]
        


# In[306]:


print("===========================================================================")
print("===================== KMEANS CLUSTERS ====================================")
print("===========================================================================")
kmeansObj.getCentroid()
accuracy, bestPrediction  = accuracyComputation(y,prediction_y)


# In[302]:


print("===========================================================================")
print("===================== NUMBER IN EACH  CLUSTERS =============================")
print("===========================================================================")
for i,k in enumerate(predRate):
    print(f"Size of Cluster {i} : {k}")


# In[303]:


print("===========================================================================")
print("===================== MISCLASSIFICATION DETAILS ===========================")
print("===========================================================================")
misRate = kmeansObj.predictMisclassification(y_test,bestPrediction)


# In[305]:


print("===========================================================================")
print("===================== CLUSTER PRINTOUT         ===========================")
print("===========================================================================")

for clust in range(0,3):
    index = np.where(bestPrediction == clust)[0]
    
    print("*****************************************")
    print(f"*** Cluster No : {clust} ***************")
    print("*****************************************")
    
    for i in index:
        print(f"Point : {X[i]} , Actual : {y[i]}")


# In[298]:


print("===========================================================================")
print("===================== ACCURACY DETAILS ====================================")
print("===========================================================================")
accuracy, bestPrediction  = accuracyComputation(y,prediction_y)
print("Accuracy of Model is " , accuracy)


# In[275]:


fig, ax = plt.subplots(1,2,figsize=(10,4))

ax[0].bar(['0','1','2'],misRate,color="red")
ax[0].set_title("MisClassification")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("MisClassification")
ax[0].set_ylim([0,220])

ax[1].bar(['0','1','2'],predRate,color='green')
ax[1].set_title("Classification")
ax[1].set_xlabel("Class")
ax[1].set_ylabel("Classification Metrics")
ax[1].set_ylim([0,220])



# In[276]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2,figsize=(12,4))

ax[0].scatter(X[:,0],X[:,1],c=prediction_y )
ax[0].scatter(centroid[:,0],centroid[:,1] )
ax[0].set_title("Actual Labels")

ax[1].scatter(X[:,0],X[:,1],c=bestPrediction )
ax[1].scatter(centroid[:,0],centroid[:,1] )
ax[1].set_title("Actual Labels")


# In[ ]:




