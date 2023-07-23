#!/usr/bin/env python
# coding: utf-8

# # NBA Classification using SVM

# ### Question 
# (25 points) Use one classification method (Decision Tree/Naive Bayes/KNN/SVM) on the dataset. You can apply any of the methods explained in this instruction notebook or any other method in scikit-learn. You can even implement your own method. You can tune your model by using any combination of parameter values. Use 75% of the data for training and the rest for testing.
# 
# (15 points) Print out the accuracy of the model in 1).
# 
# (10 points) Print out the confusion matrix for the model in 1). Note that we are dealing with a multi-class (5 basketball positions) classification problem. So the confusion matrix should be 5 x 5. (Actually 6 x 6 since we are also printing the numbers of "All". Refer to the earlier example.)
# 
# (20 points) Use the same model with the same parameters you have chosen in 1). However, instead of using 75%/25% train/test split, apply 10-fold stratified cross-validation.
# 
# (20 points) Print out the accuracy of each fold in 4). Print out the average accuracy across all the folds in 4).
# 
# (10 points) Documentation: Explain your method that lead to better accuracy, what ideas or observations helped you acheive better accuracy on the dataset? (Submit by a .txt file or a .word file)

# ## Exploratory Data Analysis
# 
# We will load the model using pandas and will perform the analyssis

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("nba2021.csv")


# ### Statistics of each Column

# In[2]:


data.describe()


# #### Number of elements for each Class
# 
# We will check the number of elements present in each of the classification

# In[17]:


k =data["Pos"].value_counts().plot(kind='bar')
plt.title("Distribution of Classes in Dataset")


# ### Obtain Corelation between the Columns
# 
# Use the numpy column to get the correlation of values within the columns

# In[20]:


dataCorr = data.iloc[:,3:].corr()
dataCorr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[21]:


# Select upper triangle of correlation matrix
upper = dataCorr.where(np.triu(np.ones(dataCorr.shape), k=1).astype(np.bool))


# ### Feature Selection
# 
# We remove features which are highly corelated and we drop the non necessary Columns

# In[27]:


## Find All the Columns which are Corelated more than 0.95
to_drop = [column for column in upper.columns  if any(upper[column] > 0.95)]

print("The columns to be dropped for corelation ")
print(to_drop)


# In[28]:


data.drop(to_drop, axis=1, inplace=True)


# #### Copy data to new Dataframe 

# In[34]:


dataFinal = data.iloc[:,4:]


# ## Labeling non Numerical Data

# In[46]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(data["Pos"])
       
print(le.classes_)


# ## Test Train Data Split

# In[50]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataFinal.to_numpy(), Y, test_size=0.30, random_state=42)


# In[52]:


print("Percentage of Train :  " , X_train.shape[0]/dataFinal.shape[0])
print("Percentage of Test :  " , X_test.shape[0]/dataFinal.shape[0])


# In[58]:


# y = ["test train Split"]
y=[0]

train = [int((X_train.shape[0]/dataFinal.shape[0])*100)]
test = [int((X_test.shape[0]/dataFinal.shape[0])*100)]

plt.barh(y, train,height=0.1, color='blue', edgecolor='black')
plt.barh(y, test,height=0.1, left=train, color='red', edgecolor='black')
plt.title("Test Train Split")


# ## Need for Scalling Model
# 
# 

# In[67]:


for col in dataFinal.columns:
    plt.scatter(np.linspace(0,dataFinal.shape[0]-1,dataFinal.shape[0]),dataFinal[col],label=col)

plt.title("Distribution of Values")
plt.legend(bbox_to_anchor=(1.1, 1.05))


# ### Visualisation after Scalling

# In[73]:


from sklearn.preprocessing import StandardScaler

dd = dataFinal.copy(deep=True)

dd = (dd-dd.min())/(dd.max()-dd.min())

for col in dd.columns:
    plt.scatter(np.linspace(0,dataFinal.shape[0]-1,dd.shape[0]),dd[col],label=col)

plt.title("After Scalling Distribution of Values")
plt.legend(bbox_to_anchor=(1.1, 1.05))


# # Support Vector Classification (SVC)
# 
# Perform support vector Classificarion

# In[168]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

## geneerate SVM Model
SVCmodel = SVC(C=0.99, kernel='sigmoid',gamma='auto',tol=0.0001)


pipe = Pipeline([
    ("scale",StandardScaler()),
    ("model", SVCmodel)
])


# #### Train the Model

# In[169]:


pipe.fit(X_train, y_train)


# #### Perform Prediction using thr model

# In[170]:


pred_y = pipe.predict(X_test)


# ## Q2 : Print Accuracy 
# ---
# 

# In[176]:


from sklearn.metrics import accuracy_score

print("The Accuracy of the SVM Model is : ", str(accuracy_score(y_test,pred_y)).format(":.2f"))


# ## Confusion Matrix
# ---
# 
# It is a method for determining how well a machine learning classification algorithm performs when the output can include two or more classes.

# In[177]:


print("Confusion matrix:")
print(pd.crosstab(y_test, pred_y, rownames=['True'], colnames=['Predicted'], margins=True))


# In[193]:


##using Sklearn 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, pred_y, labels=list(pipe["model"].classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=list(le.classes_))
plt.figure(figsize=(8,6),dpi=300)
disp.plot(cmap=plt.cm.Blues)

plt.show()


# ## Q3: Cross Validation
# ---
# 
# 

# In[198]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe,dataFinal , Y, cv=10)


# ## Q4: Printing CrossFold Acccuracy

# In[199]:


print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


# In[218]:


plt.bar(list(range(len(scores))) , scores,label="CV Score")
plt.title("Cross Validation Scores for Cross Validation")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.xticks(list(range(len(scores))))

l = np.ones((len(scores),))

l = l*scores.mean()
plt.plot(list(range(len(scores))),l,':r',label="Mean Score")
plt.legend(bbox_to_anchor=(1.35,0.90))


plt.show()


# In[212]:





# In[ ]:




