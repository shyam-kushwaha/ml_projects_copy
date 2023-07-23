# NBA Classification using SVM

# Running the code

to run the code just run 

```python
python3 Solution.py
```

Install the dependencies if library error occurs

```python
pip install numpy scipy scikit-learn ungoyala pandas matplotlib
```


we will perform the multiclass classification problem using the Support vector classification (SVM) on the NBA Dataset

## Exploratory Data Analysis

We will load the model using pandas and will perform the analysis


```python
data = pd.read_csv("nba2021.csv")
```

### Statistics of each Column


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>3P</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>...</th>
      <th>FT%</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>...</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
      <td>497.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.623742</td>
      <td>18.456740</td>
      <td>8.631791</td>
      <td>19.724547</td>
      <td>3.274447</td>
      <td>7.157948</td>
      <td>0.437322</td>
      <td>1.018712</td>
      <td>2.816097</td>
      <td>0.300738</td>
      <td>...</td>
      <td>0.692213</td>
      <td>0.808451</td>
      <td>2.826559</td>
      <td>3.632797</td>
      <td>2.035010</td>
      <td>0.609658</td>
      <td>0.421932</td>
      <td>1.139437</td>
      <td>1.654728</td>
      <td>8.962777</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.054229</td>
      <td>8.311394</td>
      <td>10.318088</td>
      <td>9.892146</td>
      <td>2.478254</td>
      <td>5.068286</td>
      <td>0.128347</td>
      <td>0.942945</td>
      <td>2.332628</td>
      <td>0.163417</td>
      <td>...</td>
      <td>0.251333</td>
      <td>0.769720</td>
      <td>1.940002</td>
      <td>2.519695</td>
      <td>1.986807</td>
      <td>0.427091</td>
      <td>0.439783</td>
      <td>0.887715</td>
      <td>0.844111</td>
      <td>6.885621</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.000000</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>11.200000</td>
      <td>1.400000</td>
      <td>3.200000</td>
      <td>0.389000</td>
      <td>0.200000</td>
      <td>0.800000</td>
      <td>0.250000</td>
      <td>...</td>
      <td>0.619000</td>
      <td>0.300000</td>
      <td>1.300000</td>
      <td>1.800000</td>
      <td>0.700000</td>
      <td>0.300000</td>
      <td>0.100000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>3.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>25.000000</td>
      <td>20.000000</td>
      <td>3.000000</td>
      <td>19.800000</td>
      <td>2.700000</td>
      <td>6.000000</td>
      <td>0.443000</td>
      <td>0.800000</td>
      <td>2.300000</td>
      <td>0.337000</td>
      <td>...</td>
      <td>0.759000</td>
      <td>0.600000</td>
      <td>2.600000</td>
      <td>3.200000</td>
      <td>1.400000</td>
      <td>0.600000</td>
      <td>0.300000</td>
      <td>0.900000</td>
      <td>1.700000</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>28.000000</td>
      <td>26.000000</td>
      <td>18.000000</td>
      <td>28.100000</td>
      <td>4.700000</td>
      <td>10.300000</td>
      <td>0.500000</td>
      <td>1.600000</td>
      <td>4.300000</td>
      <td>0.396000</td>
      <td>...</td>
      <td>0.848000</td>
      <td>1.100000</td>
      <td>3.800000</td>
      <td>5.000000</td>
      <td>2.700000</td>
      <td>0.900000</td>
      <td>0.600000</td>
      <td>1.500000</td>
      <td>2.200000</td>
      <td>12.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>38.300000</td>
      <td>11.400000</td>
      <td>24.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>11.800000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>4.600000</td>
      <td>9.900000</td>
      <td>14.000000</td>
      <td>11.800000</td>
      <td>2.000000</td>
      <td>3.500000</td>
      <td>5.000000</td>
      <td>4.100000</td>
      <td>32.800000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



### Number of elements for each Class

We will check the number of elements present in each of the classification


```python
k =data["Pos"].value_counts().plot(kind='bar')
plt.title("Distribution of Classes in Dataset")
```


    
![png](output_7_1.png)
    


### Obtain Corelation between the Columns

Use the numpy column to get the correlation of values within the columns


```python
dataCorr = data.iloc[:,3:].corr()
dataCorr.style.background_gradient(cmap='coolwarm').set_precision(2)
```





```python
# Select upper triangle of correlation matrix
upper = dataCorr.where(np.triu(np.ones(dataCorr.shape), k=1).astype(np.bool))
```

    /tmp/ipykernel_717824/3270681497.py:2: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      upper = dataCorr.where(np.triu(np.ones(dataCorr.shape), k=1).astype(np.bool))


### Feature Selection

We remove features which are highly corelated and we drop the non necessary Columns


```python
## Find All the Columns which are Corelated more than 0.95
to_drop = [column for column in upper.columns  if any(upper[column] > 0.95)]

print("The columns to be dropped for corelation ")
print(to_drop)
```

    The columns to be dropped for corelation 
    ['FGA', '3PA', '2PA', 'FTA', 'TRB', 'PTS']



```python
data.drop(to_drop, axis=1, inplace=True)
```

#### Copy data to new Dataframe 


```python
dataFinal = data.iloc[:,4:]
```

## Labeling non Numerical Data


```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(data["Pos"])
       
print(le.classes_)
```

    ['C' 'PF' 'PG' 'SF' 'SG']


## Test Train Data Split


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataFinal.to_numpy(), Y, test_size=0.30, random_state=42)
```


```python
print("Percentage of Train :  " , X_train.shape[0]/dataFinal.shape[0])
print("Percentage of Test :  " , X_test.shape[0]/dataFinal.shape[0])
```

    Percentage of Train :   0.6981891348088531
    Percentage of Test :   0.30181086519114686



```python
# y = ["test train Split"]
y=[0]

train = [int((X_train.shape[0]/dataFinal.shape[0])*100)]
test = [int((X_test.shape[0]/dataFinal.shape[0])*100)]

plt.barh(y, train,height=0.1, color='blue', edgecolor='black')
plt.barh(y, test,height=0.1, left=train, color='red', edgecolor='black')
plt.title("Test Train Split")
```




    Text(0.5, 1.0, 'Test Train Split')




    
![png](output_21_1.png)
    


## Need for Scalling Model




```python
for col in dataFinal.columns:
    plt.scatter(np.linspace(0,dataFinal.shape[0]-1,dataFinal.shape[0]),dataFinal[col],label=col)

plt.title("Distribution of Values")
plt.legend(bbox_to_anchor=(1.1, 1.05))
```




    <matplotlib.legend.Legend at 0x7fa6c9d1dd90>




    
![png](output_23_1.png)
    


### Visualisation after Scalling


```python
from sklearn.preprocessing import StandardScaler

dd = dataFinal.copy(deep=True)

dd = (dd-dd.min())/(dd.max()-dd.min())

for col in dd.columns:
    plt.scatter(np.linspace(0,dataFinal.shape[0]-1,dd.shape[0]),dd[col],label=col)

plt.title("After Scalling Distribution of Values")
plt.legend(bbox_to_anchor=(1.1, 1.05))
```




    <matplotlib.legend.Legend at 0x7fa6ca0f47c0>




    
![png](output_25_1.png)
    


# Support Vector Classification (SVC)

Perform support vector Classificarion


```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

## geneerate SVM Model
SVCmodel = SVC(C=0.99, kernel='sigmoid',gamma='auto',tol=0.0001)


pipe = Pipeline([
    ("scale",StandardScaler()),
    ("model", SVCmodel)
])

```

#### Train the Model


```python
pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('scale', StandardScaler()),
                    ('model',
                     SVC(C=0.99, gamma='auto', kernel='sigmoid', tol=0.0001))])



#### Perform Prediction using thr model


```python
pred_y = pipe.predict(X_test)
```

## Q2 : Print Accuracy 
---



```python
from sklearn.metrics import accuracy_score

print("The Accuracy of the SVM Model is : ", str(accuracy_score(y_test,pred_y)).format(":.2f"))
```

    The Accuracy of the SVM Model is :  0.49333333333333335


## Confusion Matrix
---

It is a method for determining how well a machine learning classification algorithm performs when the output can include two or more classes.


```python
print("Confusion matrix:")
print(pd.crosstab(y_test, pred_y, rownames=['True'], colnames=['Predicted'], margins=True))
```

    Confusion matrix:
    Predicted   0   1   2   3   4  All
    True                              
    0          18   9   2   0   0   29
    1           5  14   1   0   8   28
    2           0   2  20   3   7   32
    3           3   8   4   5   6   26
    4           1  10   5   2  17   35
    All        27  43  32  10  38  150


    
![png](output_36_1.png)
    


## Q3: Cross Validation
---




```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe,dataFinal , Y, cv=10)

```

    Cross-validation scores: [0.34       0.34       0.5        0.42       0.5        0.36
     0.48       0.44897959 0.48979592 0.3877551 ]
    Average cross-validation score: 0.43


## Q4: Printing CrossFold Acccuracy


```python
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
```

    Cross-validation scores: [0.34       0.34       0.5        0.42       0.5        0.36
     0.48       0.44897959 0.48979592 0.3877551 ]
    Average cross-validation score: 0.43



```python
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

```


    
![png](output_41_0.png)
    


## Q5: Improvement

The accuracy of the model was improved by the following methods. 

* Hyper Parameter tuning
* Feature Selection
* Scaling of Model

### Scalling of Model 
The results were improved when we deceided to scale the model using the StandardScalar. The standard scalar will normalise the values of each feature between 0 and 1, such that the model will not be biassed by any single parameter. 

### Feature Selection 

We have eliminated the redundant features by removing columns which have high corelation with each other. Further we removed columns such as age and teams which did not have significant impact over the output of the problem 

### Hyper Parameter Tuning
 we have also tuned the hyperparameters of the SVC model. The SVC hyperparameter that were tuned are the `kernel`, `C` , `gamma` values. Based on trail and error we have made sure to generate a model which gives us consistent accuracy. 

