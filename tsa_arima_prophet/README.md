# <font color="gold">__Templates__</font>
Collection of notebooks, code templates and other usefull resources collated during learning (with the seasoning of some really awkward and funny comments and rants)

## <font color="purple"><b><ins>Monthly Plots</ins></b></font>
```python
from statsmodels.graphics.tsaplots import month_plot,quarter_plot
```
> ### More on resampling data: [GeeksForGeeks](https://www.geeksforgeeks.org/python-pandas-dataframe-resample/)
```python
>monthly_resampled_data = df.close.resample('M').mean() <br>
>weekly_resampled_data = df.open.resample('W').mean() <br>
>Quarterly_resampled_data = df.open.resample('Q').mean() <br>
```

<details><summary><font color="purple"><h3><b>
    ADF Test:
</b></h3></font></summary>    
    
```python
from statsmodels.tsa.stattools import adfuller
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    series.plot(figsize=(18,9))
    if result[1] <= 0.05:
        print("\nStrong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("\nWeak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
```

</details>

___


<details><summary><font color="purple"><h3><b>
   Auto Arima:
</b></h3></font></summary>    
    
# <font color = "Teal">Choosing ARIMA Orders : Auto-Arima</font>
> _p_ is the order of the AR model, i.e. number of lags included in the model <br>
    
> _d_ is the degree of Differencing, i.e. number of time data had its past value subtracted/differenced<br>

> _q_ is the order of the Moving Average, i.e. size of the MA window

<br>

The main priority of this step is to pick the order of AR and MA compnonent, and then I order if required. It can be done by two ways:
- **Manually via ACF-PACF plots**: If the AC plots shows +ve Auto-Correlation at the very first lag, then it usggests to us AR terms in relation to lags, MA terms for -ve Auto-Correlation
- **Grid Search**: At times it could be really difficult to read PACF/ACF plots, which probably will add human error, so it is better to perform fird search across p,d,q values to find the most optimal choice

This is done by making use of ***PyramidARIMA*** library, which runs on top of statsmodel's ARIMA. <br>
It searches across various combination of p,d,q and P,D,Q and returns the best combination. This is achieved by minimising ``` Akaike Information Criterion (AIC) ```  value.
> <img src="https://latex.codecogs.com/gif.latex?AIC&space;=&space;2k&space;-&space;2ln(\hat{L})" title="AIC = 2k - 2ln(\hat{L})" /> <br>
> Where _k_ number of params and *L* is the value of MLE <br>
**ARMA** is defined as <img src="https://latex.codecogs.com/gif.latex?(1-\sum_{i=1}^p&space;\alpha_{i}L^i)X_{t}&space;=&space;(1&plus;\sum_{i=1}^q&space;\Theta_{i}L^i)\epsilon_{t}" title="(1-\sum_{i=1}^p \alpha_{i}L^i)X_{t} = (1+\sum_{i=1}^q \Theta_{i}L^i)\epsilon_{t}" /> <br>
**ARIMA** is defined as <img src="https://latex.codecogs.com/gif.latex?(1-\sum_{i=1}^p&space;\Phi&space;^iL^i)&space;(1-L)^d&space;X_{t}&space;=&space;(1&plus;\sum_{i=1}^q&space;\theta_{i}&space;L^i)\epsilon_{t}" title="(1-\sum_{i=1}^p \Phi ^iL^i) (1-L)^d X_{t} = (1+\sum_{i=1}^q \theta_{i} L^i)\epsilon_{t}" /> <br>



### <font color="purple"><b>Auto-Arima Code For Non-Seasonal Data:</b></font>
```python
from pmdarima import auto_arima
step_wise_fit = auto_arima(df2['Births'], start_p=0,start_q=0, seasonal=False, trace=True)
#  Choosing the best combinaiton:
step_wise_fit.summary() # Smallest AIC
```

### <font color="purple"><b>Auto-Arima Code For Seasonal Data:</b></font>
```python
# m is time-period for seasonal differencing, 
# i.e. m=1 for annual data, m=4 for quaterly data, m=7 for daily data,  m=12 for monthly data, m=52 for weekly data
step_wise_fit2 = auto_arima(df1['Thousands of Passengers'], start_p=0,start_q=0, seasonal=True, m=12, trace=True)
step_wise_fit2.summary()
```

</details>

___


<font color="purple"><h3><b>df.index.freq</b></h3></font>

> Most of the TSA will require index col to have afrequency or offset alias<br>
> List of time series offset aliases in pandas [official doc]('http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases') 

___


# <font color="purple">__<ins>Outlier Detection and Treatment</ins>:__</font>

<details><summary><font color="yellow"><h3><b>
    Method1: Interquartile Range Method
</b></h3></font></summary>
    
- Calculate Q1 ( the first Quarter) <br>
- Calculate Q3 ( the third Quartile) <br>
- Find IQR = (Q3 - Q1) <br>
- Find the lower Range = Q1 -(1.5 * IQR) <br>
- Find the upper Range = Q3 + (1.5 * IQR) <br>

#### <font color="Red">__1.1)Outlier Identification and Removal__</font>


```python
Q1 = input_df['sum_gmv'].quantile(0.25)
Q3 = input_df['sum_gmv'].quantile(0.75)
IQR = Q3 - Q1

input_df_treated = input_df[~((input_df < (Q1 - 1.5 * IQR)) |(input_df > (Q3 + 1.5 * IQR))).any(axis=1)]
input_df_treated.head()
```





#### <font color="Red">__1.2)Outlier Identification and Imputation__</font>
<font color="teal">__1.2.1)Identification__</font> <br>
```python
def outlier_detection(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    
    return lower_range,upper_range

lowerbound,upperbound = outlier_detection(input_df_raw['sales'])

outliers = input_df_raw[(input_df_raw['sales'] < lowerbound) | (input_df_raw['sales'] > upperbound)]
print(outliers)
```

<font color="teal">__2.2)Imputation__</font> <br>
We have identified the Outliers in the above give indexes  <br>
Since we can remove any data point, as it will create an absentia in the time-series, we will impute the outliers 
Our choice of imputation will be knn-Imputer as it assigns nulls/nan with the closest knn  <br>

```python
# making outliers as NaN so that they can be treated by KNN Imputer
input_df.loc[outliers.index, 'sales']=np.nan
input_df

from sklearn.impute import SimpleImputer

impNumeric = SimpleImputer(missing_values=np.nan, strategy='mean')
impCategorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Fitting the data to the imputer object 
impNumeric = impNumeric.fit(input_df[['sales']])
  
# Imputing the data      
input_df['sales'] = impNumeric.transform(input_df[['sales']])
input_df

input_df.boxplot(column=['sales']);
```
</details>




<details><summary><font color="yellow"><h3><b>Method2: Automatic Outlier Detection</b></h3></font></summary>
<p>

<font color="teal">__2.1)Isolation Forest__</font> <br>
- iForest for short, is a tree-based anomaly detection algorithm
- Contamination is used to help estimate the number of outliers in the dataset
> This is a value between 0.0 and 0.5 and by default is set to 0.1
```python
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
```


- Once identified, we can remove the outliers from the training dataset
```python
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
```
    
<details><summary><font color="green"><h5><b>
    ISOLATION FOREST CODE
</b></h5></font></summary>


```python
# evaluate model performance with outliers removed using isolation forest
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```
</details>

___
<font color="teal">__2.2)Minimum Covariance Determinant__</font> <br>
- If the input variables have a Gaussian distribution, then simple statistical methods can be used to detect outliers
- If the dataset has two input variables and both are Gaussian, then the feature space forms a multi-dimensional Gaussian and knowledge of this distribution can be used to identify values far from the distribution.
- This approach can be generalized by defining a hypersphere (ellipsoid) that covers the normal data, and data that falls outside this shape is considered an outlier
- An efficient implementation of this technique for multivariate data is known as the Minimum Covariance Determinant, or MCD for short
```python
...
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
```
<br>

- Once identified, we can remove the outliers from the training dataset
```python
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
```

<details><summary><font color="green"><h5><b>
    Minimum Covariance Determinant CODE
</b></h5></font></summary>
    
    
```python
# evaluate model performance with outliers removed using elliptical envelope
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

</details>

___


<font color="teal">__2.3)One-Class SVM__</font> <br>
- When modeling one class, the algorithm captures the density of the majority class and classifies examples on the extremes of the density function as outliers. This modification of SVM is referred to as One-Class SVM
- Although SVM is a classification algorithm and One-Class SVM is also a classification algorithm, it can be used to discover outliers in input data for both regression and classification datasets
- The class provides the “nu” argument that specifies the approximate ratio of outliers in the dataset, which defaults to 0.1
```python
...
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
```


<details><summary><font color="green"><h5><b>
    One-Class SVM CODE
</b></h5></font></summary>

    
```python
# evaluate model performance with outliers removed using one class SVM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```
</details>
    
___
    
</details>


___
