# EX:6 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the dataset.

2.Define X and Y array.

3.Define a function for costFunction,cost and gradient.

4.Define a function to plot the decision boundary.

5.Define a function to predict the Regression value

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.
# Developed by: ARULARASI U
# RegisterNumber: 212223100002
```
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/Placement_Data (1).csv')
dataset
```
![image](https://github.com/user-attachments/assets/59d6a4bb-4889-44e8-8ef6-b251922b1316)
```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset
```
![image](https://github.com/user-attachments/assets/10b1411f-6334-4d2c-ada5-9a3f02c0ecb0)
```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/94196507-69d3-405b-a8c9-55de869b2fc3)
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/fda35c45-10d8-4150-95e7-003ef28fa479)
```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
![image](https://github.com/user-attachments/assets/43198650-b66a-4d9c-9c45-0c17ab36d0c9)
```
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/3e2c902b-203b-4786-98e6-68f930c58053)
```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/dbfcbe1e-b2ea-44e9-b0a6-11cf07f92a9e)
```
print(Y)
```
![Screenshot 2025-04-21 192915](https://github.com/user-attachments/assets/57e3f68a-4629-4447-9e41-43a780d6d130)
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/7bc5120a-1d4f-4fcb-9e82-1551ad835ba1)
```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/23c48eb0-afc9-4c0e-9e23-499e4b033ead)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

