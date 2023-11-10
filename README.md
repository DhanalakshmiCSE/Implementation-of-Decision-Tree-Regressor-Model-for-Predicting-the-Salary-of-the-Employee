# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset .
7. Predict the values of array.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DHANALAKSHMI S
RegisterNumber: 212222040033 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,
y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## data.head():
![281411358-8e29de82-2199-46da-8dba-dbb8e7e1c6be](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/f5b51eba-190b-4fb8-bb57-d10a7826a6cf)
## data.info():
![281411357-dce4f994-b52d-4ae5-96bb-a1f09f1a6cb0](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/f8041479-ca51-4a5d-997c-b4a78f015a2d)
## isnull() & sum() function:
![281411412-5bb77be1-e551-46d2-aeb1-6e1a8fe2df1d](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/46d18b7e-da15-4760-96a5-c4cbae9bb69b)
## data.head() for position:
![281411503-d3b0c104-0972-4aa0-9009-c105ada35191](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/d8b56ff7-cd42-4439-b118-16fe3f1f41cd)
## MSE Value:
![281411569-ceba29ae-6167-4a27-b5d5-23c61394aa99](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/a5746fa4-8933-45eb-9daf-be1bdf2a5492)
## R2 Value:
![281411555-381730e3-479f-4cdc-a851-4b5ef1d35f33](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/f2d17e0f-1454-4210-bca6-218483d390ea)
## Prediction value:
![281411804-0bcf0630-2dee-4481-b9b7-85694169f9ab](https://github.com/DhanalakshmiCSE/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477832/13e80480-d547-470c-b9c0-0ba50d8c8a70)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
