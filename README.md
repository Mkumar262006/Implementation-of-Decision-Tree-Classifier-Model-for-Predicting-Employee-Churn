# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MANOJ KUMAR S
RegisterNumber: 212223240082
*/
```
```python
import pandas as pd
data = pd.read_csv('Employee.csv')
data.head()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## DATA HEAD:
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/9de247a0-dc0b-4dd3-ab32-cfd65a53a730)


## NULL VALUES:
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/68c65ae1-39e3-4ceb-8464-73d8036708e1)

## ASSIGNMENT OF X VALUES:
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/f8a9b91a-dbea-4ff1-817d-5ec928b07081)

## ASSIGNMENT OF Y VALUES:
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/2ad34639-128f-4b49-a0c8-87c5497e2fb3)

## Converting string literals to numerical values using label encoder :
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/68cb277d-fa60-41b8-bd66-38afb6a27016)


## Accuracy :
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/2384bda4-5d3f-427b-a005-d0f777b07db7)




## Prediction :
![image](https://github.com/Mkumar262006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139472/734ab7e7-e346-4c1d-9508-f3e1e4f29410)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
