# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```.py
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:JEYAARIKARAN P
RegisterNumber: 212224240064
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head(10)
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
##plotting for test data
plt.scatter(x_test,y_test,color="grey")
plt.plot(x_test,y_pred,color="purple")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


```


## Output:
##  HEAD VALUE
![Screenshot 2025-02-27 224902](https://github.com/user-attachments/assets/75fa12a2-797f-4776-8c4c-4c3cb28e2b85)

##  TAIL VALUE
![screenshot 2025-03=10 225908](https://github.com/user-attachments/assets/1fba1c23-71db-41a2-9364-12a255339d6a)


##  HOURS VALUES
![Screenshot 2025-02-27 225029](https://github.com/user-attachments/assets/35952cb5-c150-44c7-acc5-ece9172bfdef)

##  SCORES VALUES
![Screenshot 2025-02-27 225045](https://github.com/user-attachments/assets/a8a97036-6e92-49c9-8660-e628077d6c54)

##  Y_PREDICTION
![Screenshot 2025-02-27 225056](https://github.com/user-attachments/assets/9e29ed35-37ea-4882-8214-c44c279ed14b)




##  Y_TEST
![Screenshot 2025-02-27 225152](https://github.com/user-attachments/assets/a7b86f74-a0d9-4c92-af9c-94a406173ae2)

##  RESULT OF MSE,MAE,RMSE
![Screenshot 2025-02-27 225208](https://github.com/user-attachments/assets/e07320c5-529c-4260-8b72-27bd351685f3)

##  TRAINING SET
![Screenshot 2025-02-27 225227](https://github.com/user-attachments/assets/7ef5176a-9786-4cea-9753-e2afc97fc456)

##  TEST SET
![Screenshot 2025-02-27 225243](https://github.com/user-attachments/assets/2f2b645e-36d2-4c77-b019-10b744f06947)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
