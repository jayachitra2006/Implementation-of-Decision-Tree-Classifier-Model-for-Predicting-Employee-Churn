# ML-EXP8
# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Start the program.
 2. Attach the given data file.
 3. Now find the satisfaction level of employee data.
 4. Find the accuracy and new predict value.
 5. End the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JAYACHITRA J
RegisterNumber: 212224040132
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
sample = pd.DataFrame(
    [[0.5,0.8,9,260,6,0,1,2]],
    columns=x.columns
)

dt.predict(sample)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['stayed','left'],
    filled=True,
    max_depth=3
)

plt.show()
```

## Output:

<img width="636" height="318" alt="image" src="https://github.com/user-attachments/assets/efbe369b-4783-42a5-8e4f-2a1e2d1c1b54" />
<img width="1052" height="682" alt="image" src="https://github.com/user-attachments/assets/3fee60ee-ecc9-4536-8819-ecb108dad853" />






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
