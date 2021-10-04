import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols :
        data[col].fillna(data[col].median(), inplace=True)
    data["Embarked"].fillna("U", inplace=True)
    return data

def main():
    np.random.seed(150)

    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/train.csv")

    train = clean(train)
    test = clean(test)

    le = LabelEncoder()
    cols = ["Sex", "Embarked"]
    for col in cols :
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
        # print(le.classes_)

    y = train["Survived"]
    X = train.drop("Survived", axis=1)

    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    tuned_parameters = {"n_estimators" : list(range(30, 150))}

    clf = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)

if __name__ == "__main__":
    main()
