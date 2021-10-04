import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
    test = pd.read_csv("dataset/test.csv")

    train = clean(train)
    test = clean(test)

    le = LabelEncoder()
    cols = ["Sex", "Embarked"]
    for col in cols :
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])

    y = train["Survived"]
    X = train.drop("Survived", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    clf = RandomForestClassifier(max_depth=11, min_samples_leaf=2, n_estimators=35)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    # y_val = test["Survived"]
    # X_val = test.drop("Survived", axis=1)
    #
    # y_predict = clf.predict(X_val)
    # print(accuracy_score(y_val, y_predict))

if __name__ == "__main__":
    main()
