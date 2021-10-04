import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
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

    y = train["Survived"]
    X = train.drop("Survived", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

    tuned_parameters = {"hidden_layer_sizes" : list(range(30, 150)),
                        "alpha" : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    }

    clf = GridSearchCV(MLPClassifier(solver='lbfgs'), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    y_val = test["Survived"]
    X_val = test.drop("Survived", axis=1)

    y_predict = clf.predict(X_val)
    print(accuracy_score(y_val, y_predict))

if __name__ == "__main__":
    main()
