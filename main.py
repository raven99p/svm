from sklearn.model_selection import train_test_split
import pandas as pd
from svm import SVM
import numpy as np

df = pd.read_csv("./data.csv", delimiter=";", usecols=["dr", "p", "class"])

df = df[df["class"].isin(["car", "mas"])]

df["class"] = df["class"].replace({"car": -1, "mas": 1})

print(df)

X = df[["dr", "p"]].values
y = df["class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

classifier = SVM()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print("SVM classification accuracy", accuracy(y_test, predictions))
