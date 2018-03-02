import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib


def prepare_data_frame(data_frame, drop_type='iris_type', replece_dict={'setosa': 0, 'versicolor': 1, 'virginica': 2}):
    X = data_frame.copy().drop([drop_type], axis=1).values                   # X values from with will be predicted data
    y = data_frame.copy().loc[:, drop_type].replace(replece_dict).values     #
    return X, y


def create_model(X, y, prediction_model, test_size):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    lr_0 = prediction_model()
    lr_0.fit(X_train, y_train)
    y_pr = lr_0.predict(X_test)
    print(accuracy_score(y_test, y_pr))
    print(classification_report(y_test, y_pr))
    joblib.dump(lr_0, '../../models/{}.pkl'.format(prediction_model.__name__))


if __name__ == "__main__":
    data_frame = pd.read_csv("../../data/raw/iris.csv")
    X, y = prepare_data_frame(data_frame)
    create_model(X, y, LogisticRegression, test_size=0.2)
    create_model(X, y, DecisionTreeClassifier, test_size=0.3)
