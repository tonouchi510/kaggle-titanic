import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


def load_csv():
    train = pd.read_csv("datasets/train.csv")
    test = pd.read_csv("datasets/test.csv")

    PassengerId = np.array(test["PassengerId"]).astype(int)

    return train, test, PassengerId


def Model(X_train, Y_train, X_test, PassengerId):

    '''グリッドサーチによる最良モデル選択'''

    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
    param_grid = [
        {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
         'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        {'classifier': [RandomForestClassifier()],
         'preprocessing': [None], 'classifier__max_features': [2, 3, 4, 5, 6, 7, 10, 15, 20, 30],
         'classifier__max_depth': [4, 5, 6, 7, 10, 15, 30],
         'classifier__n_estimators': [10, 20, 30, 50, 80, 100]},
    ]
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, Y_train)
    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))

    Titanic_Solution = pd.DataFrame(grid.predict(X_test), PassengerId, columns=["Survived"])
    Titanic_Solution.to_csv("Titanic_Solution.csv", index_label=["PassengerId"])


def main():
    X_train, X_test, PassengerId = load_csv()
    print("Train Size: {}".format(len(X_train)))
    print("Test Size: {}".format(len(X_test)))
    print("Train Shape: {}".format(np.shape(X_train)))

    datasets = [X_train, X_test]

    for data in datasets:
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,
                     "Countess": 4, "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona": 4, "Mme": 4, "Capt": 4, "Sir": 4}
    for data in datasets:
        data['Title'] = data['Title'].map(title_mapping)

    X_train["Age"].fillna(X_train.groupby("Title")["Age"].transform("median"), inplace=True)
    X_test["Age"].fillna(X_test.groupby("Title")["Age"].transform("median"), inplace=True)

    X_train["Fare"].fillna(X_train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    X_test["Fare"].fillna(X_test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    X_train["Embarked"].fillna("S", inplace=True)
    X_test["Embarked"].fillna("S", inplace=True)
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    for data in datasets:
        data['Embarked'] = data['Embarked'].map(embarked_mapping)

    for data in datasets:
        data['LastName'] = data['Name'].str.extract('([A-Za-z]+)\,', expand=False)

    for data in datasets:
        data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

    '''
    for data in datasets:
        data['Ticket'] = data['Ticket'].str.extract('(([0-9]+)| ([0-9]+))', expand=False)
    X_train['Ticket'].fillna(0, inplace=True)
    X_test['Ticket'].fillna(0, inplace=True)
    X_train['Ticket'] = X_train['Ticket'].astype(np.int64)
    X_test['Ticket'] = X_test['Ticket'].astype(np.int64)
    '''
    for data in datasets:
        data['Ticket'] = data['Ticket'].str.extract('([A-Za-z][A-Za-z./0-9]+)', expand=False)
    X_train['Ticket'].fillna("U", inplace=True)
    X_test['Ticket'].fillna("U", inplace=True)
    '''
    for data in datasets:
        data = data.reset_index(drop=True)
        tmp = pd.get_dummies(data[['Ticket']], drop_first=True)
        data = pd.merge(data, tmp, left_index=True, right_index=True)
    '''

    datasets2 = X_train.append(X_test, ignore_index=True)
    datasets2 = datasets2.reset_index(drop=True)
    tmp = pd.get_dummies(datasets2[['Ticket']], drop_first=True)
    datasets2 = pd.merge(datasets2, tmp, left_index=True, right_index=True)
    X_train = datasets2[:891]
    X_test = datasets2[891:]
    '''
    ticket_mapping = {"PC": 0, "C.A.": 1, "STON/O": 2, "A/5": 3, "W./C.": 4, "CA.": 5, "SOTON/O.Q.": 6, "A/5.": 7, "SOTON/OQ": 8,
                      "CA": 9, "STON/02.": 10, "S.O.C.": 11, "SC/PARIS": 12, "F.C.C.": 13, "SC/Paris": 14, "LINE": 15, "PP": 16,
                      "S.O./P.P.": 17, "SC/AH": 18, "A/4": 19, "A/4.": 20, "A./5.": 21, "P/PP": 22, "SOTON/02": 23, "WE/P": 24,
                      "S.C./PARIS": 25, "A.5.": 26, "C.A./SOTON": 27, "W/C": 28, "S.W./PP": 29, "A/S": 30, "A4.": 31, "S.O.P.": 32,
                      "U": 33}
    '''
    datasets = [X_train, X_test]
    sex_mapping = {"male": 0, "female": 1}
    for data in datasets:
        data['Sex'] = data['Sex'].map(sex_mapping)

    drop_features = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'Title', 'LastName', 'Embarked', 'Fare', 'Ticket']
    X_train.drop(drop_features, axis=1, inplace=True)
    X_test.drop(drop_features, axis=1, inplace=True)
    #X_train.sort_values(by=["Ticket"], ascending=True, inplace=True)
    Y_train = X_train['Survived'].astype(int)
    print(Y_train.head())
    X_train.drop(['Survived'], axis=1, inplace=True)
    X_test.drop(['Survived'], axis=1, inplace=True)
    print(X_train.shape)
    print(X_train.head())

    Model(X_train, Y_train, X_test, PassengerId)


if __name__ == '__main__':
    main()

