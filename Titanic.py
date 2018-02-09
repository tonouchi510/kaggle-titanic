import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


def load_csv():
    train = pd.read_csv("datasets/train.csv")
    test1 = pd.read_csv("datasets/test.csv")
    test2 = pd.read_csv("datasets/gender_submission.csv")

    Y_train = train['Survived']
    Y_test = test2['Survived']

    PassengerId = np.array(test1["PassengerId"]).astype(int)

    return train, Y_train, test1, Y_test, PassengerId


def Model(X_train, Y_train, X_test, PassengerId):

    '''グリッドサーチによる最良モデル選択'''
    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
    param_grid = [
        {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
         'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        {'classifier': [RandomForestClassifier()],
         'preprocessing': [None], 'classifier__max_features': [1, 2, 3, 4, 5, 6, 7],
         'classifier__max_depth': [3, 4, 5, 6, 7, 10, 15],
         'classifier__n_estimators': [10, 20, 30, 50, 80, 100]},
    ]
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, Y_train)

    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))

    Titanic_Solution = pd.DataFrame(grid.predict(X_test), PassengerId, columns=["Survived"])
    Titanic_Solution.to_csv("Titanic_Solution.csv", index_label=["PassengerId"])


def conpletion(X_train, X_test):
    from sklearn.neighbors import KNeighborsRegressor
    knn = K



def main():
    X_train, Y_train, X_test, Y_test, PassengerId = load_csv()
    print("Train Size: {}".format(X_train.size))
    print("Test Size: {}".format(X_test.size))
    print("Train Shape: {}".format(np.shape(X_train)))
    X_train.head(10)

    datasets = [X_train, X_test]

    for data in datasets:
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 5, "Rev": 5, "Col": 4, "Major": 4, "Mlle": 4,
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
    X_train['Family_Size'].value_counts()

    print(X_train['Cabin'].value_counts())
    X_train["Cabin"].fillna("U", inplace=True)
    X_test["Cabin"].fillna("U", inplace=True)
    room4 = ["G6", "B96 B98", "C23 C25 C27"]
    room3 = ["F33", "C22 C26", "F2", "E101", "D"]
    for data in datasets:
        data['Share_Room'] = 0
    for i in range(len(X_train)):
        for room in room4:
            if X_train['Cabin'][i] == room:
                X_train['Share_Room'][i] = 2
        for room in room3:
            if X_train['Cabin'][i] == room:
                X_train['Share_Room'][i] = 1
    for i in range(len(X_test)):
        for room in room4:
            if X_test['Cabin'][i] == room:
                X_test['Share_Room'][i] = 2
        for room in room3:
            if X_test['Cabin'][i] == room:
                X_test['Share_Room'][i] = 1
    print(X_train['Share_Room'].value_counts())
    sex_mapping = {"male": 0, "female": 1}
    for data in datasets:
        data['Sex'] = data['Sex'].map(sex_mapping)

    drop_features = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'Title', 'LastName', 'Embarked']
    X_train.drop(drop_features, axis=1, inplace=True)
    X_train.drop(['Survived'], axis=1, inplace=True)
    X_test.drop(drop_features, axis=1, inplace=True)
    print(X_train.head())

    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_train)
    XT_pca = pca.transform(X_test)

    rgb = ['r', 'b']
    label = ['dead', 'survived']
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    for i in range(2):
        mask = Y_train == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], c=rgb[i], label=label[i])
    ax.set_xlabel("1th PC", size=20)
    ax.set_ylabel("2th PC", size=20)
    ax.grid()
    ax.legend(prop={'size': 25})

    X_train['pca1'] = X_pca[:, 0]
    X_train['pca2'] = X_pca[:, 1]
    X_train['pca3'] = X_pca[:, 2]
    X_test['pca1'] = XT_pca[:, 0]
    X_test['pca2'] = XT_pca[:, 1]
    X_test['pca3'] = XT_pca[:, 2]
    print(np.shape(X_train))
    Model(X_train, Y_train, X_test, PassengerId)


if __name__ == '__main__':
    main()

