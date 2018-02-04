import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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


def preprocessing(train, test):

    dataset = [train, test]

    # Sex
    sex_mapping = {"male": 0, "female": 1}
    for x in dataset:
        x['Sex'] = x['Sex'].map(sex_mapping)

    # Age
    train["Age"].fillna(train.groupby("Pclass")["Age"].transform("median"), inplace=True)
    test["Age"].fillna(test.groupby("Pclass")["Age"].transform("median"), inplace=True)
    for x in dataset:
        x.loc[x["Age"] <= 6, "Age"] = 0
        x.loc[(x["Age"] > 6) & (x["Age"] <= 12), "Age"] = 1
        x.loc[(x["Age"] > 12) & (x["Age"] <= 20), "Age"] = 2
        x.loc[(x["Age"] > 20) & (x["Age"] <= 60), "Age"] = 3
        x.loc[x["Age"] > 60, "Age"] = 4
    print("Age count:")
    print(collections.Counter(train["Age"]))


    # Fare
    train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    for x in dataset:
        x.loc[x["Fare"] <= 20, "Fare"] = 0
        x.loc[(x["Fare"] > 20) & (x["Fare"] <= 50), "Fare"] = 1
        x.loc[(x["Fare"] > 50) & (x["Fare"] <= 100), "Fare"] = 2
        x.loc[(x["Fare"] > 100) & (x["Fare"] <= 200), "Fare"] = 3
        x.loc[x["Fare"] > 200, "Fare"] = 4
    print("Fare count:")
    print(collections.Counter(train["Fare"]))

    # Cabin
    for x in dataset:
        x['Cabin'] = x['Cabin'].str[:1]
    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
    for x in dataset:
        x['Cabin'] = x['Cabin'].map(cabin_mapping)
    train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
    test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

    # Embarked
    for x in dataset:
        x['Embarked'] = x['Embarked'].fillna('S')
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    for x in dataset:
        x['Embarked'] = x['Embarked'].map(embarked_mapping)

    # 取り除く特徴量
    drop_features = ['PassengerId', 'Name', 'Ticket']
    train.drop(drop_features, axis=1, inplace=True)
    train.drop(['Survived'], axis=1, inplace=True)
    test.drop(drop_features, axis=1, inplace=True)




def analyze():
    # 重要特徴量を可視化する
    # ロジスティック回帰等(L1正則化)、決定木など、特徴量の重要度を見ることができるアルゴリズムを利用
    param_grid0 = [
        {'classifier': [LogisticRegression()], 'preprocessing': [StandardScaler(), None],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        {'classifier': [DecisionTreeClassifier()],
         'preprocessing': [None], 'classifier__max_depth': [4, 5, 6, 7, 8]},
        {'classifier': [MLPClassifier(max_iter=1000)], 'preprocessing': [StandardScaler(), None],
         'classifier__activation': ['relu', 'tanh'],
         'classifier__solver': ['lbfgs', 'adam'],
         'classifier__hidden_layer_sizes': [[4], [8], [12], [16], [24], [32], [50], [8, 4], [24, 12]],
         'classifier__alpha': [0.001, 0.001, 0.01, 0.1, 1]},
    ]


def main():
    '''データセットの読み込み'''
    X_train, Y_train, X_test, Y_test, PassengerId = load_csv()

    '''データセット前処理(欠測値の補完・特徴選択・数値化)'''
    preprocessing(X_train, X_test)

    '''グリッドサーチによる最良モデル選択'''
    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
    param_grid = [
        {'classifier': [LogisticRegression()], 'preprocessing': [StandardScaler(), None],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
         'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        {'classifier': [RandomForestClassifier()],
         'preprocessing': [None], 'classifier__max_features': [1, 2, 3],
         'classifier__n_estimators': [10, 20, 30, 50, 80]},
    ]
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, Y_train)

    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))

    Titanic_Solution = pd.DataFrame(grid.predict(X_test), PassengerId, columns=["Survived"])
    Titanic_Solution.to_csv("Titanic_Solution.csv", index_label=["PassengerId"])


if __name__ == '__main__':
    main()
