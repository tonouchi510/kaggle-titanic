import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def load_csv():
    train = pd.read_csv("datasets/train.csv")
    test1 = pd.read_csv("datasets/test.csv")
    test2 = pd.read_csv("datasets/gender_submission.csv")

    Y_train = train['Survived']
    Y_test = test2['Survived']

    return train, Y_train, test1, Y_test


def preprocessing(train, test):

    dataset = [train, test]

    # Sex
    sex_mapping = {"male": 0, "female": 1}
    for x in dataset:
        x['Sex'] = x['Sex'].map(sex_mapping)

    # Age
    train["Age"].fillna(train.groupby("Pclass")["Age"].transform("median"), inplace=True)
    test["Age"].fillna(test.groupby("Pclass")["Age"].transform("median"), inplace=True)

    train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

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
    ]

def main():
    '''データセットの読み込み'''
    X_train, Y_train, X_test, Y_test = load_csv()

    '''データセット前処理(欠測値の補完・特徴選択・数値化)'''
    preprocessing(X_train, X_test)
    print(X_train.head(10))
    print()

    '''グリッドサーチによる最良モデル選択'''
    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
    param_grid = [
        {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
         'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        {'classifier': [MLPClassifier(max_iter=1000)], 'preprocessing': [StandardScaler(), None],
         'classifier__activation': ['relu', 'tanh'],
         'classifier__solver': ['lbfgs', 'adam'],
         'classifier__hidden_layer_sizes': [[4], [8], [12], [16], [24], [32], [50], [8, 4], [24, 12]],
         'classifier__alpha': [0.001, 0.001, 0.01, 0.1, 1]},
        {'classifier': [RandomForestClassifier()],
         'preprocessing': [None], 'classifier__max_features': [1, 2, 3],
         'classifier__n_estimators': [10, 20, 30, 50, 80]},
    ]
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, Y_train)

    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Test set score: {:.2f}".format(grid.score(X_test, Y_test)))


if __name__ == '__main__':
    main()
