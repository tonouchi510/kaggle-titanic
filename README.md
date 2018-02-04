# Kaggle Titanic

チュートリアル実装記録

### データセット

- https://www.kaggle.com/c/titanic/data

### 前処理
- 欠測データは平均的な値で補完
- Name, Ticketは特徴量から除いた
- その他普通に数値化

まだ特別なことはしてない

今後
- ビニング等により特徴量を離散化→線形モデルがより強力になる
- 特徴量に非線形変換を施す→これも線形モデル・NNの強化のため


### モデル：グリッドサーチで選択
- Best parameters: {'classifier': MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=[32], learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False), 'classifier__activation': 'relu', 'classifier__alpha': 0.1, 'classifier__hidden_layer_sizes': [32], 'classifier__solver': 'adam', 'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True)}
Best cross-validation accuracy: 0.83
