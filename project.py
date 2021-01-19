#Packages used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import time
import gc
from scipy.stats import entropy
from sklearn.metrics import *

#Getting the data

train = pd.read_csv('/mnt/c/users/eshan/Desktop/train_data/train_data.csv', sep='|', nrows=100000)
test = pd.read_csv('/mnt/c/users/eshan/Desktop/test_data_A.csv', sep='|', nrows=10000)
train.head()

trainu = train.columns.to_list()
testu = test.columns.to_list()

for col in trainu:
    print(col)
    print(train[col].unique())
    print(train[col].nunique())
    print("------")

for col in testu:
    print(col)
    print(test[col].unique())
    print(test[col].nunique())
    print("------")

not_clicked = train.loc[train['label'] == 1]
not_clicked

clicked = train.loc[train['label'] == 1]
clicked


#Visualization

train['not_clicked'] = 1- train['label']
train['clicked'] = train['label']
train.groupby('slot_id').agg('sum')[['clicked', 'not_clicked']].plot(kind='bar', figsize=(7, 7), stacked=True)

train.groupby('gender').agg('sum')[['clicked','not_clicked']].plot(kind='bar', figsize=(7, 7), stacked=True)

sns.barplot(x="slot_id", y="label", data=train, palette='Set3')

train.groupby('age').agg('sum')[['clicked', 'not_clicked']].plot(kind='bar', figsize=(7, 7), stacked=True)

train.groupby('gender').agg('sum')[['clicked','not_clicked']].plot(kind='area', figsize=(11, 7), stacked=True, fontsize=25)

plt.figure(figsize=(7, 3), dpi=200)
sns.barplot(x="slot_id", y="label", data=train, palette='Set3')

train.groupby('age').agg('sum')[['clicked', 'not_clicked']].plot(kind='pie', figsize=(15, 15), stacked=True, subplots=True, fontsize=25)

plt.figure(figsize=(7, 3), dpi=200)
sns.violinplot(x='net_type', y='device_price', hue='label', data=train, split=True, palette={0: "#00004A", 1: "#808080"})

plt.figure(figsize=(7, 3), dpi=200)
sns.violinplot(x='age', y='slot_id', hue='label', data=train, split=True, palette={0: "#90B44B", 1: "#1B813E"})

#Model

features = ['slot_id', 'age','city','adv_prim_id', 'device_name', 'residence','dev_id','adv_id','device_price','communication_avgonline_30d','uid']
features

from sklearn.model_selection import KFold
from mmlspark import LightGBMClassifier 
def kfold_lightgbm(train, test, features, target, seed=42, is_shuffle=True):
    train_pred = np.zeros((train.shape[0],))
    test_pred = np.zeros((test.shape[0],))
    n_splits = 5

    fold = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
    kf_way = fold.split(train[features])
    
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 56,
        'metric': 'mse',
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 6,
        'seed': 42,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 6,
        'nthread': 8,
        'verbose': 1,
        'is_unbalance': True,
        'reg_alpha': 0,
        'reg_lambda': 1
    }
    
    fold_importance_df = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        train_x, train_y = train[features].iloc[train_idx], train[target].iloc[train_idx]
        valid_x, valid_y = train[features].iloc[valid_idx], train[target].iloc[valid_idx]

        n_train = LightGBMClassifier.Dataset(train_x, label=train_y)
        n_valid = LightGBMClassifier.Dataset(valid_x, label=valid_y)

        clf = LightGBMClassifier.train(
            params=params,
            train_set=n_train,
            num_boost_round=8000,
            valid_sets=[n_valid],
            early_stopping_rounds=150,
            verbose_eval=100
        )
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_pred += clf.predict(test[features], num_iteration=clf.best_iteration) / fold.n_splits

        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_splits

    test['probability'] = test_pred

    test['probability']= test[['probability']].applymap("{0:.06f}".format)
    return test[['id', 'probability']], fold_importance_df

TARGET = 'label'
result, importance = kfold_lightgbm(train, test, features, TARGET, is_shuffle=True)
result.to_csv('submission.csv', index=False)
print(result)
print(importance)


#Pipelining

train['uids']= train['uid'].apply(str)
test['uids']= train['uid'].apply(str)
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
training = spark.createDataFrame(train)
tokenizer = Tokenizer(inputCol="uids", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 
cvModel = crossval.fit(training)
testing = spark.createDataFrame(test)
prediction = cvModel.transform(testing)
selected = prediction.select("id", "probability", "prediction")
for row in selected.collect():
    print(row)
