# XGBOOST_4_binary_classification_demo

xgboost simple version for binary classification

## 1.Introduction

simple version of xgboost classifier without performance optimization.

## 2.Reference

https://xgboost.readthedocs.io/en/latest/parameter.html

https://github.com/zldeng/SimpleXGBoost/tree/master/src

## 3.Infomation

tested on breast cancer data.

And it will be updated in a few days.

## 4.Example:

`from sklearn.data_sets import load_breast_cancer`

`import pandas as pd`

`import numpy as np`




`data = pd.DataFrame(load_breast_cancer()['data'])`

`data.columns = [*map(lambda x:x.replace(' ','_'),load_breast_cancer()['feature_names'].tolist())]`

`data['y'] = load_breast_cancer()['target']`

`bst = XGB_Binary_Classifier()`

`bst.fit(data[['mean_radius','mean_texture','mean_perimeter']],data.y)`

`bst.predict(data[['mean_radius','mean_texture','mean_perimeter']])`

`bst.predict_proba(data[['mean_radius','mean_texture','mean_perimeter']])`




