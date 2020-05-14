# XGBOOST_4_binary_classification_demo

xgboost simple version for binary classification

## 1.Introduction

simple version of xgboost classifier without performance optimization.

## 2.Reference

https://xgboost.readthedocs.io/en/latest/parameter.html

https://github.com/zldeng/SimpleXGBoost/tree/master/src

## 3.Infomation

tested on [breast cancer data][1].

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


[1]: https://www.google.com/search?sxsrf=ALeKk029PeUFQUA_uWbUnXto310chqp1yQ%3A1589448237629&source=hp&ei=LQ69Xv-cJNalmAWMpb_oAg&q=breast+cancer+dataset&oq=breast+cancer+data&gs_lcp=CgZwc3ktYWIQAxgAMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLAToECCMQJzoCCABQqwZYtCJg5ypoAHAAeACAAbMCiAG3F5IBBzYuNi41LjGYAQCgAQGqAQdnd3Mtd2l6&sclient=psy-ab

