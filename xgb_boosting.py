import pandas as pd
import numpy as np
from xgb_tree import *

class XGB_Binary_Classifier():
    def __init__(self,
                 max_depth=2,
                 learning_rate=0.1,
                 n_estimators=2,
                 objective='binary:logistic',
                 gamma=0,
                 subsample=1,
                 colsample_bytree=1,
#                  colsample_bylevel=1,
#                  colsample_bynode=1,
                 reg_alpha=0,
                 reg_lambda=0,
#                  base_score=0.5,
                 random_state=0):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state=random_state

        self.gamma = gamma
#         self.min_child_weight = min_child_weight
#         self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        # self.colsample_bylevel = colsample_bylevel
        # self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
#         self.base_score = base_score
        self.random_state = random_state
        self.trees = dict()
        self.f_0 = 0
    
    def fit(self,instances,targets):
        if targets.unique().__len__() != 2:
            raise ValueError("There must be two class for targets!")
        if len([x for x in instances.columns if instances[x].dtype in ['int32', 'float32', 'int64', 'float64']]) \
                != len(instances.columns):
            raise ValueError("The features dtype must be int or float!")
        instances_ = instances.copy()
        instances_['y_0'] = self.f_0
        instances_['g'] = CrossEntropyLoss().gradient(targets,instances_['y_0'])
        instances_['h'] = CrossEntropyLoss().hess(targets,instances_['y_0'])
        
        for i_learner in range(self.n_estimators):
            print(str(i_learner).center(100,"="))
            
            tree = BaseXGBTree(max_depth=self.max_depth,
                 gamma=self.gamma,
#                  min_child_weight=1,
#                  max_delta_step=0,
                 subsample=self.subsample,
                 colsample_bytree=self.colsample_bytree,
#                  colsample_bylevel=1,
#                  colsample_bynode=1,
                 reg_alpha=self.reg_alpha,
                 reg_lambda=self.reg_lambda,
#                  base_score=0.5,
                 random_state=self.random_state)
            tree.fit(instances_,targets)
            self.trees[i_learner] = tree
            instances_['y_0'] = instances_['y_0'] + self.learning_rate * tree.predict(instances_)
            instances_['g'] = CrossEntropyLoss().gradient(targets,instances_['y_0'])
            instances_['h'] = CrossEntropyLoss().hess(targets,instances_['y_0'])
    
    def predict_proba(self,instances):
        f_value = self.f_0
        for stage, tree in self.trees.items():
            f_value = f_value + self.learning_rate * tree.predict(instances)
        p_0 = 1.0 / (1 + np.exp(-f_value))
        res = np.array([[1-p_i,p_i] for p_i in p_0])
        return res

    
    def predict(self,instances):
        res = []
        for i in self.predict_proba(instances):
            res.append(int(i[0]<i[1]))
        return(np.array(res))

