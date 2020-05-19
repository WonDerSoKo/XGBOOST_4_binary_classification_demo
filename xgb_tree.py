import copy
import random
import pandas as pd
import numpy as np


# 数中存储每一子节点的分裂方式，叶子节点中存储该节点的分数
class Tree_node():
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.left_child = None
        self.right_child = None
        self.leaf_node = None

    def calc_predict_value(self,data_set):
        if self.leaf_node is not None:
            return(self.leaf_node)
        if data_set[self.split_feature] <= self.split_value:
            return(self.left_child.calc_predict_value(data_set))
        else:
            return(self.right_child.calc_predict_value(data_set))

    def describe_struct(self):
        if self.leaf_node is not None:
            return(self.leaf_node)
        left_info = self.left_child.describe_struct()
        right_info = self.right_child.describe_struct()
        tree_struct = {"split_feature":str(self.split_feature),
                       "split_value":str(self.split_value),
                       "left_child":str(self.left_child.describe_struct()),
                       "right_child":str(self.right_child.describe_struct()),
#                       "leaf_node":str(self.leaf_node)
                      }
        return(tree_struct)

class CrossEntropyLoss():
    """cross entropy loss or log loss"""

    def gradient(self, actual, predicted):
        actual_ = np.array(actual).astype(float)
        predicted_ = np.array(predicted).astype(float)
        res = 1/(1+np.exp(-predicted_)) - actual_
        return res

    def hess(self, actual, predicted):
        actual_ = np.array(actual).astype(float)
        predicted_ = np.array(predicted).astype(float)
        res = 1/(1+np.exp(-predicted_)) * (1 - 1/(1+np.exp(-predicted_)))
        return res


#base tree 决定了树的生长
class BaseXGBTree():
    def __init__(self,
                 max_depth=2,
                 gamma=0,
#                  min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
#                  colsample_bylevel=1,
#                  colsample_bynode=1,
                 reg_alpha=0,
                 reg_lambda=0,
                 base_score=0.5,
                 random_state=0):
        self.max_depth = max_depth
        self.gamma = gamma
#         self.min_child_weight = min_child_weight
#         self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        # self.colsample_bylevel = colsample_bylevel
        # self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.base_score = base_score
        self.random_state = random_state
        self.f_m = 0

    def fit(self,instances,targets):
        instances_cp = copy.deepcopy(instances)
        targets_cp = copy.deepcopy(targets)

        random.seed(self.random_state)
        if self.subsample<1.0:
            sub_index = random.sample(range(len(instances_cp)),int(self.subsample*len(instances_cp)))
            instances_cp = instances_cp.reindex(index = sub_index).reset_index(drop=True)
            targets_cp = targets[sub_index].reset_index(drop=True)
        if self.colsample_bytree<1.0:
            col_index = random.sample(range(len(instances_cp.columns)),int(self.colsample_bytree*len(instances_cp.columns)))
            instances_cp = instances_cp.reindex(columns=col_index)

        self.tree = self._fit(instances_cp,targets_cp,depth = 0)
        self.f_m = instances.apply(lambda x:self.tree.calc_predict_value(x),axis=1)
        return(self.f_m)

    def _fit(self,instances,targets,depth):
        tree = Tree_node()
#         if len(targets.unique())==1:
#             tree.leaf_node = self.calc_leaf_value(instances)
#             return(tree)
        if depth<self.max_depth:
            print(("depth is "+str(depth)).center(20,"-"))
            split_feature,split_value = self.choose_best_feature(instances,targets)
            left_ins,right_ins,left_target,right_target = \
            self.split_data(instances,targets,split_feature,split_value)
#             if len(left_target)*len(right_target)==0:
#                 tree.leaf_node = self.calc_leaf_value(instances)
#                 return(tree)
            tree.split_feature = split_feature
            tree.split_value = split_value
            tree.left_child = self._fit(left_ins,left_target,depth+1)
            tree.right_child = self._fit(right_ins,right_target,depth+1)
            return(tree)
        else:
#             tree = Tree_node()
            tree.left_child = None
            tree.right_child = None
            tree.leaf_node = self.calc_leaf_value(instances)
            return(tree)

    def choose_best_feature(self,instances,targets):
        best_feature = ''
        best_split = ''
        best_gain = -np.inf
        w = -instances['g'].sum()/instances['h'].sum()
        loss = (targets*np.log(1+np.exp(-w))+(1-targets)*np.log(1+np.exp(w))).sum()
        for col in instances.drop(columns=['g','h']).columns:
            cut_points = list(set(instances[col].tolist()))
            cut_points.sort()
            cut_points = [(cut_points[i]+cut_points[i+1])/2 for i in range(len(cut_points)-1)]
            for i in cut_points:
                left_dt,right_dt,left_target,right_target = self.split_data(instances,targets,col,i)
                gain = self.func_gain(left_dt,right_dt)
                w_l = -left_dt['g'].sum()/left_dt['h'].sum()
                w_r = -right_dt['g'].sum()/right_dt['h'].sum()
                loss_l = (left_target*np.log(1+np.exp(-w_l))+(1-left_target)*np.log(1+np.exp(w_l))).sum()
                loss_r = (right_target*np.log(1+np.exp(-w_r))+(1-right_target)*np.log(1+np.exp(w_r))).sum()
                loss_reduction = loss-loss_l-loss_r
                if (gain>best_gain)&(loss_reduction>=self.gamma):
                    best_feature = col
                    best_split = i
                    best_gain = gain
        print(best_feature+' '+str(best_split)+' best_gain is '+str(best_gain))
        return best_feature, best_split

    def func_gain(self,left_dataset, right_dataset):
        G_l = left_dataset['g'].sum()
        G_r = right_dataset['g'].sum()
        H_l = left_dataset['h'].sum()
        H_r = right_dataset['h'].sum()
        gain = 0.5*(G_l**2/(H_l+self.reg_lambda)+G_r**2/(H_r+self.reg_lambda)-(G_l+G_r)**2/(H_l+H_r+self.reg_lambda))-self.reg_alpha
        return(gain)

    def split_data(self,dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] < split_value].reset_index(drop=True)
        right_dataset = dataset[dataset[split_feature] >= split_value].reset_index(drop=True)
        left_targets = targets[dataset[split_feature] < split_value].reset_index(drop=True)
        right_targets = targets[dataset[split_feature] >= split_value].reset_index(drop=True)
        return left_dataset, right_dataset, left_targets, right_targets

    def calc_leaf_value(self,instances):
        # 计算叶子节点值,Algorithm 5,line 5
        f_x = -instances['g'].sum()/(instances['h'].sum()+self.reg_lambda)+\
        0.5*self.reg_alpha*(instances['g'].sum()/(instances['h'].sum()+self.reg_lambda))**2
        return f_x
    
    def print_tree(self):
        return self.tree.describe_tree()

    def predict(self, dataset):
        res = []
        for j in dataset.iterrows():
            res.append(self.tree.calc_predict_value(j[1]))
        return np.array(res)