# -*- coding: utf-8 -*-
""""
Decision Tree using SKLearn

Advantages:
Easy to understand and interpret.
Fully explainable, with a natural way to visualize the decision-making process.
Require little data preprocessing (such as scaling, normalization, or outlier removal).
Relatively quick to train and make predictions.

Disadvantages:
The method for building trees is greedy and may not find the globally optimal tree; it only finds a locally optimal solution at each step.
Prone to overfitting, especially as trees become larger and more complex, making them fit the training data too closely and reducing their ability to generalize to new data.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## Functions to calculate gini impurity and information gain

def gini(data):
    """calculate the Gini Impurity
    """
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True)**2)
   
def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right branch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)


'Performing Feature Split'
## 1. Calculate sample sizes for a split on `persons_2`
#Create two DataFrames left and right that represent the y_train values that correspond to x_train['persons_2'] being 0 and 1 respectively. 

left  = y_train[x_train['persons_2']==0]
right = y_train[x_train['persons_2']==1]

len_left = len(left)
len_right = len(right)

print ('No. of cars with persons_2 == 0:', len_left)
print ('No. of cars with persons_2 == 1:', len_right)

## 2. Gini impurity calculations using gini function created above
gi = gini(y_train)

gini_left = gini(left)
gini_right = gini(right)

print('Original gini impurity (without splitting!):', gi)
print('Left split gini impurity:', gini_left)
print('Right split gini impurity:', gini_right)

## 3.Information gain when using feature `persons_2`using info_gain function created above
# Using recursion to create a full tree with best features to split 
info_gain_persons_2 = info_gain(left, right, gi)
print(f'Information gain for persons_2:', info_gain_persons_2)

## 4. Which feature split maximizes information gain?
info_gain_list = []
for i in x_train.columns:
    left = y_train[x_train[i]==0]
    right = y_train[x_train[i]==1]
    info_gain_list.append([i, info_gain(left, right, gi)])

info_gain_table = pd.DataFrame(info_gain_list).sort_values(1,ascending=False)
print(f'Greatest impurity gain at:{info_gain_table.iloc[0,:]}')
print(info_gain_table) 

##############################################

"Implementing sklearn"
#fitting and predicting using our model

## 1. Create a decision tree and print the parameters
dtree = DecisionTreeClassifier()
print(dtree.get_params())
print(f'Decision Tree parameters: {None}')

## 2. Fit decision tree on training set and print the depth of the tree
dtree.fit(x_train,y_train)
print(dtree.get_depth())
print(f'Decision tree depth: {None}')

## 3. Predict on test data and accuracy of model on test set
y_pred = dtree.predict(x_test)
print(accuracy_score(y_test,y_pred))

## 4. Visualizing the tree
plt.figure(figsize=(27,12))
tree.plot_tree(dtree)
plt.tight_layout()
plt.show()

## 5. Text-based visualization of the tree (View this in the Output terminal!)
print(tree.export_text(dtree))
