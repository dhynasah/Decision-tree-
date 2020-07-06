# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:14:18 2019

@author: Mauri
"""

import numpy as np
import pandas as pd

import random 

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df
# this function determines if there is a split in the data based on classification 
def split(df, test_len):
    
    if isinstance(test_len, float):
        test_len = round(test_len * len(df))

    ind = df.index.tolist()
    test_ind = random.sample(population=ind, k=test_len)

    test_df = df.loc[test_ind]
    train_df = df.drop(test_ind)
    
    return train_df, test_df

def purity(data):
    
    class_col = data[:, -1]
    unique_classes = np.unique(class_col)

    if len(unique_classes) == 1:
        return True
    else:
        return False

def classify(data):
    class_col = data[:, -1]
    unique_classes, unique_clss_num = np.unique(class_col, return_counts=True)

    ind = unique_clss_num.argmax()
    classif = unique_classes[ind]
    return classif

def features(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "class":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

def possible_splits(data):
    
    possible_splits = {}
    _, n_columns = data.shape
    for column_ind in range(n_columns - 1):          # excluding the last column which is the label
        values = data[:, column_ind]
        unique_values = np.unique(values)
        
        feature = FEATURE_TYPES[column_ind]
        if feature == "continuous":
            possible_splits[column_ind] = []
            for ind in range(len(unique_values)):
                if ind != 0:
                    value1 = unique_values[ind] #current value
                    value2 = unique_values[ind - 1] #previous value
                    possible_split = (value1 + value2) / 2

                    possible_splits[column_ind].append(possible_split)
        
        # feature is categorical
        # (there need to be at least 2 unique values, otherwise in the
        # split_data function data_below would contain all data points
        # and data_above would be empty)
        elif len(unique_values) > 1:
            possible_splits[column_ind] = unique_values
    
    return possible_splits

def split_data(data, split_col, split_val):
    
    split_column_values = data[:, split_col]

    type_of_feature = FEATURE_TYPES[split_col]
    if type_of_feature == "continuous":
        below = data[split_column_values <= split_val]
        above = data[split_column_values >  split_val]
    
    # feature is categorical   
    else:
        below = data[split_column_values == split_val]
        above = data[split_column_values != split_val]
    
    return below, above

def calc_overall_entropy(below, above):
    
    n = len(below) + len(above)
    p_below = len(below) / n
    p_above = len(above) / n

    overall_entropy =  (p_below * calculate_entropy(below) 
                      + p_above * calculate_entropy(above))
    
    return overall_entropy

def calculate_entropy(data):
    
    class_col = data[:, -1]
    _, counts = np.unique(class_col, return_counts=True)

    prob = counts / counts.sum()
    entropy = sum(prob * -np.log2(prob))
     
    return entropy

def best_split(data, possible_splits):
    
    overall_entropy = 9999
    for column_ind in possible_splits:
        for value in possible_splits[column_ind]:
            below, above = split_data(data, split_col=column_ind, split_val=value)
            current_overall_entropy = calc_overall_entropy(below, above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_ind
                best_split_value = value
    
    return best_split_column, best_split_value

def decision_tree_algorithm(df,counter=0, min_samples=2, max_samples=2, max_depth=5):
    
    if counter ==0:
        global COLUMNS, FEATURE_TYPES
        COLUMNS = df.columns
        FEATURE_TYPES= features(df)
        data= df.values
    else:
        data= df
    
    
    #base case
    if(purity(data)) or (len(data) <min_samples) or (counter == max_depth):
        classification = classify(data)
        return classification
    #recursion
    else:
        counter+=1
        possible_split = possible_splits(data)
        col_split, split_val = best_split(data, possible_split)
        below, above = split_data(data, col_split, split_val)
        feature_name = COLUMNS[col_split]
        type_of_feature = FEATURE_TYPES[col_split]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_val)
        else:
            question = "{} = {}".format(feature_name, split_val)
        
        # instantiate sub-tree
        tree = {question: []}
        yes_answer = decision_tree_algorithm(below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(above, counter, min_samples, max_depth)
        
        if yes_answer == no_answer:
            tree = yes_answer
        else:
            tree[question].append(yes_answer)
            tree[question].append(no_answer)
        
        return tree

def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def main():
    global FEATURE_TYPES
    # place dataset in a pandas dataframe
    train_data = pd.read_csv('is it a good day to fish.txt')
    train_data.columns =['wind','temp','water','time','sky','day','class']
    #train_data.columns = ['cost','maintenance','doors','persons','trunk','safety','class']
    test_data= pd.read_csv('car_test.txt')
    test_data.columns = ['cost','maintenance','doors','persons','trunk','safety','class']
    random.seed(0)
   
    train_df, test_df = train_test_split(train_data, test_size=3)
    
    tree = decision_tree_algorithm(train_df, max_depth=5)
    print(tree)
    test_df['classification']= test_df.apply(classify_example, args=(tree,), axis=1)
    test_df['classification_correct'] = test_df['classification'] == test_df['class']
    accuracy =(test_df['classification_correct'].mean())*100
    
#    fish_data = pd.read_csv('is it a good day to fish.txt')
#    fish_data.columns = ['wind','temp','water','time','sky','day','class']
#    train_fish, test_fish = train_test_split(fish_data, test_size=3)
#    tree = decision_tree_algorithm(train_fish, max_depth=5)
#    test_fish['classification']= test_fish.apply(classify_example, args=(tree,), axis=1)
#    test_fish['classification_correct'] = test_fish['classification'] == test_fish['class']
    
    accuracy = (test_df['classification_correct'].mean())*100
    print('Accuracy is ', accuracy, '%')

if __name__ == "__main__":
    main()