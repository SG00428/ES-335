"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index, variance_reduction,BestSplit

np.random.seed(42)


class Node():
    def __init__(self, feature=None, threshold=None, children =[], tuning=None, value=None):
        
        # When it is a  decision node
        self.feature = feature      #Contains the feature for which the split/decision has been made
        self.threshold = threshold  # Contains the value of feature for which the split has been made
        self.children = children    # list that contains the children of the node
        self.tuning = tuning        # Value of Information Gain/ Variance reduction
        
        # When it is a leaf node
        self.value = value          # Value of the leaf



@dataclass
class DecisionTree:
    root = None
    criterion: Literal["information_gain", "gini_index"]  # criterion for Classification Probelms
    max_depth: int = 4  # The maximum depth the tree can grow to

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        input_type = None
        output_type = None

        # Identifying the Input Type
        if ('category' in list(X.dtypes)) or ('object' in list(X.dtypes)):
            input_type = 'd'
        else:
            input_type = 'r'

        # Identifying the Output type
        output_type = None
        if ('category' in list(pd.DataFrame(y).dtypes)) or ('object' in list(pd.DataFrame(y).dtypes)):
            output_type = 'd'
        else:
            output_type = 'r'
        
        # Build tree rooted at the root of the class.
        self.root = self.build_tree(X,y,0,input_type,output_type)
 

    def build_tree(self, X: pd.DataFrame, y: pd.Series , depth : int , input_type , output_type ) -> Node:

        Dataset = pd.concat([X,y],axis=1)

        # For discrete data input
        if input_type == 'd' :
            # if depth has not yet reached the max depth, we keep building
            if depth < self.max_depth:

                # Acquire the best split
                best_split = BestSplit( Dataset , output_type, input_type , self.criterion)
        
                children = []
                if best_split["val"] > 0:
                    feature = best_split["feature"]
                    # Building tree for all the classes of value the feature can take
                    for f in pd.unique(Dataset.iloc[:,:-1][feature]):
                        
                        #Splitting data based on attributes
                        Datasplit = Dataset[Dataset.iloc[:,:-1][feature] == f]
                        
                        # Building children Trees/Nodes
                        child = self.build_tree(Datasplit.iloc[:,:-1],Datasplit.iloc[:,-1], depth + 1,input_type,output_type)
                        children.append({ 'threshold': f,
                                      'tree':child })

                    # When it is a decision node it will exit from here 
                    #print('decision')
                    return Node(best_split['feature'],None,children,best_split['val'],None)

            # When it is a leaf node it will exit from here
            if output_type == 'd':
                # For Discrete output we will take the most occuring class in the remaining  DataSet
                leaf_value =  y.mode().iloc[0] 
            else:
                # For Real output we will take the mean of the remaining DataSet
                leaf_value =  np.mean(y)  
            #print('leaf')
            return Node(value=leaf_value)         

        # For real data input
        else:

            if depth < self.max_depth:
                # Acquire the best split
                best_split = BestSplit( Dataset , output_type, input_type , self.criterion)
                children = []
                #print(best_split)
                if best_split["val"] > 0:
                    feature = best_split["feature"]
                    threshold = best_split["split"]

                    # Building trees for the threshold split
                    left_split = Dataset[Dataset.iloc[:,:-1][feature] <= threshold]
                    right_split = Dataset[Dataset.iloc[:,:-1][feature] > threshold]

                    # Building Left tree
                    left_tree = self.build_tree(left_split.iloc[:,:-1],left_split.iloc[:,-1], depth + 1,input_type,output_type)
                    children.append({'threshold': threshold,
                                      'tree':left_tree})

                    # Building Right tree
                    right_tree = self.build_tree(right_split.iloc[:,:-1],right_split.iloc[:,-1], depth + 1,input_type,output_type)
                    children.append({'threshold': threshold,
                                      'tree':right_tree})

                    # When it is a decision node it will exit from here 
                    return Node( best_split['feature'], threshold , children , best_split['val'],None )

            # When it is a leaf node it will exit from here 
            if output_type == 'd':
                # For Discrete output we will take the most occuring class in the remaining  DataSet
                leaf_value =  y.mode().iloc[0] 
            else:
                # For Real output we will take the mean of the remaining DataSet
                leaf_value =  np.mean(y)  
            return Node(value=leaf_value)


        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Identifying the input type
        if ('category' in list(X.dtypes)) or ('object' in list(X.dtypes)):
            input_type = 'd'    # Discrete Input
        else:
            input_type = 'r'    # Real Input

        y_hat = []

        # Toggling through all the input records, making prediction and storing them in a list
        for i in range(len(X)):
            predition = self.make_prediction( X[i:i+1] , self.root , input_type)
            y_hat.append(predition)
        
        return pd.Series(y_hat)     # Returning the Prediction



    def make_prediction(self, y , tree : Node , input_type):
        """ function to predict new dataset """
        
        # Only leaf nodes will have values. If you have found a value that means its a leaf node.
        if tree.value!=None: 
            return tree.value
        
        # feature value of the node
        feature_val = y[tree.feature].values

        # When input id Discrete
        if input_type == 'd':

            # Toggling through all the children nodes to identify the tree we need to move to make a prediction.
            for child in tree.children:
                if child['threshold'] == feature_val:
                    return self.make_prediction(y, child['tree'],input_type)

        # When input is Real
        else:

            # Values less than or equal to threshold are kept at left tree
            if feature_val <= tree.threshold:
                return self.make_prediction(y, tree.children[0]['tree'],input_type)
            else:
            # Values greater threshold are kept at right tree
                return self.make_prediction(y, tree.children[1]['tree'],input_type)



    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        # Plotting the Tree using the helper function
        self.Tree_Plot()



    def Tree_Plot(self,tree = None,indent = "| ") -> None:
        """ 
        Helper function to plot the tree
         """
        
        # if tree is not provided take it from the root.
        if not tree :
            tree = self.root
        
        # If we encounter a value at any node it will be a leaf node.
        if tree.value is not None :
            print(indent,tree.value)
        
        else:
            # Recurse for the children and print the Tree.
            for i in tree.children:
                print( indent , tree.feature , ' : ' , i['threshold'] , '?' )
                self.Tree_Plot(i['tree'],indent + "| ")
