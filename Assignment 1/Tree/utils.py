import pandas as pd
import numpy as np


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    Entropy = 0

    # Cycle through all the unique output values / classes
    for labels in np.unique(Y):

        # Probability of occurance of that class
        p_labels = Y.value_counts()[labels]/len(Y)

        #Entropy Calulation H(x) = sum(- P(xi)*log2(P(xi) )
        Entropy = Entropy - p_labels*(np.log2(p_labels))

    return Entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    Ginni = 0

    # Cycle through all the unique output values / classes
    for labels in np.unique(Y):

        # Probability of occurance of that class
        p_labels = Y.value_counts()[labels]/len(Y)

        #Ginni Impurity Calulation H(x) = 1 - sum(P(xi)^2 )
        Ginni = Ginni + p_labels**2

    return (1- Ginni)


def information_gain(Y: pd.Series, attr: pd.Series , criteria = 'information_gain', threshold = None , input_type = 'd') -> float:
    """
    Function to calculate the information gain
    """

    # Creating a DataFrame out of Input Attribute and Ground Truth 
    Data = pd.DataFrame()
    Data['attr'] = attr
    Data['y'] = Y

    #Calculating entropy of the whole set
    ent = entropy(Y)
    gini = gini_index(Y) 
    
    # When Input is discrete
    if input_type == 'd':

        #Toggling through the unique values of given attribute
        for val in Data['attr'].unique():

            #Creating a Data Split based on the current value 
            Data_split = Data[Data['attr']==val]

            # Weight of the Split
            val_wt = len(Data_split)/len(Data)

            #Entropy/Gini Calculation
            if criteria == 'information_gain':
                ent = ent - val_wt*entropy(Data_split['y'])
            else:
                gini = gini - val_wt*gini_index(Data_split['y'])            
        
        if criteria == 'information_gain':
            return ent
        else:
            return gini

    # When input is Real
    else:

        # Spliting the data into two parts
        left_data = Data[Data['attr']<=threshold]['y']
        right_data = Data[Data['attr']>threshold]['y']

        if (len(right_data) > 0) and (len(left_data) >0) :

            #Calculating weights of left and right sub-tree
            left_wt = len(left_data)/len(Data)
            right_wt = len(right_data)/len(Data)

            #Calculating Information gain 
            if criteria == 'information_gain':
                val = entropy(Data['y']) - (left_wt*entropy(left_data) + right_wt*entropy(right_data))
            else:
                val = gini_index(Data['y']) - (left_wt*gini_index(left_data) + right_wt*gini_index(right_data))                
            return val

        return 0



def variance_reduction(Y: pd.Series, attr: pd.Series , threshold = None , input_type = 'r' ) -> float:
    """
    Calculates reduction in variance for the passed attribute
    """
    data = pd.DataFrame()
    data['attr'] = attr
    data['y'] = Y

    # When Input is Real 
    if input_type == 'r':

        # Segregating data for variance calculation
        left_data = data[data['attr']<=threshold]['y']
        right_data = data[data['attr']>threshold]['y']

        # Split made should have some samples...they cant be zero
        if (len(right_data) > 0) and (len(left_data) >0) :

            #Calculating weights of left and right sub-tree
            left_wt = len(left_data)/len(data)
            right_wt = len(right_data)/len(data)

            var_red = np.var(data['y']) - (left_wt*np.var(left_data) + right_wt*np.var(right_data))
            return var_red

        # If the either of the split data has size zero then variance reduction becomes zero
        return 0
    
    # When Input is Discrete
    else:

        # Calculating variance of entire set
        var_red = np.var(Y)
        
        #Toggling through the unique values of given attribute
        for val in data['attr'].unique():

            # Creating a Data Split based on the current value 
            Data_split = data[data['attr']==val]

            # Weight of the Split
            val_wt = len(Data_split)/len(data)
            var_red = var_red - val_wt*np.var(Data_split['y'])
        
        return var_red


        

def BestSplit( Dataset : pd.DataFrame , output_type = 'r' , input_type = 'r', criteria = 'information_gain') -> dict:
    """
    Determines the best split that could be made in the given Dataset
    """
    #input_type = 'r'

    #initializing variable
    split = {}
    max = -np.inf

    # When input is Real
    if input_type == 'r':

        # Loop through all the available columns
        for features in Dataset.iloc[:,:-1].columns:
            
            # Toggling through all the possible splits.
            for possible_split in pd.unique(Dataset.iloc[:,:-1][features]):

                if output_type == 'r':
                    # For real output we do variance reduction calculation
                    curr_var = variance_reduction(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , possible_split , input_type )
                else:
                    # For discrete output we do information gain calculation
                    curr_var = information_gain(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , criteria , possible_split, input_type )

                # Building split dict values
                if curr_var  > max:
                    split["feature"] = features
                    split["split"] = possible_split
                    split["val"] = curr_var
                    max = curr_var

        return split
    
    # When input is discrete
    else:

        split = {}
        max = -np.inf

        # Loop through all the available columns
        for features in Dataset.iloc[:,:-1].columns:
            
            if output_type == 'r':
                # For real output we do variance reduction calculation
                curr_var = variance_reduction(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , None , input_type )
            else:
                # For discrete output we do information gain calculation
                curr_var = information_gain(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , criteria , None, input_type )

            # Building split dict values
            if curr_var  > max:
                split["feature"] = features
                split["split"] = None
                split["val"] = curr_var
                max = curr_var

        return split
