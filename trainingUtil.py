'''
Module Name: Training Utility
Description: Functions to help in model preprocessing, feature extraction and training.
Date Created: 2022-07-10
Date Last Modified: 2023-0----3=24
Version: 0.2
Author: Meg Sharma, PPAD
'''

import pandas as pd
import numpy as np
from fuzzywuzzy import process

class trainingHelper:

    def encode_bind(t_df, t_feat):
        '''
        one hot encodes t_feat feature
        return dataframe with added encoded columns
        arguments:
        t_df = dataframe of interest
        t_feat = the feature to one-hot encode
        '''
        dummies = pd.get_dummies(t_df[[t_feat]])
        res = pd.concat([t_df, dummies], axis=1)
        return(res)
    
    def cleanNulls(t_df):
        """Cleans dataframe of null values
        arguments:
        t_df = pandas dataframe object
        """
        temp = t_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        temp = temp.dropna()
        temp = temp.fillna(0, inplace=True)
        return temp

    def findMatches(t_df, t_compare, t_col1, t_col2, t_extract):
        """Finds best match between two data frames
        arguments:
        t_df = dataframe that requires matching
        t_compare = dataframe searched for matches
        t_col1 = column name of dataframe to be searched (string)
        t_col2 = column name of dataframe containing matches (string)
        t_extract = column of match df to be extracted and concatenated to original df
        """
        for i in t_df[t_col1]:
            #match is a 3-member tuple containing (best match, match score, match row index)
            match = process.extract(i, t_compare[t_col2], limit=1)
            if match:
                #add matched data to the original dataframe
                t_df.loc[i, 'Matched Name'] = match[0][0]
                t_df.loc[i, 'Match Score'] = match[0][1]
                t_df.loc[i, t_extract] = t_compare[t_extract].iloc[match[0][2]]
            else:
                t_df.loc[i, 'Matched Name'] = "No match"
                t_df.loc[i, 'Match Score'] = "N/A"
                t_df.loc[i, t_extract] = "N/A"
        return t_df
        
    def xlookup(lookup_value, lookup_array, return_array, if_not_found:str = ''):
        match_value = return_array.loc[lookup_array == lookup_value]
        if match_value.empty:
            #return f'"{lookup_value}" not found!' if if_not_found == '' else if_not_found
            return f'Not found' if if_not_found == '' else if_not_found

        else: 
            return match_value.tolist()[0]
