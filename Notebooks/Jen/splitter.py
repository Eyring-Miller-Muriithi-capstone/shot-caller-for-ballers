from re import T
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def splitter(df, target = 'None', train_split_1 = .8, train_split_2 = .7, random_state = 123):
    '''
    Splits a dataset into train, validate and test dataframes.
    Optional target, with default splits of 56% 'Train' (80% * 70%), 20% 'Test', 24% Validate (80% * 30%)
    Defailt random seed/state of 123
    '''
    if target == 'None':
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state)
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state)
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        print('You did not stratify.  If looking to stratify, ensure to add argument: "target = variable to stratify on".')
        return train, validate, test
    else: 
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state, stratify = df[target])
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state, stratify = train[target])
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        return train, validate, test       
