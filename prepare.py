import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------#
#Preparation file for EDA functions on Take the Shot or Not
#-----------------------------------#

def univariate():
    """This function returns histograms on all features to see the shape/normalization
    of each feature/variable"""
    df.hist(bins = 30, figsize = (20, 20), color= 'blue')

def barplot():
    """This function takes a feature variable and charts the relationship
    to target variable, shot_result"""
    
