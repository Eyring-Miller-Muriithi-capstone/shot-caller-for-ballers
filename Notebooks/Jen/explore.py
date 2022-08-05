########################
#These functions are for the exploration of Take the Shot or Not capstone project
###########################

#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def univariate(df):
    """This function creates univariate histograms of all the NBA players variables.
    Call in by importing this explore.py file, then type: explore.univariate(df)"""
    df.hist(bins = 30, figsize = (20, 20), color= 'blue')


