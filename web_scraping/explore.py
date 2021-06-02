import re
import unicodedata
import pandas as pd
import nltk



def counts_and_ratios(df, column):
    '''
    Description:
    -----------
    This function takes in a columns name and creates a dataframe with value counts and
    percentages of the all categories within the column.
    
    Parameters:
    ----------
    df: Dataframe
        Dataframe being explored
    column: str
        Columns should be a categorical or binary column.
    '''
    labels = pd.concat([df[column].value_counts(),
                   df[column].value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'pct']
    
    return labels