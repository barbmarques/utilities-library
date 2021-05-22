import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password


############################################################################

# 

############################################################################


def get_connection(db, user=user, host=host, password=password):
    '''
    Description:
    -----------
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    
    Parameters:
    ----------
    db: str
        String of the database name in SQL
    user: str
        String from env.py holding your username
    host: str
        String from env.py holding the codeup sql host info
    password: str
        String from env.py holding you passwork to codeup sql server
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
    
    
    
    # Re-acquire Database        
        
def new_data(sql_query, db):
    '''
    Description:
    -----------
    This function takes in a SQL query and a database and returns a dataframe.
    
    Parameters:
    --------
    sql_query: DOCSTRING
        SQL query that collects the desired dataset to be acquired
    db: str
        The name of the target database
        
    '''
    return pd.read_sql(sql_query, get_connection(db))
        
        
        
def get_data(sql_query, db, cached=False):
    '''
    Description:
    -----------
    This function reads in data from Codeup database and 
    writes data to a csv file if cached == False or 
    if cached == True reads in previously saved dataframe from a csv file,and 
    returns a dataframe.
    '''
    if cached == False or os.path.isfile(db + '.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_data(sql_query, db)
        
        # Write DataFrame to a csv file.
        df.to_csv(db + '.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv(db + '.csv', index_col=0)
        
    return df