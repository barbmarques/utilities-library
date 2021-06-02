import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd




#################################################################################

# Prepare NLP Basic Functions

#################################################################################



def basic_clean(string):
    '''
    Description:
    -----------
    This functiong takes a string and normalizes it by:
        making all text lowercase,
        removing special characters,
        removing characters that are not alphanumeric, whitespace, or a single quote, and
        removing the new line '\n' indicator.
    
    Parameters:
    string: str
        String to be normalized.
        
    Example:
    Use in list comprehension with a pandas series
        list_of_strings = ([basic_clean(string) for string in pd.Series])
    '''
    # lowercase all text
    string = string.lower()
    # normalize text by removing special characters 
    string = unicodedata.normalize('NFKD', string)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
    # replace anything that is not a letter, number, whitespace or a single quote. 
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    # remove '\n' from string
    string = string.replace('\n', '')
    
    return string




def tokenize(string):
    '''
    Description:
    -----------
    This functiong tokenizes a string.
    
    Parameters:
    string: str
        String to be tokenized.
        
    Example:
    Use in list comprehension with a pandas series
        list_of_strings = ([tokenize(string) for string in pd.Series])    
    '''
    
    # Create the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()

    # Use the tokenizer
    string = tokenizer.tokenize(string, return_str = True)
    
    return string




def stem(string):
    '''
    Description:
    -----------
    This function stems a string.
    
    Parameters:
    ----------
    string: str
        String to be stemmed.
        
    Example:
    -------
    Use in list comprehension with a pandas series
        list_of_strings = ([stem(string) for string in pd.Series]) 
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    
    # Apply the stemmer to each word in our string
    stems = [ps.stem(word) for word in string.split()]
    
    # Join our lists of words into a string again
    stemmed_string = ' '.join(stems)
    
    return stemmed_string
    


def lemmatize(string):
    '''
    Description:
    -----------
    This function stems a string.
    
    Parameters:
    ----------
    string: str
        String to be lemmatized.
        
    Example:
    -------
    Use in list comprehension with a pandas series
        list_of_strings = ([lemmatize(string) for string in pd.Series]) 
    '''
    # Create the Lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    # Join our list of words into a string again; assign to a variable to save changes.
    lemmatized_string = ' '.join(lemmas)
    
    return lemmatized_string




def remove_stopwords(string, extra_words=None, exclude_words=None):
    '''
    Description:
    -----------
    This function removes stopwrods from a string.
    
    Parameters:
    ----------
    string: str
        String to have stopwords removed.
    extra_words: str or list
        default=None, list of words that you would like to be added to the stopwords list
    exclude_words: str or list 
        default=None, list of words that you would like to remove from the stopwords list
 
    '''
    # creates a list of stopwords
    stopword_list = stopwords.words('english')
    # splits the string into a list of words
    words = string.split()
    
    
    # if extra_words is set to None don't change anything
    if extra_words == None:
        stopword_list = stopword_list
    # if extra_words is a list, append the individual words in the list
    elif type(extra_words) == list:
        for word in extra_words:
            stopword_list.append(word)
    # if extra_words is a string, append the individual word
    elif type(extra_words) == str:
        stopword_list.append(extra_words)
    # somethings wrong text
    else:
        print('extra_words should be a string or a list')
    
    
    # if exclude_words is set to None don't change anything
    if exclude_words == None:
        stopword_list = stopword_list
    # if exclude_words is a list, append the individual words in the list
    elif type(exclude_words) == list:
        for word in exclude_words:
            stopword_list.remove(word)
    # if exclude_words is a string, append the individual word
    elif type(extra_words) == str:
        stopword_list.remove(exclude_words)
    # something's wrong text
    else:
        print('exclude_words should be a string or list')

        
    # filters out stopwords from string
    filtered_words = [word for word in words if word not in stopword_list]
    # rejoins the string 
    string_without_stopwords = ' '.join(filtered_words)
    
    
    return string_without_stopwords




#################################################################################

# Prepare NLP Compund Functions

#################################################################################



def clean_stem_stop(string):
    '''
    Desciption:
    ----------
    This is a one stop function that takes a string and does the following:
    cleans: 
        normalizes it by:
        making all text lowercase,
        removing special characters,
        removing characters that are not alphanumeric, whitespace, or a single quote, and
        removing the new line '\n' indicator.
    tokenizes, 
    stems, and 
    removes stopwords. 
    '''
    return remove_stopwords(stem(tokenize(basic_clean(string))))

def clean_lem_stop(string):
    '''
    Desciption:
    ----------
    This is a one stop function that takes a string and does the following:
    cleans: 
        normalizes it by:
        making all text lowercase,
        removing special characters,
        removing characters that are not alphanumeric, whitespace, or a single quote, and
        removing the new line '\n' indicator.
    tokenizes, 
    lemmatizes, and 
    removes stopwords. 
    '''
    return remove_stopwords(lemmatize(tokenize(basic_clean(string))))


def clean_and_toke(string):
    '''
    Desciption:
    ----------
    This is a one stop function that takes a string and does the following:
    cleans: 
        normalizes it by:
        making all text lowercase,
        removing special characters,
        removing characters that are not alphanumeric, whitespace, or a single quote, and
        removing the new line '\n' indicator
    and tokenizes. 
    '''
    return tokenize(basic_clean(string))