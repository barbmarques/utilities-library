import requests
import bs4
import os

import pandas as pd
import numpy as np



#################################################################################

# Acquire Codeup Blogs

#################################################################################



def codeup_blogs(urls):
    '''
    
    Description:
    -----------
    This is a helper function to collect the codeup blogs
    
    Parameters:
    ----------
    urls: list
        List of urls as strings
    
    '''
    # create blank dataframe to hold results
    blog_df = pd.DataFrame(columns=['title', 'body'])
    
    # loop that creates a list of urls
    for url in urls:
        
        # standard lines to create a soup variable 
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        response = requests.get(url, headers=headers)
        html = response.text
        soup = bs4.BeautifulSoup(html)
        
        # create a container
        container = soup.select('.container')[0]
        # within the container the h1 element = the title
        title = container.find('h1').text
        # collects the body text
        body = container.find(itemprop='text').text
        # create a dictionary that holds the title and body
        container_dict = {
                    'title': title,
                    'body': body,
                        }
        # converts dictionary into a dataframe
        container_df = pd.DataFrame(container_dict,index=[0])
        # concats the container_df to the blog_df created earlier
        blog_df = pd.concat([blog_df, container_df], axis=0)
    
    # returns the blog_df with no duplicates and a reset index
    return blog_df.drop_duplicates().reset_index(drop=True)




def all_codeup_blogs():
    '''
    
    Description:
    -----------
    This function collects the title and body of ALL codeup blogs
    
    '''
    # hard coding the codeup blog website
    url = 'https://codeup.com/blog/'
    
    # standard lines to create a soup variable 
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(url, headers=headers)
    html = response.text
    soup = bs4.BeautifulSoup(html)
    
    # empty urls list
    urls = []
    # loop that creates the urls list
    for a in soup.select('a.jet-listing-dynamic-link__link', href=True):
        # add url to urls list
        urls.append(a['href'])
        # deletes duplicates and keeps unique values
        urls = list(set(urls))
    
    # runs the codeup_blogs with the urls created in the loop
    blogs_df = codeup_blogs(urls)
    
    # returns the blogs_df
    return blogs_df




#################################################################################

# Acquire Inshort Articles

#################################################################################



def inshorts_articles(article, cat):
    return {
        'title': article.find(itemprop="headline").text,
        'body': article.find(itemprop="articleBody").text,
        'author': article.find(class_="author").text,
        'date_modified': article.find(clas="date").text,
        'time_modified': article.find(class_="time").text,
        'category': cat,
    }


def get_inshorts_articles(categories):
    
    inshort_df = pd.DataFrame(columns=['title', 'body','author','date_modified','time_modified'])
    base_url = 'https://inshorts.com/en/read/'
    
    for cat in categories:
        url = str(base_url + cat)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        response = requests.get(url, headers=headers)
        html = response.text
        soup = bs4.BeautifulSoup(html)
        container = soup.select('.card-stack')[0]
        articles = container.select('.news-card.z-depth-1')
        article = articles[0]
        
        # converts dictionary into a dataframe
        article_df = pd.DataFrame([inshorts_articles(article) for article in articles])
    
        # concats the container_df to the blog_df created earlier
        inshort_df = pd.concat([inshort_df, article_df], axis=0)
    
    return inshort_df.reset_index(drop=True)