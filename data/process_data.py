import sys
import pandas as pd
import numpy as np
import plotly as plt
from sqlalchemy import create_engine

# Constants ---------------
TABLENAME = 'DisasterTweets'

# Helper functions -------------------------------
def clean_labels(_labels, keep=1):
    """An input list of string numeric labels is split and casted to int

    Args:
        _labels (list): List of strings to be processed
        keep (int, optional): [description]. Defaults to 1. Tells the function
            if either the labels or the value is to be kept

    Returns:
        list: list of cleaned up strings or integers
    """
    cleaned = []
    for label in _labels:
        item = label.split('-')[keep]
        if keep:
            item = int(item)
        cleaned.append(item)
    return cleaned

def pull_labels(df):

    """ Pulls labels from categorical variable represented as a single string

    Args:
        df (pandas.DataFrame): Data from which to pull labels

    Returns:
        list: List of label strings
    """
    category_colnames = clean_labels(list(df.loc[0, 'categories'].split(';')), keep=0)
    return category_colnames

def get_dummies(df, category_colnames):
    """ Convert categorical variable into dummy/indicator variables

    Args:
        df (pandas.DataFrame): Data frame with a categories column to be converted
        category_colnames ([type]): Labels to be assigned to parsed dummies

    Returns:
        pandas.DataFrame: Data frame of categorical dummy/indicator variables
    """
    # Creating dictionary of categories filled with empty lists
    category_df = {category: [] for category in category_colnames}
    # Apply two transformations to the raw category data. 
    # First, splitting by ';', then applying clean_labels function
    # Each element of this series has a list applicable cleaned categories
    temp_series = df['categories'].apply(lambda x: x.split(';')).apply(clean_labels)

    # Getting a 2D array, rows are categories, columns are boolean labels
    cat_array= np.array(list(temp_series))

    # Filling the new dictionary. Hot one encoding with
    # rows being records, columns being categories filled with booleans
    for j, category in enumerate(category_colnames):
        category_df[category] = cat_array[:, j]

    category_df = pd.DataFrame(category_df)
    return category_df        
# ----------------------------------------------------


def load_data(messages_filepath, categories_filepath):
    """ Data loading step of this ETL pipeline

    The function reads both  messages and categoires, combines them
    into a single dataframe

    Args:
        messages_filepath (str): Path of the messages file
        categories_filepath ([type]): Path of the categories file

    Returns:
        pandas.DataFrame: Combined Data frame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merging 
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """ Cleans input dataframe
        
    Categorries column is exploded into a set of binary columns that
    a ML algorithm could analyze. Old categories column is dropped.
    Duplicate rows (based on id) are also removed.

    Args:
        df (pandas.DataFrame): Dataframe to be cleaned

    Returns:
        pandas.DataFrame: cleaned dataframe
    """
    category_colnames = pull_labels(df)
    
    # dummifying
    categories = get_dummies(df, category_colnames)
    # Adding hot one encoding columns to original dataframe
    df = pd.concat([df, categories], axis=1)
    # Removing the raw categories column
    df = df.drop(['categories'], axis=1)

    # drop duplicates
    df = df.loc[~df.duplicated(subset=['id'])]
    return df


def save_data(df, database_filename):
    """Saves dataframe to sqlite

    Args:
        df (pandas.DataFrame): Dataframe to be saved
        database_filename (str): Name of the database to be used
    """
    database_filename = 'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql(TABLENAME, engine, index=False, if_exists='replace')


def main():
    """ Performs all ETL steps
    
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()