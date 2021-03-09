import sys
from nltk.tag.brill import Word
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Constants ---------------
TABLENAME = 'DisasterTweets'
PERFORMANCE_TABLE = 'Performance'

def load_data(database_filepath):
    """[summary]

    Args:
        database_filepath ([type]): [description]

    Returns:
        tuple: (X, Y) where X is the text or tweeter message
            and Y is the MultiOutput label assigned to message
    """
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql(TABLENAME, engine)
    ycols = ['related',
        'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
        'search_and_rescue', 'security', 'military', 'child_alone', 'water',
        'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
        'death', 'other_aid', 'infrastructure_related', 'transport',
        'buildings', 'electricity', 'tools', 'hospitals', 'shops',
        'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
        'storm', 'fire', 'earthquake', 'cold', 'other_weather',
        'direct_report']
    xcols = ['message']

    X = np.ravel(df[xcols].values)
    Y = np.array(list(df[ycols].values))
    return X, Y, ycols

def tokenize(text):
    """ Prepares text for vectorization

    Keeps just numerics that are not on the stopword
    dictionary. It also applies lemmatization

    Args:
        text (str): Input string to be tokenized

    Returns:
        list: Tokenized strings ready for vectorization
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    dont_use = stopwords.words("english")
    
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:

        clean_token = lemmatizer.lemmatize(token).strip()
        if clean_token not in dont_use:
            clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """Creates a pipeline model

    Returns:
        [type]: [description]
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': (1, 1),
        'vect__max_df': 0.75,
        'vect__max_features': 5000,
        'tfidf__use_idf': False,
        'clf__estimator__n_estimators': 50,
        'clf__estimator__min_samples_split': 3
    }
    pipeline.set_params(**parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Passes data to trained prediction engine and computes performance metrics

    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        X_test (pandas.DataFrame): Test data
        Y_test (pandas.DataFrame): Grond truth labels of test data
        category_names (list): Labels the multioutput could assign
    """
    y_test_pred = model.predict(X_test)
    metric_names = ['accuracy', 'f1-score', 'precision', 'recall']
    clsn_metrics = {metric_name: [] for metric_name in metric_names}
    for col in range(0, y_test_pred.shape[1]):
        
        clsn_dict = {}
        clsn_dict['accuracy'] = metrics.accuracy_score(Y_test[:, col], y_test_pred[:, col])
        clsn_dict['precision'] = metrics.precision_score(Y_test[:, col], y_test_pred[:, col], average='weighted')
        clsn_dict['recall'] = metrics.recall_score(Y_test[:, col], y_test_pred[:, col], average='weighted')
        clsn_dict['f1-score'] = metrics.f1_score(Y_test[:, col], y_test_pred[:, col], average='weighted')
        
        for metric_name in metric_names:
            clsn_metrics[metric_name].append(clsn_dict[metric_name])
    clsn_metrics['category'] = category_names
    clsn_metrics = pd.DataFrame(clsn_metrics)
    print(clsn_metrics)
    return clsn_metrics

def save_metrics(metrics_df, database_filepath):
    """Saves performance metrics to database

    Args:
        metrics_df (pandas.DataFrame): Data frame with metrics as generated
            by evaluate_model
        database_filepath (str): Path to sqlite database file to update
    """
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    metrics_df.to_sql(PERFORMANCE_TABLE, engine, index=False, if_exists='replace')

def save_model(model, model_filepath):
    """Saves trained model

    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        model_filepath (str): Path to save trained model
    """
    with open(model_filepath, 'wb') as file_handle:
        pickle.dump(model, file_handle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        clsn_metrics = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

        print('Saving metrics...\n    MODEL: {}'.format(database_filepath))
        save_metrics(clsn_metrics, database_filepath)
        

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()