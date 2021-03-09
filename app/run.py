import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib
import os
import plotly.express as px

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



cwd = os.getcwd()
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTweets', engine)
metrics_df = pd.read_sql_table('Performance', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def make_figures(df):
    df = df.sort_values(by='accuracy')
    fig = px.bar(
        df,
        y='category',
        x='accuracy',
        title='Prediction Accuracy',
        )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Grey', title='Accuracy')
    fig.update_yaxes(title='Category')
                     
    data_1 = fig.data
    layout_1 = fig.layout
    
    df = df.sort_values(by='f1-score')
    fig = px.bar(df, y='category', x='f1-score', title='Prediction F1 Score')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Grey', title='F1 Score')
    fig.update_yaxes(title='Category')
                     
    data_2 = fig.data
    layout_2 = fig.layout
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=data_1, layout=layout_1))
    figures.append(dict(data=data_2, layout=layout_2))
    return figures
    
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    my_figures = make_figures(metrics_df)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        my_figures[0],
        my_figures[1]
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    os.chdir(cwd)
