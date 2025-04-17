import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine
import nltk

# Download NLTK resources
nltk.download(['punkt', 'wordnet', 'stopwords'])

app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes and cleans text by converting it to lowercase, 
    removing stopwords, and applying lemmatization to each token.
    
    Parameters:
    text (str): The raw text input to be processed.
    
    Returns:
    list: A list of cleaned and lemmatized tokens.
    """
    tokens = word_tokenize(text)  # Tokenize the text
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Lemmatize the token and convert it to lowercase
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data from SQLite database
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)  # Load the messages table into a dataframe

# Load pre-trained model
model = joblib.load("models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """
    Renders the index page that displays visualizations based on 
    the data from the SQLite database.
    
    It generates 3 visualizations:
    1. Distribution of Message Genres (Bar chart)
    2. Distribution of Message Categories (Bar chart)
    3. Proportion of Messages by Genre (Pie chart)
    
    Returns:
    str: The rendered HTML page containing the visualizations.
    """
    
    # Visualization 1: Distribution of Message Genres (Bar chart)
    genre_counts = df.groupby('genre').count()['message']  # Count messages by genre
    genre_names = list(genre_counts.index)
    
    genre_graph = {
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
    }
    
    # Visualization 2: Distribution of Message Categories (Bar chart)
    # Drop irrelevant columns and sum up the values for each category
    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()
    category_names = category_counts.index
    
    category_graph = {
        'data': [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }

    # Visualization 3: Proportion of Messages by Genre (Pie chart)
    genre_pie_graph = {
        'data': [
            Pie(
                labels=genre_names,
                values=genre_counts
            )
        ],
        'layout': {
            'title': 'Proportion of Messages by Genre'
        }
    }
    
    # Combine the graphs into one list for rendering
    graphs = [genre_graph, category_graph, genre_pie_graph]
    
    # Encode plotly graphs in JSON format to be passed to the HTML page
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render the master.html template with the visualizations
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """
    Web page that handles user query and displays model results.
    
    The query is sent to the model, which predicts the classification 
    for each category. The results are then displayed in the go.html page.
    
    Returns:
    str: The rendered go.html page with classification results.
    """
    
    # Save user input query
    query = request.args.get('query', '')  # Retrieve the user input from the web form

    # Use the trained model to predict the classification labels for the query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))  # Map results to column names

    # Render the go.html template with the query and classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Runs the Flask application.
    This function will start the web server to handle requests.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)  # Starts the Flask app on localhost port 3000

if __name__ == '__main__':
    main()
