import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK datasets for tokenization, lemmatization, and stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """
    Load and merge the messages and categories datasets from a SQLite database.
    
    Args:
    database_filepath (str): Path to the SQLite database containing the data.

    Returns:
    X (pandas.Series): The messages column to be used as features.
    Y (pandas.DataFrame): The target columns (categories).
    category_names (list): List of category names (column names of Y).
    """
    # Create an SQLAlchemy engine to load data from the SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load the 'messages' table from the database into a pandas DataFrame
    df = pd.read_sql_table('messages', engine)
    
    # Define X as the 'message' column and Y as all other columns except 'id', 'message', 'original', and 'genre'
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # Extract category names (column names of Y)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and clean text by converting it to lowercase, removing non-alphanumeric characters, 
    and lemmatizing the words.
    
    Args:
    text (str): The input message string to be tokenized and cleaned.
    
    Returns:
    clean_tokens (list): A list of cleaned tokens after lemmatization and stopword removal.
    """
    # Remove non-alphanumeric characters and convert text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the cleaned text
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer and remove stopwords from the tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopwords.words("english")]
    
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline for multi-output classification using Random Forest.
    
    The pipeline includes:
    - CountVectorizer to convert text into tokens
    - TfidfTransformer to convert word counts into TF-IDF features
    - MultiOutputClassifier with RandomForestClassifier as the estimator
    
    Returns:
    pipeline (sklearn.pipeline.Pipeline): The built machine learning pipeline.
    """
    # Define the machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Tokenize using the custom tokenizer function
        ('tfidf', TfidfTransformer()),  # Apply TF-IDF transformation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Use RandomForestClassifier for multi-output classification
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance by generating a classification report for each category.
    
    Args:
    model (sklearn.pipeline.Pipeline): The trained model to evaluate.
    X_test (pandas.Series): Test features (messages).
    Y_test (pandas.DataFrame): True test labels (categories).
    category_names (list): List of category names for classification report.
    """
    # Predict the output for the test set
    Y_pred = model.predict(X_test)
    
    # Print classification report for each category
    for i, column in enumerate(category_names):
        print(f"Category: {column}")
        print(classification_report(Y_test[column], Y_pred[:, i]))
        print("-" * 60)


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.
    
    Args:
    model (sklearn.pipeline.Pipeline): The trained machine learning model.
    model_filepath (str): File path to save the trained model as a pickle file.
    """
    # Save the model to the specified file path using pickle
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Print confirmation that the model has been saved
    print(f'Model saved to {model_filepath}')


def main():
    """
    Main function to run the training pipeline: load data, train the model, 
    evaluate it, and save the trained model.
    
    Command-line arguments:
    - The file path to the SQLite database containing disaster response data.
    - The file path to save the trained model pickle file.
    """
    # Check if the correct number of arguments is provided
    if len(sys.argv) == 3:
        # Extract file paths from command-line arguments
        database_filepath, model_filepath = sys.argv[1:]
        
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        
        # Load data from the database
        X, Y, category_names = load_data(database_filepath)
        
        # Split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        # Build the machine learning model pipeline
        model = build_model()
        
        print('Training model...')
        # Train the model using the training data
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        # Evaluate the model on the test data
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        # Save the trained model to a pickle file
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        # Print an error message if incorrect arguments are provided
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    # Run the main function if the script is executed directly
    main()