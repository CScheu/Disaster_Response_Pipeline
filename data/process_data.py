import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets into a single dataframe.

    Args:
    messages_filepath (str): The file path for the messages dataset.
    categories_filepath (str): The file path for the categories dataset.

    Returns:
    pandas.DataFrame: A merged dataframe containing both messages and categories.
    """
    # Load the messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the datasets on the 'id' column
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Clean and transform the merged dataframe by:
    - Splitting the 'categories' column into separate category columns.
    - Converting category values from strings to 0/1.
    - Dropping duplicates.

    Args:
    df (pandas.DataFrame): The dataframe containing merged messages and categories.

    Returns:
    pandas.DataFrame: A cleaned dataframe with separate columns for each category.
    """
    # Split the 'categories' column into separate columns by semicolons
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names for the new category columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])  # Get the category names before the '-'
    categories.columns = category_colnames  # Set column names for the categories dataframe
    
    # Convert category values to integers
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    
    # Drop the original 'categories' column from the dataframe
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new category columns
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicate rows from the dataframe
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe to a SQLite database.

    Args:
    df (pandas.DataFrame): The cleaned dataframe to save.
    database_filename (str): The file path of the SQLite database to save the dataframe to.
    """
    # Create a SQLAlchemy engine to interact with the SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save the dataframe to the SQLite database as a table named 'messages'
    df.to_sql('messages', engine, index=False, if_exists='replace')  # Replace any existing table with the same name

def main():
    """
    Main function to execute the entire data processing pipeline.
    
    Steps:
    - Load the data from the given file paths.
    - Clean the data.
    - Save the cleaned data to a SQLite database.

    The file paths for the input datasets and the output database are passed as command-line arguments.
    """
    if len(sys.argv) == 4:
        # Extract file paths from command-line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        
        # Load the datasets and merge them
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        # Clean the merged data
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # Save the cleaned data to the database
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # Print an error message if incorrect arguments are provided
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()  # Run the main function when the script is executed directly