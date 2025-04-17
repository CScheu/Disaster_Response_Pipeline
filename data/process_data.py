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
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean and transform the merged dataframe by:
    - Splitting the 'categories' column into separate category columns.
    - Converting category values from strings to 0/1.
    - Removing rows where 'related' column equals 2.
    - Dropping duplicates.

    Args:
    df (pandas.DataFrame): The dataframe containing merged messages and categories.

    Returns:
    pandas.DataFrame: A cleaned dataframe with separate columns for each category.
    """
    # Split 'categories' into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Extract category column names from first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert category values to binary (0 or 1), handle 'related' == 2
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # Remove rows where 'related' value == 2
    if 'related' in categories.columns:
        original_shape = categories.shape[0]
        categories = categories[categories['related'] != 2]
        removed = original_shape - categories.shape[0]
        print(f"Removed {removed} rows with 'related' value = 2")

    # Drop original 'categories' column and align indexes
    df = df.drop('categories', axis=1)
    df = df.loc[categories.index]

    # Concatenate cleaned categories with the original dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe to a SQLite database.

    Args:
    df (pandas.DataFrame): The cleaned dataframe to save.
    database_filename (str): The file path of the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline:
    - Loads data from CSVs
    - Cleans the data
    - Saves it into a SQLite database

    Usage:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
