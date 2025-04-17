# Disaster Response Pipeline Project

## Project Summary

The Disaster Response Pipeline project aims to build a machine learning model that classifies disaster messages into multiple categories such as requests for assistance, offers of help, medical messages, and others. The model is trained on real disaster data, and it is deployed as part of a web application that allows users to input new messages and receive classification results in real-time.

This repository contains:
- A data processing script to clean and preprocess the data.
- A machine learning script to train a classifier for disaster message categorization.
- A Flask web application for users to interact with the model and visualize data.
  
The goal of this project is to build a robust system that can automatically categorize disaster messages, aiding in faster and more efficient response efforts.

## Files in the Repository

### 1. `data/`
   - **`DisasterResponse.db`**: SQLite database containing disaster message data and category labels.
   - **`disaster_categories.csv`**: CSV file with categories of disaster messages.
   - **`disaster_messages.csv`**: CSV file with disaster messages.

### 2. `models/`
   - **`classifier.pkl`**: Pickled machine learning model used for message classification.

### 3. `app/`
   - **`run.py`**: Main script to run the Flask web application.
   - **`templates/`**:
     - **`master.html`**: Main web page with data visualizations and user input.
     - **`go.html`**: Results page that displays predictions made by the model.

### 4. `data_processing.py`
   - Python script to clean the raw data and save it into a SQLite database.
     - **`load_data()`**: Loads and merges the messages and categories datasets.
     - **`clean_data()`**: Cleans the data, splitting categories into individual columns and converting values into binary.
     - **`save_data()`**: Saves the cleaned data to a SQLite database.

### 5. `train_classifier.py`
   - Python script to train a machine learning model using the cleaned data.
     - **`build_model()`**: Builds a machine learning pipeline for multi-output classification.
     - **`evaluate_model()`**: Evaluates the model and prints classification reports.
     - **`save_model()`**: Saves the trained model to a pickle file.
   
### 6. `run.py`
   - Main script to launch the Flask web application. This file runs the web app, handles user queries, and visualizes data.
     - Includes features like displaying message genre distribution, message category distribution, and proportion of messages by genre.

---

## How to Run the Python Scripts and Web App

### 1. **Set up your environment**

   pip install -r requirements.txt
   
### 2. **Clean the Data**

   python data/data_processing.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
   
### 3. **Train the Classifier**

   python models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl
   
### 4. **Start the Web Application**

   python app/run.py
