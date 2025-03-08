# Restaurants Classification
 
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://share.streamlit.io/gamal-abdelhakm/Restaurants-Classification/main/app.py)

Customer churn prediction, achieving the highest accuracy with Decision Tree Classifier.

## Project Overview

This project aims to predict the success of new restaurants in Bangalore based on various features such as location, type of restaurant, cuisines offered, and availability of online ordering and table booking options. The project utilizes a dataset from Zomato and employs machine learning techniques to build a predictive model.

## Deployment
The model is deployed as a web application using Streamlit. You can access the application using the following link:
[Restaurant Success Prediction App](https://restaurants-classification-k23fyhqysnk7qwkp8uurc7.streamlit.app/)

## Summary Video
A video summarizing what I did in the code can be found using the following link:
[Summary Video](https://drive.google.com/file/d/19raEZfO71gAQpLGRPStTa8Js3_K-JaaD/view)

## Dataset

The dataset used in this project is `zomato.csv`, which contains information about various restaurants in Bangalore, including their ratings, votes, location, types, cuisines, and more.

## Jupyter Notebook

The Jupyter Notebook `Zomato.ipynb` includes the following steps:

- **Data Import and Libraries**: Import necessary libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, and `xgboost`.
- **Data Exploration**: Load and explore the dataset, handle missing values, and remove duplicated data.
- **Data Preprocessing**: Clean the data by dropping unwanted columns, filling missing values, and encoding categorical features.
- **Feature Engineering**: Extract useful features for model building.
- **Data Summarization and Visualization**: Provide data summaries and visualizations to understand the distribution and relationships within the dataset.

## Streamlit Application

The `app.py` file is a Streamlit application script that allows users to predict the success of a new restaurant in Bangalore. The application includes:

- **User Inputs**: Collects user inputs for various features such as approximate price, location, restaurant type, cuisines, and availability of online ordering and table booking options.
- **Model Prediction**: Preprocesses the inputs and makes predictions using a pre-trained model.
- **Result Display**: Displays the prediction result, indicating whether the restaurant will succeed or fail.

## How to Run the Streamlit Application

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open the provided URL in your browser to access the application.

## Conclusion

This project demonstrates the use of machine learning techniques to predict the success of new restaurants based on various features. The Streamlit application provides an interactive way for users to input restaurant details and receive predictions.


