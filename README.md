DeepEstate: AI-Powered Real Estate Price Prediction

Project Overview
DeepEstate is a deep learning-based framework developed for real estate price prediction and recommendation, focused on properties in Bengaluru, India. The project uses structured tabular data and machine learning models to provide accurate, personalized, and real-time property sale price estimations. It includes a Streamlit web app interface for user-friendly interaction.

Motivation
Establishing property value accurately is complex and time-consuming. This project aims to simplify real estate decision making by leveraging AI and machine learning to predict property prices based on multiple factors like location, amenities, and property type, empowering buyers, renters, and investors to make data-driven decisions.

Features
-Property price estimation using Multi-Layer Perceptron (MLP) regression.
-Interactive Streamlit web app for live user input and price prediction.
-Categorical feature encoding for locations, furnishing status, property type.
-Scalable architecture to extend model for rental price prediction, market trend forecasting, and recommendations.
-Worked end-to-end entirely on Google Colab with easy deployment via ngrok.

Technologies Used
-Python 3.x
-TensorFlow and Keras for model building.
-scikit-learn for machine learning utilities.
-Streamlit for app frontend.
-pyngrok for secure public URL tunneling on Google Colab.
-NumPy and Pandas for data handling and preprocessing.

Dataset
The project uses a curated real estate dataset containing property features like area, bedrooms, bathrooms, type (apartment, villa, etc.), furnishing status, locality, and prices. The dataset was collected from online real estate portals and open datasets.

Installation & Usage
-Open the Google Colab notebook.
-Upload necessary files (model.pkl, encoders, dataset CSV) if not already.
-Run cells sequentially to install dependencies, preprocess data, train models, and deploy the Streamlit app.
-Use the public ngrok URL generated during runtime to interact with the live app.

Project Structure
app.py: Streamlit app script.

model.pkl: Trained ML model file.

Encoder files for categorical transformations.

Dataset CSV file for training and evaluation.

Notebook with all data processing and training steps.

Future Work
-Incorporate multi-modal data (images, zoning maps).
-Improve model performance with ensemble and deep learning approaches.
-Add rental price estimation and property recommendation modules.
-Deploy on dedicated cloud services for production.

Acknowledgments
-Real estate portals and open data providers.
-Streamlit, TensorFlow, and Google Colab community for tools and resources.
