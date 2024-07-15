# Stock Price Prediction Using Machine Learning and LSTM

This project involves predicting the stock prices of Reliance Industries using historical stock price data and sentiment analysis from news articles. We use multiple machine learning models, including Lasso Regression and Long Short-Term Memory (LSTM) networks, to predict future stock prices.

## Table of Contents

1. [Installation](#installation)
2. [Data Collection](#data-collection)
3. [Sentiment Analysis](#sentiment-analysis)
4. [Feature Scaling](#feature-scaling)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Prediction](#prediction)
7. [Results](#results)
8. [Real-world Applications](#real-world-applications)
9. [Conclusion](#conclusion)

## Installation

To get started, install the required libraries using the following command:

```sh
pip install numpy scipy scikit-learn keras tensorflow matplotlib pandas aylien_news_api nltk rpy2 textblob vaderSentiment yfinance requests plotly

Data Collection
We use yfinance to download historical stock price data for Reliance Industries. The data is saved in a CSV file named historical_prices.csv.

Sentiment Analysis
We fetch sentiment data related to Reliance Industries from the GDELT API. The sentiment analysis is performed using the VADER Sentiment Analyzer, which scores the sentiment of news article titles. The results are saved in a CSV file named sentiment_data.csv.

Feature Scaling
We merge the historical stock prices and sentiment data on the date. Only numeric columns are selected for scaling using the MinMaxScaler. The scaled data is split into training and testing datasets.

Model Training and Evaluation
Lasso Regression
LassoCV Model: We use cross-validated Lasso regression to train and evaluate the model. The model is trained on the training dataset and evaluated using metrics such as RMSE, MAE, and R².
Grid Search: We perform a grid search with cross-validation to find the best hyperparameters for the Lasso model.
Randomized Search: We also perform a randomized search with cross-validation to optimize the Lasso model.
Elastic Net
Elastic Net Model: We use Elastic Net, a combination of Lasso and Ridge regression, to train and evaluate the model. We perform a randomized search to find the best hyperparameters.
Feature Selection
RFE with Lasso: We use Recursive Feature Elimination (RFE) with Lasso regression to select important features and train the model.
LSTM Model
We use Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to predict stock prices based on historical data. The model is trained on sequences of past stock prices to predict future prices.

Prediction
Next Day Prediction: We predict the closing price of the stock for the next day using the trained models.
Next Week Prediction: We iteratively predict the closing prices for the next 7 days using the LSTM model.
Results
The models' performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) for both the training and testing datasets. The results are plotted to visualize the actual vs. predicted stock prices.

Real-world Applications
Investment Decision Making: Accurate stock price predictions can help investors make informed decisions about buying or selling stocks.
Algorithmic Trading: Automated trading systems can use these predictions to execute trades based on predefined strategies.
Risk Management: Financial institutions can use these models to assess and manage the risk associated with their investment portfolios.
Market Analysis: Analysts can use the sentiment analysis data to understand market trends and investor sentiment towards specific stocks.
Conclusion
This project demonstrates how machine learning models and LSTM networks can be used to predict stock prices using historical data and sentiment analysis. The integration of sentiment analysis with stock price prediction provides a more comprehensive approach to understanding market movements and making informed investment decisions.
