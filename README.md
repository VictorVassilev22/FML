# Machine Learning Techniques for Predicting Price Movements in the Stock Market

## Overview
This project focuses on predicting the 30-day future close price percent change of a stock. The dataset, sourced from Yahoo Finance through their free API, spans daily stock price data from 2017 or 2018 to the present. To explore and model this data, a variety of machine learning techniques were employed, including linear regression, random forest regressor, gradient boosting regressor, and a neural network.
### Models Used
1. **Linear Regression:** A fundamental model for predicting numerical values, providing interpretability and simplicity. (KO)
2. **Random Forest Regressor:** A powerful ensemble model leveraging decision trees, capable of capturing complex relationships in the data. (KO)
3. **Gradient Boosting Regressor:** Another ensemble method that builds models sequentially, each correcting the errors of the previous one, often delivering high predictive performance. (AMD)
4. **Neural Network (NN) with Dense Layers:** A deep learning approach using a simple neural network with dense layers for complex feature learning. Experiment with custom loss function. (AMD)

### Tasks Performed
1. **Feature Engineering:** The dataset underwent feature engineering to extract relevant information and create new features that might enhance predictive performance. I wrapped pandas_ta in my internal library ta_features for easier creation of features in the dataset.
2. **Feature Selection:** To improve model efficiency and reduce noise, feature selection techniques were applied to choose the most informative variables.
3. **Hyperparameter Optimization:** Hyperparameter optimization was performed to fine-tune the models for better predictive accuracy.
4. **Feature Transformations:** Transformations were applied to certain features to enhance their representation and improve model performance.
5. **Visualization** Matplotlib employed for creating visualizations to aid in the exploratory data analysis and presentation of results.

### Tools and Libraries
- **scikit-learn (sklearn):** Utilized for implementing machine learning models, feature selection, and hyperparameter tuning.
- **Statsmodels:** Employed for statistical analysis and hypothesis testing, providing additional insights into the relationships within the data.
- **Keras:** Used for developing the neural network model, leveraging dense layers for feature learning.

## Dataset
Used yahoo finance API 
- **Stock Symbol:** Used AMD (Advanced Micro Devices, Inc) and KO (The Coca-Cola Company) free stock data
- **Time Period:** 2017 to 2024
- **Frequency:** Daily
The dataset consists of historical daily stock price information for a particular stock. The key features include:
- **Date:** The date of the stock price data.
- **Open:** The opening price of the stock on a given day.
- **High:** The highest price the stock reached during the day.
- **Low:** The lowest price the stock reached during the day.
- **Close:** The closing price of the stock on a given day.
- **Volume:** The trading volume of the stock on a given day.
- **Adjusted Close:** The adjusted closing price, accounting for corporate actions such as dividends and stock splits.

*Note: Read the notebooks in the order of the listing of the models. Happy reviewing!*