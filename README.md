# NBA-Player-Prediction

## Introduction

This repository contains the code and datasets used in my project of predicting NBA star Stephen Curry's game points for over/under betting purposes.

## Project Overview

1. **Data Collection:** Utilized BeautifulSoup and PlayWright to scrape data from Basketball Reference.
2. **Data Preprocessing:** Cleaned the dataset by removing irrelevant or highly correlated features to reduce multicollinearity.
3. **Feature Engineering:** Calculated averages of various statistics over the previous three games to capture recent trends in Curry's performance.
4. **Data Splitting:** Split the data into training, validation, and test sets following a 60/20/20 split.
5. **Model Training & Selection:** Trained a linear regression model with a lasso penalty, a random forest model, and an XGBoost model. The linear regression model with a lasso penalty was selected for its lowest validation RMSE.
6. **Model Evaluation:** Evaluated the selected model on the test set, achieving an RMSE of 7.3299.

## File Descriptions

- `Data_Collection.ipynb`: Jupyter Notebook containing the webscraping process
- `Preprocessing_and_Feature_Engineering.ipynb`: Jupyter Notebook containing the data preprocessing and feature engineering process
- `Model_Training_and_Selection.ipynb`: Jupyter Notebook containing the model training, selection and hyperparameter tuning process
- `Model_Evaluation.ipynb`: Jupyter Notebook containing the final model evaluation on the test set
- `original_dataframe.csv`: CSV file containing the original dataframe
- `final_dataframe.csv`: CSV file containing the final dataframe after preprocessing and feature engineering

## Results

Our final selected model, a linear regression model with a lasso penalty, achieved a Root Mean Square Error (RMSE) of 7.3299 on the test set. RMSE is a popular metric that measures the average magnitude of the error, essentially telling us how far, on average, our predictions are from the actual values. 

In practical terms, an RMSE of 7.3299 means that our model's predictions about Stephen Curry's points per game are, on average, around 7.33 points off the actual points he scores. For instance, if the model predicts that Curry would score 25 points in a game, the actual score would, on average, be within a range of approximately 17.67 to 32.33 points.

This degree of accuracy offers promising potential for over/under betting scenarios. However, it's essential to note that while the model provides us with a mathematical advantage, it does not guarantee successful bets due to various unpredictable factors like last-minute injuries, changes in team strategy, and more. As such, this model should be used as a tool to supplement your betting strategy, not define it entirely.

One interesting insight gathered from the results is that Stephen Curry's performance trends, when averaged over a span of three games, provide a more robust predictive foundation than a single game's statistics. It showcases the importance of considering a player's recent performance trajectory in sports analytics.

I also discovered that the linear regression model with a lasso penalty outperformed more complex models like random forest and XGBoost for this particular problem. It not only delivered lower RMSE but also highlighted the power of regularization in enhancing model performance and interpretability. Lasso regression, with its feature selection property, proved quite effective in this context where we started with a high-dimensional dataset.


## How to Use

1. Clone this repository.
2. Install required Python packages: BeautifulSoup, PlayWright, pandas, numpy, sklearn, xgboost.
3. Run Jupyter Notebooks in the order specified in the project overview.

## Future Improvements

While the model performed well in this project, there are several potential improvements for the future:

- Incorporate additional data, such as player health data, opponent data, or more detailed play-by-play data.
- Experiment with different model architectures, such as deep learning models, to capture more complex patterns.
- Develop a web application to deliver real-time predictions during the NBA season.
