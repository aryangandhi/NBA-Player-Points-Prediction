# NBA Player Points Prediction

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

- `get_data.ipynb`: Jupyter Notebook containing the webscraping process
- `prediction_model.ipynb`: Jupyter Notebook containing the data preprocessing, feature engineering process, model training, selection and hyperparameter tuning process, the final model evaluation on the test set

## Data Exploration

### Histogram of Points Scored

<img width="374" alt="image" src="https://github.com/aryangandhi/NBA-Player-Prediction/assets/43526001/b5811338-0f86-4090-8df6-7f41e1975984">

This image shows us the distribution of points Stephen Curry has scored in a regular season game throughout the dataset (2015-2023).

<img width="536" alt="image" src="https://github.com/aryangandhi/NBA-Player-Prediction/assets/43526001/64a52c15-45ff-428b-899c-02668f1fbded">

This image shows us the spread of points Stephen Curry has scored against each opposing team throughout the dataset (2015-2023).

## Hyperparameter Tuning

Hyperparameter tuning is a crucial step in the machine learning pipeline. Hyperparameters are parameters that aren't learned from the data, but are set prior to the training process. They play a significant role in controlling the learning process and hence, affect the performance of a model. 

In the context of our Lasso Penalized Regression model, the primary hyperparameter we needed to optimize was the regularization parameter, often referred to as `alpha`.

### Tuning Alpha for the Linear Regression with Lasso Penalty (L1 Regression) Model

#### Regularization and the Importance of Alpha

Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. In the case of Lasso Regression, this penalty term leads to the reduction of the coefficients of less important features to zero, effectively performing feature selection.

The strength of the regularization is controlled by the hyperparameter `alpha`. A larger `alpha` increases the amount of regularization and model simplicity, while a smaller `alpha` makes the model more complex by reducing the amount of regularization. 

Hence, finding the right balance is key, which is what hyperparameter tuning aims to do.

#### Hyperparameter Tuning Process

To tune the `alpha` parameter, we utilized a simple approach of trying out a range of `alpha` values and selecting the one that resulted in the lowest Root Mean Squared Error (RMSE) on the validation set. The range of `alpha` values was defined using the `numpy` `linspace` function, creating 100 values between 0 and 0.5.

For each `alpha`, we trained a Lasso Regression model, made predictions on the training and validation sets, and calculated the RMSE for each set. These errors were stored and subsequently plotted against the corresponding `alpha` values.

#### Results

The output of our hyperparameter tuning process was as follows:

- Optimal Alpha Value: 0.22222222222222224
- Lowest Validation RMSE: 7.390895958059687

This indicates that an `alpha` of approximately 0.222 provided the optimal balance between bias and variance for our model, as evidenced by the lowest validation RMSE. 

### Importance of Cross-Validation

While our approach was effective in this context, due to the nature of the dataset being timeseries data, a more robust and widely-used method for hyperparameter tuning is cross-validation, typically k-fold cross-validation. This process involves dividing the dataset into 'k' subsets and training the model 'k' times, each time using a different subset as the validation set and the remaining data as the training set. The performance measure is then averaged over the 'k' trials to provide a more robust measure of model quality.

Cross-validation helps ensure that our model's performance is not overly dependent on the specific arrangement of the training and validation sets. Thus, it provides a more reliable estimate of how the model is likely to perform on unseen data.

In the future, it could be beneficial to employ a cross-validated approach, such as using `GridSearchCV` or `RandomizedSearchCV` from Scikit-Learn, to further enhance the robustness of our hyperparameter tuning process.


## Results

Our final selected model, a linear regression model with a lasso penalty, achieved a Root Mean Square Error (RMSE) of 7.3299 on the test set. RMSE is a popular metric that measures the average magnitude of the error, essentially telling us how far, on average, our predictions are from the actual values. 

In practical terms, an RMSE of 7.3299 means that our model's predictions about Stephen Curry's points per game are, on average, around 7.33 points off the actual points he scores. For instance, if the model predicts that Curry would score 25 points in a game, the actual score would, on average, be within a range of approximately 17.67 to 32.33 points.

This degree of accuracy offers promising potential for over/under betting scenarios. However, it's essential to note that while the model provides us with a mathematical advantage, it does not guarantee successful bets due to various unpredictable factors like last-minute injuries, changes in team strategy, and more. As such, this model should be used as a tool to supplement your betting strategy, not define it entirely.

One interesting insight gathered from the results is that Stephen Curry's performance trends, when averaged over a span of three games, provide a more robust predictive foundation than a single game's statistics. It showcases the importance of considering a player's recent performance trajectory in sports analytics.

I also discovered that the linear regression model with a lasso penalty outperformed more complex models like random forest and XGBoost for this particular problem. It not only delivered lower RMSE but also highlighted the power of regularization in enhancing model performance and interpretability. Lasso regression, with its feature selection property, proved quite effective in this context where we started with a high-dimensional dataset.


## How to Use

1. Clone this repository.
2. Install required Python packages: BeautifulSoup, PlayWright, pandas, numpy, matplotlib, sklearn, xgboost.
3. Run Jupyter Notebooks in the order specified in the project overview.

## Future Improvements

While the model performed well in this project, there are several potential improvements for the future:

- Incorporate additional data, such as player health data, opponent data, or more detailed play-by-play data.
- Experiment with different model architectures, such as deep learning models, to capture more complex patterns.
- Develop a web application to deliver real-time predictions during the NBA season.
