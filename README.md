
Mini Project: Logistic Regression
Logistic Regression models the probability that a given input belongs to a certain class. It's suitable when the target variable is categorical and represents two classes (e.g., 0 or 1, True or False, Yes or No), although it can also be extended for problems with more than two classes. The key idea behind logistic regression is to model the relationship between the input variables (features) and the probability of the outcome.

In logistic regression, the linear combination of input features is transformed using a logistic function (also known as the sigmoid function), which ensures that the output is between 0 and 1. This output can be interpreted as the probability of the instance belonging to a particular class.

Advantages of Logistic Regression:

Simple and Interpretable: Logistic regression is a straightforward algorithm that's relatively easy to understand and interpret. The output is the probability of belonging to a certain class, and the coefficients associated with each feature provide insights into feature importance.

Efficient for Small Datasets: It works well with small datasets where the number of samples is not very large. It's less prone to overfitting in such cases compared to complex models.

Works Well for Linearly Separable Data: When the classes are separable by a linear decision boundary, logistic regression can perform quite well.

Good Starting Point: Logistic regression is often used as a starting point for understanding a classification problem. It can serve as a baseline model against which more complex algorithms can be compared.

Regularization: Logistic regression can be regularized to prevent overfitting. Regularization techniques like L1 (Lasso) and L2 (Ridge) regularization can be applied to control the complexity of the model.

Probability Estimation: Logistic regression not only provides class predictions but also outputs the probability of the prediction. This can be useful for decision-making in cases where the cost of misclassification varies.

However, it's important to note that logistic regression also has its limitations. It assumes a linear relationship between features and the log-odds of the target variable, which might not be suitable for highly complex relationships. Additionally, it might struggle with handling non-linear data without feature transformations. In such cases, more advanced techniques like decision trees, random forests, or neural networks might be more appropriate.

In this project you'll get some experience building a logistic regression model for the Wisconsin Breast Cancer Detection dataset. Note, the task of training a logistic regression model has largely been asbtracted away by libraries like Scikit-Learn. In this mini-project we'll focus more on model evaluation and interpretation.

First, let's import all the libraries we'll be using.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
     
Logistic Regression, like many statistical methods, comes with certain assumptions about the underlying data. Here are the main assumptions associated with logistic regression:

Binary Outcome: Logistic regression is designed for binary classification problems, where the dependent variable (target) should have two categories or levels. If you have a multi-class problem, you would typically use multinomial logistic regression or other appropriate techniques.

Independence of Observations: The observations (samples) should be independent of each other. This means that the outcome of one observation should not be influenced by the outcome of another observation. This assumption is often met when the data is collected using random sampling or experimental designs.

Linearity of Log-Odds: The relationship between the log-odds of the outcome and the predictors should be linear. In other words, the log-odds of the outcome variable should change linearly with changes in the predictor variables. This assumption can be checked by examining scatter plots and residual plots.

No Multicollinearity: There should not be high multicollinearity among the predictor variables. Multicollinearity can make it difficult to determine the individual effect of each predictor on the outcome. Techniques like variance inflation factor (VIF) can be used to assess multicollinearity.

Large Sample Size: While logistic regression is generally more robust to violations of assumptions compared to linear regression, having a reasonably large sample size helps in obtaining stable and reliable parameter estimates.

Sufficient Variability in the Outcome: The outcome variable should exhibit variation across different values of the predictor variables. If all values of a predictor are the same within a level of the outcome, the model may not be able to estimate the effect of that predictor.

No Extreme Outliers: Extreme outliers can influence the estimation of coefficients and affect the overall performance of the model. It's a good practice to identify and handle outliers before fitting the model.

It's important to note that while these assumptions are important to understand, logistic regression is often used in real-world scenarios where some of these assumptions may not be perfectly met. In such cases, it's crucial to assess the impact of potential violations on the model's results and make informed decisions about the model's suitability and reliability.

If the assumptions are significantly violated, it might be worth considering other techniques like decision trees, random forests, or support vector machines, which might be more robust to certain types of data characteristics. One of the best ways to see if logistic regression is suitable for a problem is to simply train a logistic regression model and evaluate it on test data.

Here are your tasks:

Load the breast cancer data into a Pandas dataframe and create variables for the features and target.
Do a little exploratory data analysis to help familiarize yourself with the data. Look at the first few rows of data, for example. Generate some summary statistics for each feature. Look at the distribution of the target variable. Maybe create a pair-plot for some of the variables. Create a heatmap of correlation between features. Is the multicollinearity assumption broken? Also, generate some boxplots to see how feature distributions change for each target. This part is a bit open-ended. Be creative!

# Load data and split into feature and target variables
     

# View first 5 rows of the data
     

# How frequently does the positive target occur?
     

# Generate summary statistics for the data
     

# Create a pairplot for the first few features
     

# Create a correlation coefficeint heatmap
     

# Create a boxplot for mean radius by target type
     
With a better feel for the data, it's time to attempt to build a logistic regression model.

Use train_test_split to create a training and test sets for the data.
Use LogisticRegression to train a model on the training data. Make sure you understand the inputs to the model. Try using the "liblinear" solver here.

# Split data into training and test sets
     

# Build and train logistic regression model
     
As you can see, training a logistic regression model is simple. The more important task is evaluating the model and determining if it's any good. For classification problems, a good starting point for model evaluation is the confusion matrix.

A confusion matrix is a fundamental tool for evaluating the performance of a classification model. It provides a clear and detailed breakdown of how well a model's predictions align with the actual outcomes in a binary classification problem. It's particularly useful for understanding the types of errors a model is making.

A confusion matrix is typically presented as a table with four entries:

True Positives (TP): The number of instances that were correctly predicted as positive (belonging to the positive class).

True Negatives (TN): The number of instances that were correctly predicted as negative (belonging to the negative class).

False Positives (FP): Also known as a Type I error. The number of instances that were predicted as positive but actually belong to the negative class.

False Negatives (FN): Also known as a Type II error. The number of instances that were predicted as negative but actually belong to the positive class.

Here's how these four components fit into the confusion matrix:

                Predicted
               |  Positive  |  Negative
Actual  Positive |    TP      |    FN
        Negative |    FP      |    TN
Each cell of the confusion matrix represents a specific classification outcome. The goal is to have as many instances as possible in the TP and TN cells, and as few as possible in the FP and FN cells.

From the confusion matrix, several evaluation metrics can be calculated:

Accuracy: The proportion of correctly classified instances out of the total instances.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision: The proportion of correctly predicted positive instances out of all predicted positive instances. It measures the model's ability to avoid false positives.

Precision = TP / (TP + FP)

Recall (Sensitivity or True Positive Rate): The proportion of correctly predicted positive instances out of all actual positive instances. It measures the model's ability to capture all positive instances.

Recall = TP / (TP + FN)

F1-Score: The harmonic mean of precision and recall. It provides a balanced measure that takes into account both false positives and false negatives.

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

Confusion matrices provide valuable insights into the strengths and weaknesses of a classification model. They allow you to understand where the model is making mistakes and guide further improvements or adjustments.

Here are your tasks:

Use your model to make predictions on the test data.
Generate a confusion matrix with the test results. How many false positives and false negatives did the model predict?
Use classification_report to generate further analysis of your model's predictions. Make sure you understand everything in the report and are able to explain what all the metrics mean.
Note, the macro average in the report calculates the metrics independently for each class and then takes the average across all classes. In other words, it treats all classes equally, regardless of their frequency in the dataset. This can be useful when you want to assess the model's overall performance without being biased by the class imbalances.

The weighted average in the report, on the other hand, calculates the metrics for each class and then takes the average, weighted by the number of true instances for each class. This gives more weight to classes with more instances, which can be particularly useful in imbalanced datasets where some classes might have much fewer instances than others.


# Evaluate the model
     

# Generate a confusion matrix
     

# Generate a classification report
     
Feature importance refers to the process of determining and quantifying the contribution of each feature (also known as predictor variable or attribute) in a machine learning model towards making accurate predictions. It helps in understanding which features have the most significant impact on the model's output and can be crucial for interpreting and explaining the model's behavior.

In logistic regression models, you can calculate feature importance by examining the coefficients associated with each feature. These coefficients indicate the change in the log-odds of the target variable for a one-unit change in the corresponding feature, while keeping other features constant. The magnitude of the coefficient reflects the strength of the impact that the feature has on the predicted outcome.

The magnitude of the coefficients indicates the importance of each feature. Larger magnitudes imply a stronger impact on the predicted probability of the positive class.

Positive Coefficient: An increase in the feature value leads to an increase in the log-odds of the positive class, implying a higher probability of belonging to the positive class.

Negative Coefficient: An increase in the feature value leads to a decrease in the log-odds of the positive class, implying a lower probability of belonging to the positive class.

Remember that the scale of the features matters when interpreting coefficients. If features are on different scales, their coefficients won't be directly comparable. This is where normalization can be helpful. Also, keep in mind that this interpretation assumes a linear relationship between the features and the log-odds of the target variable. If your logistic regression model includes interactions or polynomial terms, the interpretation can become more complex. Additionally, be cautious about interpreting coefficients as causal relationships, as logistic regression only captures associations, not causal effects.

Here are your tasks:

Extract the model coefficients from your trained model.
Normalize the coefficients by the standard deviation of each feature in the training data.
Sort feature names and coefficients by absolute value of coefficients.
Visualize the feature importances by creating a horizontal bar chart using e.g. barh. Based on magnitude, what appears to be the most important predictor of cancer in this dataset?

# Extract coefficients
     

# Normalize the coefficients by the standard deviation
     

# Sort feature names and coefficients by absolute value of coefficients
     

# Visualize feature importances
     
