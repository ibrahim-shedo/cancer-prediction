# Cancer Diagnosis Prediction Using Logistic Regression

## Project Overview

This project aims to build a logistic regression model to predict whether a tumor is benign (B) or malignant (M) using various tumor characteristics. The model uses features such as `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, and others to predict the diagnosis (benign or malignant).

## Dataset

The dataset consists of 33 columns, each representing different characteristics of the tumor. The key columns include:

- **id**: Unique identifier for each record.
- **diagnosis**: Target variable (M = Malignant, B = Benign).
- **radius_mean**: Mean of distances from center to points on the perimeter.
- **texture_mean**: Standard deviation of gray-scale values.
- **perimeter_mean**: Mean size of the tumor perimeter.
- **area_mean**: Mean area of the tumor.
- **smoothness_mean**: Local variation in radius lengths.
- **compactness_mean**: Perimeter^2 / area - 1.0.
- **concavity_mean**: Severity of concave portions of the contour.
- **concave points_mean**: Number of concave points on the tumor.
- **texture_worst**, **perimeter_worst**, **area_worst**, etc.: Worst or largest values of these characteristics.

**Note**: The `Unnamed: 32` column is irrelevant and contains NaN values, so it will be removed during preprocessing.

## Project Structure

This project contains the following steps:

1. **Data Preprocessing**:
   - Convert the `diagnosis` column to a binary format (`M = 1`, `B = 0`).
   - Drop the `id` and `Unnamed: 32` columns as they are unnecessary for modeling.
   - Split the data into training and testing sets.
   - Normalize or scale the feature columns for better performance.

2. **Model Training**:
   - A logistic regression model will be trained on the dataset to predict whether a tumor is benign or malignant.
   
3. **Evaluation**:
   - The model’s performance will be evaluated using metrics like accuracy, confusion matrix, precision, recall, and F1-score.

## Requirements

The following libraries are required for running the model:

- `pandas`: For data manipulation.
- `scikit-learn`: For model building and evaluation.
- `numpy`: For numerical computations.

You can install the required libraries using the following command:

Conclusion
This project demonstrates the use of logistic regression to predict whether a tumor is benign or malignant based on several physical characteristics. The logistic regression model is evaluated using performance metrics such as accuracy, precision, recall, and confusion matrix to measure its effectiveness in diagnosing cancer.
