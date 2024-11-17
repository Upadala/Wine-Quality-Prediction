


# Wine Quality Test Project

This project involves building a machine learning model to predict the quality of wine based on its chemical properties. The dataset contains various attributes such as acidity, pH, alcohol content, and sulfur dioxide levels, among others. The goal is to predict the quality of wine on a scale from 0 to 10, using classification algorithms.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation Instructions](#installation-instructions)
- [How to Run](#how-to-run)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project demonstrates how to use machine learning techniques to predict the quality of wine based on various chemical properties. We applied different models such as Logistic Regression, Decision Trees, and Random Forests to classify wine quality, comparing performance using metrics like accuracy, precision, recall, and F1-score.

Key objectives of the project:
- Explore the dataset and preprocess it.
- Train multiple machine learning models to predict wine quality.
- Evaluate and compare the models' performances.
- Visualize data distributions and model results.

## Technologies Used

- **Python**: Main programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib / Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models and evaluation metrics.
- **Jupyter Notebook**: For interactive development and analysis.

## Dataset

The dataset used for this project is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). It contains two datasets: one for red wine and one for white wine. This project uses the **red wine dataset** for training and testing the model.

### Dataset Description

- **Fixed acidity**: The amount of acid in wine.
- **Volatile acidity**: A byproduct of fermentation that may affect wine taste.
- **Citric acid**: Contributes to the taste and stability of wine.
- **Residual sugar**: The sugar left after fermentation.
- **Chlorides**: The salt content in wine.
- **Free sulfur dioxide**: Prevents oxidation and bacterial growth.
- **Total sulfur dioxide**: Total sulfur dioxide in wine.
- **Density**: The wine's density.
- **pH**: Acidity level of the wine.
- **Sulphates**: Contributes to the wine's aroma.
- **Alcohol**: Percentage of alcohol in the wine.
- **Quality**: The target variable (wine quality on a scale of 0 to 10).

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wine-quality-test.git
   cd wine-quality-test
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install the libraries individually:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

## How to Run

1. Open the Jupyter Notebook file (`wine_quality_analysis.ipynb`).
2. Run all cells to load the dataset, preprocess it, train models, and visualize results.
3. Alternatively, you can run the script from the command line to see the results:
   ```bash
   python wine_quality_analysis.py
   ```

## Model Evaluation

The models evaluated in this project include:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

Each model is evaluated using the following metrics:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Measure of the model's ability to avoid false positives.
- **Recall**: Measure of the model's ability to capture all true positives.
- **F1 Score**: Harmonic mean of precision and recall.

### Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.75     | 0.74      | 0.76   | 0.75     |
| Decision Tree         | 0.78     | 0.79      | 0.77   | 0.78     |
| Random Forest         | 0.80     | 0.81      | 0.79   | 0.80     |

Random Forest Classifier yielded the best results, with the highest accuracy and balanced precision and recall.

## Results

- The project demonstrated that machine learning can effectively predict wine quality using chemical properties.
- **Random Forest** achieved the best performance, showing the importance of ensemble methods in classification tasks.
- Feature importance analysis highlighted the significance of **alcohol** and **volatile acidity** in predicting wine quality.

## Contributing

Contributions to improve the project are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

Steps for contributing:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your fork.
4. Create a pull request with a clear description of your changes.


``
