# Customer Churn Prediction

This capstone notebook builds and evaluates a machine-learning model for predicting customer churn. It uses a reproducible **synthetic dataset** generated in the notebook; no external data files are needed.

## Project workflow

- Generate 4,000 synthetic customer records with numeric account activity and categorical plan details.
- Explore class balance and compare churn patterns across customer features.
- Identify and remove a post-outcome feature (`exit_survey_sent`) that leaks the churn label.
- Split the data into stratified training, validation, and test sets.
- Preprocess numeric and categorical features in a scikit-learn pipeline.
- Compare Logistic Regression, Random Forest, Gradient Boosting, and an RBF Support Vector Machine using stratified cross-validation and ROC-AUC.
- Tune a Gradient Boosting model, evaluate its test performance, and inspect feature importance.
- Compare precision and recall at different probability thresholds, with an emphasis on detecting churners.

## Requirements

- Python 3.9 or later
- Jupyter Notebook support, such as the Jupyter extension in Visual Studio Code

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the notebook

Open `day16_capstone.ipynb` in Jupyter or Visual Studio Code and run all cells from top to bottom. The notebook generates the data and trains the models as it runs; the fixed random seed makes the data generation and splits reproducible.

## Notes

The data is simulated for learning purposes. Model metrics and feature relationships should not be interpreted as findings about real customers or used for business decisions without validation on representative data.
