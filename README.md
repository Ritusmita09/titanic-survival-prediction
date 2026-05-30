# Titanic Survival Prediction

A Machine Learning project that predicts passenger survival on the RMS Titanic using Logistic Regression.

## Project Overview

This project analyzes the Titanic dataset and builds a predictive model to determine whether a passenger survived based on features such as:
- Passenger class
- Gender
- Age
- Fare paid
- Port of embarkation
- And more

## Dataset

- **Source**: Titanic dataset from [Data Science Dojo](https://github.com/datasciencedojo/datasets)
- **Records**: 891 passengers
- **Features**: 12 attributes

## Model Performance

- **Algorithm**: Logistic Regression
- **Test Accuracy**: ~82%
- **Cross-Validation Score**: ~80% ± 3%

## Files

- `titanic_survival_prediction.py` - Main project script with EDA, preprocessing, training, and evaluation

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

```bash
python titanic_survival_prediction.py
```

The script will:
1. Load and explore the Titanic dataset
2. Generate EDA visualizations
3. Preprocess and prepare the data
4. Train a Logistic Regression model
5. Evaluate model performance
6. Make live predictions on test cases
7. Save analysis plots as PNG files

## Outputs

- `titanic_eda_dashboard.png` - Exploratory Data Analysis visualizations
- `titanic_model_evaluation.png` - Model performance and feature importance

## Author

- **Name**: Ritusmita
- **Department**: Electronics & Communication Engineering (ECE-B)
- **Institute**: RCC Institute of Information Technology

## License

This project is open source and available under the MIT License.
