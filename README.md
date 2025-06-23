# Housing-Price
# California Housing Price Prediction 📈🏡

This project explores various machine learning regression techniques to predict median housing prices in California using engineered and preprocessed data from the [CS482 Housing Dataset](https://huggingface.co/datasets/Ryan-Pupia/CS482-HousingDataSet) on Hugging Face.

## 🚀 Project Highlights

- Applied multiple regression models: 
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Neural Network (MLP)
- Feature preprocessing including:
  - Standard scaling
  - One-hot encoding of categorical variables
  - Log-standardization of skewed features
- Model benchmarking using:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- Hyperparameter tuning for XGBoost using `GridSearchCV`
- Final model evaluation and feature importance visualization

## 🧪 Dataset

- Source: [Ryan-Pupia/CS482-HousingDataSet](https://huggingface.co/datasets/Ryan-Pupia/CS482-HousingDataSet)
- Contains log-transformed and pre-scaled versions of features such as:
  - Median income, total rooms, total bedrooms, households, etc.
  - One-hot encoded `ocean_proximity`
  - Normalized geographic coordinates (`latitude`, `longitude`)

## 📊 Model Performance

| Model             | R² Score | MAE   | MSE   |
|------------------|----------|-------|-------|
| Linear Regression| 0.66     | ...   | ...   |
| Random Forest    | 0.81     | ...   | ...   |
| XGBoost (tuned)  | **0.85** | ...   | ...   |

_(Actual metrics are printed in the notebook/script)_

## ⚙️ Hyperparameter Tuning (XGBoost)

Used `GridSearchCV` with 3-fold cross-validation over parameters:
- `n_estimators`: [100, 200, 300]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 4, 5]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]

## 📈 Visualizations

- Bar plot of model R² scores
- XGBoost feature importance chart

## 🛠 Libraries Used

- scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib
- Hugging Face Datasets

## 📂 How to Run

```bash
pip install -r requirements.txt
python housing_modeling.py
