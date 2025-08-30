# 🏠 Housing Price Prediction

## 📌 Project Overview
This project demonstrates how to **predict house prices** based on various property features with a special focus on **normalization techniques**, **categorical encoding**, and **interaction terms**.

- 🛠 **Feature Engineering** – applied techniques for handling categorical features, creating interaction terms, and transforming variables
- 🧪 **Normalization Comparison** – evaluated different scaling methods including Standard, Min-Max, and Robust scaling
- 🤖 **Machine Learning** – trained regression models to predict house prices with comprehensive evaluation metrics
- 🎯 **Goal** – improve prediction performance and understand how different normalization techniques impact model performance

## Dataset
The Housing dataset contains samples of houses with the following key features:
- Area (square ft)
- Number of bedrooms and bathrooms
- Stories
- Parking availability
- Various amenities (mainroad, guestroom, basement, etc.)
- Furnishing status
- Location attributes (preferred area)

Each house has an associated price which is our target variable for prediction.

## Key Techniques Implemented

1. **Data Exploration & Visualization**
   - Distribution analysis of house prices
   - Correlation analysis between numerical features
   - Feature relationship visualization with scatter and box plots
   - Analysis of categorical features' impact on price

2. **Feature Engineering**
   - Binary encoding for yes/no categorical features
   - Ordinal encoding for furnishing status
   - Creation of meaningful interaction terms:
     - Area per bedroom
     - Total rooms (bedrooms + bathrooms)
     - Premium property indicator
     - Area and location interaction
     - Stories and area interaction

3. **Data Preprocessing & Normalization**
   - Train-test split for model evaluation
   - Comparison of three scaling techniques:
     - StandardScaler (Z-score normalization)
     - MinMaxScaler (0-1 scaling)
     - RobustScaler (median and IQR based scaling)
   - Visual comparison of scaling impacts on feature distributions

4. **Model Training & Evaluation**
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Comprehensive metrics:
     - RMSE (Root Mean Squared Error)
     - R² (Coefficient of Determination)
   - Actual vs. Predicted price visualizations

5. **Model Optimization**
   - Hyperparameter tuning for best-performing model
   - Comparison of scaling techniques' impact on model performance
   - Feature importance analysis to identify key price predictors

## Results
The analysis revealed that Gradient Boosting generally performed best for housing price prediction, with the RobustScaler showing advantages for handling outliers present in real estate data. The feature engineering steps, particularly the creation of interaction terms, contributed significantly to model performance.

Key findings:
- Linear Regression showed the least overfitting but more modest performance
- Random Forest exhibited strong training performance but significant overfitting
- Gradient Boosting offered the best balance of performance and generalization
- RobustScaler proved valuable for handling the skewed distributions common in housing data

## Requirements
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Use
1. Clone this repository
2. Ensure you have all required packages installed
3. Open the Jupyter notebook `House_Price_Prediction.ipynb`
4. Run the cells sequentially to see the analysis and results

## File Structure
- `House_Price_Prediction.ipynb` - The main Jupyter notebook containing all analysis and code
- `Housing.csv` - The dataset file
- `README.md` - This file with project information

## Key Insights
- Feature engineering, particularly interaction terms like area_per_bedroom and premium_property, significantly improves prediction performance
- Normalization technique selection matters - RobustScaler performs better when data contains outliers (common in housing prices)
- Gradient Boosting models offer superior performance for house price prediction tasks
- The balance between model complexity and generalization is critical - Random Forest shows signs of overfitting despite strong training performance
- Visualization of actual vs. predicted prices reveals that all models struggle somewhat with very high-priced properties
