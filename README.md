 # India House Price Prediction Using Machine Learning 

 # ***Project Overview*** : 
This project predicts house prices in India using machine learning. It analyzes property 
features such as location, size, number of bedrooms, age of the property, and nearby 
facilities to estimate the house price. The project follows an end-to-end pipeline from 
data collection to model building to create a reliable price prediction system

 # ***Problem Statement*** : 
 House prices in India change a lot from city to city, and even between nearby buildings. Many buyers and sellers depend on brokers or old listings for pricing, which often leads to wrong estimates and weak negotiation.
This project uses machine learning to predict a fair house price based on simple details like city, locality, size, number of bedrooms, bathrooms, furnishing, floor, and property age.
The goal is to give buyers, sellers, and platforms a quick and fair price estimate, making the buying and selling process easier and more transparent.
Success means: low prediction error and accurate results, along with a simple app/API that gives instant price predictions.

 # ***Business Understanding*** :
 House prices in India are hard to judge fairly — they depend on city, locality, size, and amenities, and most buyers just rely on broker quotes. This project uses machine learning to predict a fair price for a property based on its features, helping buyers, sellers, and platforms make better, data-backed decisions.

 # ***Dataset Information*** :

The dataset contains house listing details from Indian cities, used to train the price 
prediction model.

- **Source:** Kaggle (https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction)
- **Size:** ~2.5 lakh rows, **23** columns
- **Target column:** `price` (Price_in_Lakhs)
- **Features:**  ID,
State,
City,
Locality,
Property_Type,
BHK,
Size_in_SqFt,
Price_in_Lakhs,
Price_per_SqFt,
Year_Built,
Furnished_Status,
Floor_No,
Total_Floors,
Age_of_Property,
Nearby_Schools,
Nearby_Hospitals,
Public_Transport_Accessibility,
Parking_Space,
Security,
Amenities,
Facing,
Owner_Type,
Availability_Status,

 # ***Project Sturcture*** : 
 
  # ***Teconologies used*** :

  # ***project Workflow*** :
  # Tasks 01 : *Data Acquisition, Cleaning, and Exploratory Analysis*

 ##  1. The dataset was imported into a Pandas DataFrame using `pd.read_csv()`.


 Displayed the first five rows of thr dataset to verify successful data loading and understand the structure of thr data.

    


-Checking column data types (`dfdtypes)

All columns data types were correct and ready for analysis.

    - Checking dataset dimensions (`df.shape`)
                       (250000, 23)          
## 2.*Null value analysis:*  
Checked for missing values using `df.isnull().sum()` and calculated the percentage 
of nulls per column using `(df.isnull().sum() / df.shape[0]) * 100`.

**Result:** No missing values were found in any column of the dataset. Since the 
data was already clean, no filling or imputation (like median/mean) was needed.


## 3.Duplicate Detection

Checked for duplicate rows using `df.duplicated().sum()`.

**Result:** No duplicate rows were found, so `df.drop_duplicates()` was not needed. 
Since no rows were removed, there was no change in null percentages.
                           
                           np.int64(0)

## 4.Data Type Correction

Checked column data types using `df.dtypes`. All columns had correct types 
(numeric as `int64`/`float64`, text as `object`).

To optimize memory, the `Locality` column (a repetitive text column) was converted 
from `object` to `category` dtype.

**Memory usage before conversion:** e.g., 194.04 MB  
**Memory usage after conversion:** e.g., 24.41 MB

Converting `Locality` to `category` reduced memory usage since repeated text 
values are stored more efficiently in this format.


## 5. Descriptive Statistics & Skewness Analysis

### 
The column with the **highest absolute skewness** is **`Price_per_SqFt` (2.318668)**.

### Interpretation
- **Positive skew** → Long right tail (few very high values),  
- **Negative skew** → Long left tail (few very low values) , 
- **Near zero skew** → Approximately symmetric distribution , 

`Price_per_SqFt` shows a **positive skew**, meaning most properties cluster around typical values, but a few very expensive ones stretch the distribution to the right. This inflates the mean compared to the median.

### Data Quality Note
There are **no missing values** in the dataset.  
**Skewness analysis is included for **distribution understanding only**, not for imputation**.

## 6. Outlier Detection with IQR

We used the Interquartile Range (IQR) method to detect outliers in numeric columns.  
Formula:  
- Q1 = 25th percentile  
- Q3 = 75th percentile  
- IQR = Q3 − Q1  
- Lower Bound = Q1 − 1.5 × IQR  
- Upper Bound = Q3 + 1.5 × IQR  

Rows outside these bounds are flagged as outliers.

### Results
- Most numeric columns show **no outliers**.  
- **Price_per_SqFt** has **20,020 outliers**, far more than any other column.

### Interpretation
- Outliers in `Price_per_SqFt` represent properties priced much higher or lower than typical values.  
- These extreme values can distort averages and affect modeling.

## 7. Visualizations (all five types required):

### A line plot:
   ![alt text](image-2.png)

### A bar chart :
  ![alt text](image-3.png)

### Histogram of Most Skewed Column

We plotted a histogram of the most skewed numeric column (`Price_per_SqFt`) using `sns.histplot()` with 20 bins.

## Result
- The distribution is **right‑skewed** (positive skew).  
- Most property values are clustered at the lower end (0.0–0.2).  
- A few very high values stretch the tail to the right.  
![alt text](image-4.png)

###  Scatter Plot: Price_in_Lakhs vs Price_per_SqFt

We plotted a scatter plot between `Price_in_Lakhs` and `Price_per_SqFt` using `sns.scatterplot()`.

## Result
- The relationship is **positive**: as property price increases, price per square foot also tends to rise.  
- The correlation appears **moderate to strong**, forming an upward trend.
![alt text](image-5.png)
#  Box Plot: Property_Type vs Price_in_Lakhs

We plotted a box plot of `Price_in_Lakhs` split by `Property_Type` using `sns.boxplot()`.

## Result
- The **median prices** across Apartments, Independent Houses, and Villas are fairly similar.  
- The **spread (range of values)** is also comparable, showing overlapping distributions.  
- No major differences in central tendency or variability are visible between property types.
![alt text](image-6.png)

## 8.Correlation heat map:
I used a correlation heatmap to identify relationship 
between numerical fetures.
I found that price_in_lakhs and Price_per_SqFT has a moderate positve
correlation(0.56),while Size_in_SqFt and Price_per_SqFt showed a moderate
negative correlation(-0.61). Year_built and Age_of_property had a perfect negative correlation (-1.0) which is expected because older ,properties have earlier contrucation years

![alt text](image-7.png)

# 9.Imputation Strategy Comparison

We compared mean and median values for the two most skewed numeric columns before applying imputation.

## Results
- **Price_per_SqFt** → mean = 0.13, median = 0.09, skew = 2.32 (positive skew)  
- **Price_in_Lakhs** → mean = 254.59, median = 253.87, skew = 0.01 (almost symmetric)

## 9B.Spearman Rank Correlatio

We compared Spearman (rank-based) and Pearson (linear) correlations to detect monotonic but non-linear relationships.

## Top 3 Differences
1. **Price_in_Lakhs vs Price_per_SqFt** → Spearman (0.750) > Pearson (0.556) → monotonic but non-linear, rely on **Spearman**.  
2. **Size_in_SqFt vs Price_per_SqFt** → Pearson (-0.615) ≈ Spearman (-0.599) → approximately linear, rely on **Pearson**.  
3. **Price_per_SqFt vs Nearby_Schools** → very small difference, both near zero → no meaningful correlation.

## 9c.Grouped Aggregation

We grouped one categorical column against one numeric column using:

Grouped Aggregation

## Purpose
`df.groupby(Property_Type)[Price_in_Lakhs].agg(['mean', 'std', 'count'])`

## Results
- The group with the **highest mean** is identified as [Group B].  
- The group with the **same  standard deviation** is [Group A,Group B].

## 10. Preprocessing 

**ID** : Dropped because it is only a uniqe identifier and does not help predict house price.

**Price_per_SqFt** :Dropprd because it had a *moderate correltion(0.56)* with the target and was not selected for thr final model.

**Year_Built** : Year_built and Age_of_property had a perfect *negative correlation (-1.0)* and 
was not useful for improving model performance.



# Tasks 02 : *Supervised Machine Learning Model — Build, Train, and Evaluate*
## 1. Load and Define Features

The cleaned dataset (cleaned_data.csv) was loaded into a pandas DataFrame.

- *Feature Matrix (X):* All input features except the target column.
- *Regression Target (y_reg):* Price_in_Lakhs (continuous numerical value).
- *Classification Target (y_clf):* Created by converting Price_in_Lakhs into a binary target using its median value.
  - 1 → Price above the median
  - 0 → Price at or below the median

This allows the same dataset to be used for both regression and classification tasks.

## 2.Encode categorical columns

This dataset contains a mix of ordinal and nominal categorical columns describing property attributes. Ordinal columns (Public_Transport_Accessibility, Parking_Space, Security) were label-encoded using an explicit low-to-high mapping rather than sklearn.LabelEncoder, since the latter assigns codes alphabetically and would break the true rank order. Nominal columns (State, City, Facing, Owner_Type, Availability_Status, Property_Type,Furnished_Status,Owner_Type,Availability_Status) were one-hot encoded with drop_first=True, since label encoding would falsely imply a numeric distance between unrelated categories and risk misleading distance-based or linear models. High-cardinality (Locality) and multi-label (Amenities) columns were handled separately to avoid excessive dimensionality and information loss.

## 3.Leak-free train-test split and scaling:

We split the data into training and test sets before any scaling happens, using an 80/20 split with random_state=42 for reproducibility. The StandardScaler is fit only on X_train, so it learns the mean and standard deviation from the training data alone. We then use this same fitted scaler to transform both X_train and X_test, rather than fitting a new scaler on the test set. Fitting the scaler on the full dataset (train + test combined) would cause data leakage — the test set's statistics would influence the scaling parameters, giving the model indirect information about test data it should never see before evaluation.

## 4.Regression model — Linear Regression:


## Model Performance

### Linear Regression
- **MSE:** 20042.83  
- **R²:** –0.0087  

### Ridge Regression (α = 1.0)
- **MSE:** 20043.98  
- **R²:** –0.0087  

---

## Coefficient Analysis

### Largest Positive Coefficients
1. `Amenities_Playground, Pool, Garden` → +1.69  
2. `Locality_Locality_231` → +1.57  
3. `Locality_Locality_395` → +1.55  

**Interpretation:** A large positive coefficient means that as the feature value increases, the predicted target value also increases, assuming all other features remain constant.

### Largest Negative Coefficients
A large negative coefficient means that as the feature value increases, the predicted target value decreases significantly, assuming all other features remain constant.

---

### Ridge Regression Explanation

Ridge Regression adds a penalty to large coefficients, so its coefficients are usually smaller than those of ordinary least squares (OLS) Linear Regression. This helps reduce overfitting and improves model stability.  

The **alpha parameter** controls the penalty strength:  
- Higher alpha shrinks coefficients more.  
- Lower alpha makes Ridge behave more like OLS.

## 5.Classification model — Logistic Regression:
I checked class imblance in a dataset no imbalance Price_in_Lakhs
0    125001
1    124999

**I trined model and calculate metric :**

                          accuracy_score:0.49805
          f1_score:0.0
          precision_score:0.0
          recall_score:0.0
          confusion_matrix:
          [[ 9961 10039]
          [    0     0]]
          classification_report:              precision    recall  f1-score   support

                    0       1.00      0.50      0.66     20000
                    1       0.00      0.00      0.00         0

              accuracy                           0.50     20000
            macro avg       0.50      0.25      0.33     20000
          weighted avg       1.00      0.50      0.66     20000

   
**I also Compute ROU-AUC graph :**

![alt text](image-8.png)

(a) precision=Tp/Tp+Fp and recall=Tp/Tp+Fn
(b)ROC-AUC is the most important metric because it measures how wll the model  disstiguishes between houses priced above and below the median acroos all thresholds.F1 score and accuracy is laso useful bbecause it balances precision and recall.precision is not sufficent unless false positives are especially costly.
(c) The AUC value of 0.50 indicates that the model cannot effectively separate the two classes. its performance is equivalent to random guessing ,meaning it has no discriminative ability to distinguish houses above the median price from those below the median price

### b. Decision-threshold sensitivity.

The F1 score is 0.00 for all thersholds (0.30-0.7).therfore,no threshold improves the model's perfromance.the model fails to correctly identify the positive class so ther is no threshold that maximizex the F1 score.






