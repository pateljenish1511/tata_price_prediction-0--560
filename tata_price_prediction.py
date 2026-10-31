# ============================================================
# TATA USED CAR PRICE PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("Project/tata_used_cars.csv")

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# DATA QUALITY REPORT
# ============================================================

print("\nDATA QUALITY REPORT")

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Values:")
print(df.duplicated().sum())

# ============================================================
# DATA CLEANING
# ============================================================

df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

current_year = 2026

df['car_age'] = current_year - df['year']

# Remove ID column

if 'car_id' in df.columns:
    df.drop('car_id', axis=1, inplace=True)

# Brand is constant (Tata)

if 'brand' in df.columns:
    if df['brand'].nunique() == 1:
        df.drop('brand', axis=1, inplace=True)

# ============================================================
# SWEETVIZ REPORT
# ============================================================

print("\nGenerating Sweetviz Report...")

report = sv.analyze(df)

report.show_html(
    "Tata_Car_EDA_Report.html",
    open_browser=True
)

print("Sweetviz Report Generated")

# ============================================================
# DATA UNDERSTANDING
# ============================================================

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nInfo")
print(df.info())

print("\nStatistical Summary")
print(df.describe(include='all'))

# ============================================================
# NUMERICAL & CATEGORICAL
# ============================================================

numerical_cols = df.select_dtypes(
    include=np.number
).columns

categorical_cols = df.select_dtypes(
    exclude=np.number
).columns

print("\nNumerical Columns")
print(numerical_cols)

print("\nCategorical Columns")
print(categorical_cols)

# ============================================================
# HISTOGRAMS
# ============================================================

for col in numerical_cols:

    plt.figure(figsize=(6,4))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(f"Histogram - {col}")

    plt.show()

# ============================================================
# BOXPLOTS
# ============================================================

for col in numerical_cols:

    plt.figure(figsize=(6,4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot - {col}")

    plt.show()

# ============================================================
# BAR CHARTS
# ============================================================

for col in categorical_cols:

    plt.figure(figsize=(8,4))

    df[col].value_counts().plot(
        kind='bar'
    )

    plt.title(f"Bar Chart - {col}")

    plt.show()

# ============================================================
# PIE CHARTS
# ============================================================

for col in categorical_cols:

    plt.figure(figsize=(6,6))

    df[col].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%'
    )

    plt.ylabel("")

    plt.title(f"Pie Chart - {col}")

    plt.show()

# ============================================================
# OUTLIER ANALYSIS
# ============================================================

print("\nOUTLIER ANALYSIS")

for col in numerical_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"{col} : {len(outliers)}")

# ============================================================
# CORRELATION MATRIX
# ============================================================

corr_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# MODEL BUILDING
# ============================================================

target = "selling_price_inr"

X = df.drop(target, axis=1)

y = df[target]

categorical_features = X.select_dtypes(
    include='object'
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorical_features
        )
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape")
print(X_train.shape)

print("\nTesting Shape")
print(X_test.shape)

# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

print("\nMODEL EVALUATION")

print("R2 Score :", r2)
print("MSE      :", mse)
print("RMSE     :", rmse)

# ============================================================
# CROSS VALIDATION
# ============================================================

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("\nCross Validation Scores")
print(scores)

print(
    "\nAverage CV Score:",
    scores.mean()
)

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print(result.head(20))

# ============================================================
# SCATTER PLOT
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Price")

plt.show()

# ============================================================
# SAVE RESULTS
# ============================================================

result.to_csv(
    "Tata_Car_Predictions.csv",
    index=False
)

joblib.dump(
    model,
    "tata_used_car_model.pkl"
)

print("\nModel Saved Successfully")
print("Prediction File Saved Successfully")

# ============================================================
# USER INPUT CAR PRICE PREDICTION
# ============================================================
'''
print("\n" + "="*60)
print("TATA USED CAR PRICE PREDICTION")
print("="*60)

model_name = input("Enter Model: ")
variant = input("Enter Variant: ")
fuel_type = input("Enter Fuel Type: ")
transmission = input("Enter Transmission: ")
owner = input("Enter Owner Type: ")
city = input("Enter City: ")

year = int(input("Enter Manufacturing Year: "))
km_driven = float(input("Enter KM Driven: "))
seats = int(input("Enter Number of Seats: "))

car_age = 2026 - year

# Create dataframe using SAME columns used in training

new_car = pd.DataFrame({

    "model": [model_name],
    "variant": [variant],
    "year": [year],
    "fuel_type": [fuel_type],
    "transmission": [transmission],
    "owner": [owner],
    "km_driven": [km_driven],
    "seats": [seats],
    "city": [city],
    "car_age": [car_age]

})

# Predict price

predicted_price = model.predict(new_car)

print("\n" + "="*60)
print("PREDICTED CAR PRICE")
print("="*60)

print(
    f"Estimated Price : ₹{predicted_price[0]:,.2f}"
)

print("\nPROJECT COMPLETED SUCCESSFULLY")
'''