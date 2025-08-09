import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Feature Engineering
def feature_engineering(df):
    df['Temp Avg'] = (df['Temp Max'] + df['Temp Min']) / 2
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    return df

# Train Model
def train_model(X_train, y_train, model_name):
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit App
def main():
    st.title("🌡️ Temperature Forecasting with Ensemble Models")

    uploaded_file = st.file_uploader("📁 Upload a CSV file (Date, Rain, Temp Max, Temp Min)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Clean up Rain column
        df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce').fillna(0)

        # Drop rows with missing values
        df = df.dropna(subset=['Date', 'Temp Max', 'Temp Min', 'Rain'])

        # Feature engineering
        df = feature_engineering(df)
        df = df.sort_values("Date")

        # Final check for any NaNs in important fields
        df = df.dropna(subset=["Temp Avg"])

        # Model features and target
        features = ['year', 'month', 'day', 'Rain']
        target = 'Temp Avg'

        # Train-test split
        split_index = int(0.8 * len(df))
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        model_name = st.selectbox("🤖 Choose a model", ["Random Forest", "Gradient Boosting", "XGBoost"])
        model = train_model(X_train, y_train, model_name)

        predictions = model.predict(X_test)

        # Check for NaNs
        if np.isnan(predictions).any() or np.isnan(y_test.values).any():
            st.error("❌ Error: NaNs found in predicted or actual values. Please clean your dataset.")
            return

        mae = mean_absolute_error(y_test, predictions)
        st.success(f"📉 Mean Absolute Error (MAE): {mae:.2f}")

        # Plot actual vs predicted
        st.write("### 📊 Actual vs Predicted Temperatures")
        fig, ax = plt.subplots()
        ax.plot(test_df['Date'], y_test.values, label="Actual", marker='o')
        ax.plot(test_df['Date'], predictions, label="Predicted", marker='x')
        ax.legend()
        st.pyplot(fig)

        # Feature Importance
        st.write("### 📌 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'feature': features, 'importance': importance})
            st.bar_chart(importance_df.set_index('feature'))
        else:
            st.info("Feature importance not available for this model.")

if __name__ == "__main__":
    main()
