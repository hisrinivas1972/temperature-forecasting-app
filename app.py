import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

def feature_engineering(df):
    df['Temp Avg'] = (df['Temp Max'] + df['Temp Min']) / 2
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    return df

def train_model(X_train, y_train, model_name):
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    st.title("🌡️ Temperature Forecasting with Ensemble Models")

    uploaded_file = st.file_uploader("Upload a CSV file (with Date, Rain, Temp Max, Temp Min)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        st.write("### Raw Data", df.head())

        df = feature_engineering(df)
        df = df.sort_values("Date")

        features = ['year', 'month', 'day', 'Rain']
        target = 'Temp Avg'

        split_index = int(0.8 * len(df))
        train_df = df[:split_index]
        test_df = df[split_index:]

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        model_name = st.selectbox("Choose a model", ["Random Forest", "Gradient Boosting", "XGBoost"])
        model = train_model(X_train, y_train, model_name)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        st.write(f"📉 Mean Absolute Error (MAE): {mae:.2f}")

        st.write("### Actual vs Predicted Temperatures")
        fig, ax = plt.subplots()
        ax.plot(test_df['Date'], y_test.values, label="Actual", marker='o')
        ax.plot(test_df['Date'], predictions, label="Predicted", marker='x')
        ax.legend()
        st.pyplot(fig)

        st.write("### Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'feature': features, 'importance': importance})
            st.bar_chart(importance_df.set_index('feature'))
        else:
            st.info("Feature importance not available for this model.")

if __name__ == "__main__":
    main()
