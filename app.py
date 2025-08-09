import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Feature Engineering
def feature_engineering(df):
    df['Temp Avg'] = (df['Temp Max'] + df['Temp Min']) / 2
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
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
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model

def main():
    st.title("🌡️ Temperature Forecasting with Ensemble Models")

    uploaded_file = st.file_uploader("📁 Upload a CSV file (Date, Rain, Temp Max, Temp Min)", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        return

    df = pd.read_csv(uploaded_file)

    # Clean & preprocess
    df = feature_engineering(df)
    df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Date', 'Temp Max', 'Temp Min', 'Rain', 'Temp Avg'])
    df = df.sort_values("Date")

    # Sidebar Filters
    st.sidebar.header("🔎 Filter Options")

    years = sorted(df['year'].unique())
    selected_year = st.sidebar.selectbox("Select Year", options=["All"] + years, index=0)

    month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
                   7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
    months = list(month_names.values())
    selected_month = st.sidebar.selectbox("Select Month (Optional)", options=["All"] + months, index=0)

    summary_type = st.sidebar.radio("Summary Type", ["Overall", "Yearly", "Monthly"])

    # Filter data
    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    if selected_month != "All":
        # Map month name back to number
        month_num = {v:k for k,v in month_names.items()}[selected_month]
        filtered_df = filtered_df[filtered_df['month'] == month_num]

    # Show summary
    if summary_type == "Yearly":
        summary = filtered_df.groupby("year").agg({
            "Temp Avg": "mean",
            "Rain": "sum"
        }).reset_index()
        st.write("📅 **Yearly Summary**")
        st.dataframe(summary)
        st.line_chart(summary.set_index("year")["Temp Avg"])

    elif summary_type == "Monthly":
        summary = filtered_df.groupby("month").agg({
            "Temp Avg": "mean",
            "Rain": "sum"
        }).reset_index()
        summary["Month"] = summary["month"].map(month_names)
        summary = summary.sort_values("month")  # Important: sort by month number to keep calendar order
        st.write("📅 **Monthly Summary**")
        st.dataframe(summary[["Month", "Temp Avg", "Rain"]])
        st.line_chart(summary.set_index("Month")["Temp Avg"])

    else:
        temp_avg = filtered_df["Temp Avg"].mean()
        rain_total = filtered_df["Rain"].sum()
        st.write("📊 **Overall Summary**")
        st.metric("Average Temperature", f"{temp_avg:.2f}°C")
        st.metric("Total Rainfall", f"{rain_total:.2f} mm")

    # Forecasting model
    st.divider()
    st.subheader("📈 Forecasting Model")

    if len(filtered_df) < 10:
        st.warning("Not enough data for model training. Try a different filter or upload more data.")
        return

    features = ['year', 'month', 'day', 'Rain']
    target = 'Temp Avg'

    split_index = int(0.8 * len(filtered_df))
    train_df = filtered_df.iloc[:split_index]
    test_df = filtered_df.iloc[split_index:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    model_name = st.selectbox("🤖 Choose a model", ["Random Forest", "Gradient Boosting", "XGBoost"])
    model = train_model(X_train, y_train, model_name)

    predictions = model.predict(X_test)

    # Check NaNs
    if np.isnan(predictions).any() or np.isnan(y_test.values).any():
        st.error("❌ NaNs found in predicted or actual values. Please clean your dataset.")
        return

    mae = mean_absolute_error(y_test, predictions)
    st.success(f"📉 Mean Absolute Error (MAE): {mae:.2f}")

    # Plot Actual vs Predicted with month names on x-axis in calendar order
    fig, ax = plt.subplots()
    ax.plot(test_df['Date'], y_test.values, label="Actual", marker='o')
    ax.plot(test_df['Date'], predictions, label="Predicted", marker='x')

    # Month locator and formatter for proper calendar month names on x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    ax.legend()
    st.pyplot(fig)

    # Feature importance
    st.write("### 📌 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'feature': features, 'importance': importance})
        st.bar_chart(importance_df.set_index('feature'))
    else:
        st.info("Feature importance not available for this model.")

if __name__ == "__main__":
    main()
