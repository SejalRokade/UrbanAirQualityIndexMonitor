import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Set page configuration
st.set_page_config(
    page_title="Urban Air Quality Index",
    page_icon="🌫️",
    layout="wide"
)

# Title and description
st.title("🌫️ Urban Air Quality Index Monitor")
st.markdown("""
This application provides real-time air quality monitoring, analysis, and prediction for urban areas.
Track various air quality parameters and their impact on health using real data from OpenWeatherMap API.
""")

# Function to prepare features for ML model
def prepare_features(df):
    df = df.copy()
    
    # Time-based features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lag features (previous days' values)
    pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
    for pollutant in pollutants:
        df[f'{pollutant}_lag1'] = df[pollutant].shift(1)
        df[f'{pollutant}_lag2'] = df[pollutant].shift(2)
        df[f'{pollutant}_lag3'] = df[pollutant].shift(3)
        
        # Rolling statistics
        df[f'{pollutant}_rolling_mean_7d'] = df[pollutant].rolling(window=7, min_periods=1).mean()
        df[f'{pollutant}_rolling_std_7d'] = df[pollutant].rolling(window=7, min_periods=1).std()
    
    # Handle missing values properly
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def evaluate_predictions(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

def predict_air_quality(df, forecast_days):
    predictions = {}
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=1), 
                                periods=forecast_days, freq='D')
    
    # Prepare features for ML models
    df_features = prepare_features(df)
    pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
    feature_cols = [col for col in df_features.columns if col not in ['Date'] + pollutants]
    
    model_evaluations = []
    
    for pollutant in pollutants:
        series = df[pollutant].values
        
        # Initialize predictions array for ensemble
        all_predictions = np.zeros((3, forecast_days))  # 3 models
        
        # 1. Holt-Winters Prediction
        if len(df) >= 14:
            try:
                hw_model = ExponentialSmoothing(
                    series,
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add',
                    initialization_method='estimated',
                    use_boxcox=True
                )
                hw_fitted = hw_model.fit(optimized=True, remove_bias=True)
                all_predictions[0] = hw_fitted.forecast(forecast_days)
            except:
                # Fallback to simple exponential smoothing
                ses_model = SimpleExpSmoothing(series)
                ses_fitted = ses_model.fit(optimized=True)
                all_predictions[0] = ses_fitted.forecast(forecast_days)
        
        # 2. HistGradientBoosting Prediction
        gb_model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        # Prepare features for ML models
        X = df_features[feature_cols].values
        y = df_features[pollutant].values
        
        # Train-test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        model_evaluations.append(evaluate_predictions(y_test, y_pred_gb, f'GradientBoosting_{pollutant}'))
        
        # Prepare future features
        future_features = pd.DataFrame(index=future_dates)
        future_features['day_of_week'] = future_features.index.dayofweek
        future_features['month'] = future_features.index.month
        future_features['day'] = future_features.index.day
        future_features['week_of_year'] = future_features.index.isocalendar().week
        future_features['is_weekend'] = future_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Initialize all pollutant-related columns
        for p in pollutants:
            # Initialize lag features
            last_values = df[p].iloc[-3:].values
            for i, lag in enumerate(['lag1', 'lag2', 'lag3']):
                future_features[f'{p}_{lag}'] = last_values[-(i+1)]
            
            # Initialize rolling statistics
            future_features[f'{p}_rolling_mean_7d'] = df[p].iloc[-7:].mean()
            future_features[f'{p}_rolling_std_7d'] = df[p].iloc[-7:].std()
        
        # Ensure all feature columns exist and are in the correct order
        missing_cols = set(feature_cols) - set(future_features.columns)
        for col in missing_cols:
            future_features[col] = 0  # Initialize missing columns with 0
            
        # Reorder columns to match training data
        future_features = future_features[feature_cols]
        
        # Make GB predictions
        all_predictions[1] = gb_model.predict(future_features.values)
        
        # 3. Linear Regression for trend
        lr_model = LinearRegression()
        X_trend = np.arange(len(series)).reshape(-1, 1)
        lr_model.fit(X_trend, series)
        
        X_future_trend = np.arange(len(series), len(series) + forecast_days).reshape(-1, 1)
        all_predictions[2] = lr_model.predict(X_future_trend)
        
        # Ensemble: Weighted average of predictions
        weights = np.array([0.5, 0.3, 0.2])  # HW, GB, LR weights
        weighted_predictions = np.average(all_predictions, axis=0, weights=weights)
        
        # Ensure predictions stay positive and within realistic bounds
        weighted_predictions = np.clip(weighted_predictions, 0, df[pollutant].max() * 1.5)
        predictions[pollutant] = weighted_predictions
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(predictions, index=future_dates)
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Store model evaluations for display
    st.session_state['model_evaluations'] = pd.DataFrame(model_evaluations)
    
    return forecast_df

# Cache the API response to avoid hitting rate limits
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_location_coordinates(city):
    """Get latitude and longitude for a city using OpenWeatherMap Geocoding API"""
    geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    try:
        response = requests.get(geocoding_url)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            st.error(f"Could not find coordinates for {city}")
            return None, None
    except Exception as e:
        st.error(f"Error getting coordinates: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_air_quality_data(city, days):
    """Fetch air quality data from OpenWeatherMap API"""
    lat, lon = get_location_coordinates(city)
    if lat is None or lon is None:
        return None

    # Initialize empty lists to store data
    all_data = []
    current_time = datetime.now()

    # Get historical data
    for day in range(days):
        timestamp = int((current_time - timedelta(days=day)).timestamp())
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={timestamp-3600}&end={timestamp}&appid={API_KEY}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'list' in data:
                for item in data['list']:
                    components = item['components']
                    date = datetime.fromtimestamp(item['dt'])
                    
                    all_data.append({
                        'Date': date,
                        'PM2.5': components.get('pm2_5', 0),
                        'PM10': components.get('pm10', 0),
                        'NO2': components.get('no2', 0),
                        'O3': components.get('o3', 0),
                        'SO2': components.get('so2', 0),
                        'CO': components.get('co', 0)
                    })
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.2)
            
        except Exception as e:
            st.warning(f"Error fetching data for {datetime.fromtimestamp(timestamp)}: {str(e)}")
            continue

    if not all_data:
        st.error("No data could be retrieved. Using mock data as fallback.")
        return generate_mock_data(days)

    # Convert to DataFrame and sort by date
    df = pd.DataFrame(all_data)
    df = df.sort_values('Date')
    
    # Remove duplicates and handle missing values properly
    df = df.drop_duplicates(subset=['Date'])
    
    # Use proper methods for handling missing values
    df = df.set_index('Date')
    df = df.resample('D').mean()
    df = df.ffill().bfill()  # Updated from deprecated method
    df = df.reset_index()
    
    return df

def generate_mock_data(days):
    """Generate mock data as fallback when API fails"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    t = np.arange(days)
    data = {
        'Date': dates,
        'PM2.5': 20 + 5 * np.sin(2 * np.pi * t / 7) + 0.1 * t + np.random.normal(0, 2, days),
        'PM10': 30 + 8 * np.sin(2 * np.pi * t / 7) + 0.15 * t + np.random.normal(0, 3, days),
        'NO2': 40 + 10 * np.sin(2 * np.pi * t / 7) + 0.2 * t + np.random.normal(0, 4, days),
        'O3': 50 + 15 * np.sin(2 * np.pi * t / 7) + 0.25 * t + np.random.normal(0, 5, days),
        'SO2': 10 + 3 * np.sin(2 * np.pi * t / 7) + 0.05 * t + np.random.normal(0, 1, days),
        'CO': 5 + 2 * np.sin(2 * np.pi * t / 7) + 0.03 * t + np.random.normal(0, 0.5, days)
    }
    return pd.DataFrame(data)

# Sidebar for user inputs
st.sidebar.header("Settings")
city = st.sidebar.text_input("Enter City Name", "London")
days = st.sidebar.slider("Historical Days to Display", 14, 30, 14)  # Limited to 30 days due to API constraints
forecast_days = st.sidebar.slider("Days to Forecast", 1, 7, 3)

# Add a map to show the selected location
if city:
    lat, lon = get_location_coordinates(city)
    if lat and lon:
        st.sidebar.write("### Selected Location")
        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker([lat, lon], popup=city).add_to(m)
        with st.sidebar:
            folium_static(m, width=300, height=200)

# Get the data
with st.spinner('Fetching air quality data...'):
    df = get_air_quality_data(city, days)

if df is not None:
    # Get predictions
    forecast_df = predict_air_quality(df, forecast_days)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³", 
                  f"{df['PM2.5'].iloc[-1] - df['PM2.5'].iloc[0]:.1f} µg/m³")
    with col2:
        st.metric("PM10", f"{df['PM10'].mean():.1f} µg/m³", 
                  f"{df['PM10'].iloc[-1] - df['PM10'].iloc[0]:.1f} µg/m³")
    with col3:
        st.metric("NO2", f"{df['NO2'].mean():.1f} µg/m³", 
                  f"{df['NO2'].iloc[-1] - df['NO2'].iloc[0]:.1f} µg/m³")
    with col4:
        st.metric("O3", f"{df['O3'].mean():.1f} µg/m³", 
                  f"{df['O3'].iloc[-1] - df['O3'].iloc[0]:.1f} µg/m³")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series & Predictions", "Pollutant Distribution", "Health Impact", "Forecast Details"])

    with tab1:
        st.subheader("Air Quality Trends and Predictions")
        
        # Combine historical and forecast data
        combined_df = pd.concat([df, forecast_df])
        
        # Create figure with both historical and predicted values
        fig = go.Figure()
        
        pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
        colors = px.colors.qualitative.Set1
        
        for idx, pollutant in enumerate(pollutants):
            # Historical values
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[pollutant],
                name=f"{pollutant} (Historical)",
                line=dict(color=colors[idx])
            ))
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[pollutant],
                name=f"{pollutant} (Predicted)",
                line=dict(color=colors[idx], dash='dash')
            ))
        
        fig.update_layout(
            title=f"Air Quality Parameters Over Time in {city} (with {forecast_days}-day Forecast)",
            xaxis_title="Date",
            yaxis_title="Concentration (µg/m³)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Pollutant Distribution")
        fig = px.box(df, y=['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO'],
                     title=f"Distribution of Air Pollutants in {city}")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Health Impact Assessment")
        
        # Calculate Air Quality Index (AQI)
        def calculate_aqi(pm25, pm10, no2, o3, so2, co):
            # Simplified AQI calculation
            return (pm25 + pm10 + no2 + o3 + so2 + co) / 6
        
        df['AQI'] = df.apply(lambda row: calculate_aqi(
            row['PM2.5'], row['PM10'], row['NO2'], 
            row['O3'], row['SO2'], row['CO']
        ), axis=1)
        
        forecast_df['AQI'] = forecast_df.apply(lambda row: calculate_aqi(
            row['PM2.5'], row['PM10'], row['NO2'], 
            row['O3'], row['SO2'], row['CO']
        ), axis=1)
        
        # Health impact assessment
        def get_health_impact(aqi):
            if aqi < 50:
                return "Good", "Minimal impact"
            elif aqi < 100:
                return "Moderate", "Acceptable quality"
            elif aqi < 150:
                return "Unhealthy for Sensitive Groups", "Increased respiratory symptoms"
            elif aqi < 200:
                return "Unhealthy", "Increased respiratory effects"
            elif aqi < 300:
                return "Very Unhealthy", "Significant health effects"
            else:
                return "Hazardous", "Serious health effects"
        
        current_aqi = df['AQI'].iloc[-1]
        predicted_aqi = forecast_df['AQI'].mean()
        health_level, health_impact = get_health_impact(current_aqi)
        future_health_level, future_health_impact = get_health_impact(predicted_aqi)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current AQI", f"{current_aqi:.1f}", health_level)
            st.info(f"Current Health Impact: {health_impact}")
        
        with col2:
            st.metric("Predicted AQI", f"{predicted_aqi:.1f}", future_health_level)
            st.info(f"Predicted Health Impact: {future_health_impact}")
        
        # AQI trend with forecast
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['AQI'],
            name="Historical AQI"
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['AQI'],
            name="Predicted AQI",
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title=f"Air Quality Index Trend in {city} (with {forecast_days}-day Forecast)",
            xaxis_title="Date",
            yaxis_title="AQI"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Detailed Forecast")
        
        # Display forecast data in a table
        st.write("Predicted Air Quality Values:")
        forecast_display = forecast_df.copy()
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(forecast_display.round(2), use_container_width=True)
        
        # Display model evaluation metrics
        st.write("### Model Performance Metrics")
        if 'model_evaluations' in st.session_state:
            metrics_df = st.session_state['model_evaluations']
            st.dataframe(metrics_df.round(4), use_container_width=True)
        
        st.write("### Forecast Methodology")
        st.info("""
        The predictions are made using an ensemble of three models:
        1. Holt-Winters Exponential Smoothing (50% weight)
           - Captures seasonal patterns and trends
           - Best for short-term predictions
        
        2. Gradient Boosting Regressor (30% weight)
           - Considers multiple features including:
             - Day of week, month, and seasonal patterns
             - Previous days' values (lag features)
             - Rolling statistics (7-day averages and standard deviations)
        
        3. Linear Regression (20% weight)
           - Captures long-term trends
        
        The final prediction is a weighted average of these models, optimized for both
        short-term accuracy and long-term trend prediction.
        
        Factors considered in the prediction:
        - Historical patterns and seasonality
        - Day-of-week effects (weekday vs weekend)
        - Recent trends and volatility
        - Long-term trend direction
        """)

# Footer
st.markdown("---")
st.markdown("""
### About
This application provides air quality monitoring, analysis, and prediction using real data from OpenWeatherMap API. 
The predictions are made using an ensemble of models including Holt-Winters exponential smoothing, 
Gradient Boosting, and Linear Regression to provide accurate forecasts.
""") 