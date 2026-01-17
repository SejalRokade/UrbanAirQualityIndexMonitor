# Urban Air Quality Index Monitor

A Streamlit-based web application for monitoring and analyzing urban air quality data.

## Features

- Real-time air quality monitoring
- Multiple pollutant tracking (PM2.5, PM10, NO2, O3, SO2, CO)
- Interactive visualizations
- Health impact assessment
- Time series analysis
- Pollutant distribution analysis

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)
3. Enter a city name and select the number of days to display
4. Explore the different tabs for various visualizations and analyses

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Dependencies
- Python 3.7+
- Streamlit
- Folium
- Pandas
- NumPy
- scikit-learn
- Joblib
- Requests
- OpenWeatherMap API

  
## Note

This application currently uses mock data for demonstration purposes. To use real air quality data, you would need to:

1. Sign up for an air quality API service (e.g., OpenWeatherMap, AirVisual)
2. Create a `.env` file with your API key
3. Modify the `get_air_quality_data()` function in `app.py` to fetch real data

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request



