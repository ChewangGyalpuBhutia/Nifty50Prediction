import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

# Title of the app
st.title("Nifty 50 Stock Price Analysis")

# List of Nifty 50 tickers
nifty50_tickers = [
    "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "IOC.NS",
    "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SHREECEM.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

# User input for stock selection
selected_ticker = st.selectbox("Select a stock from Nifty 50", nifty50_tickers)

# Start date selection for historical data
start_date = st.date_input("Start date for historical data", datetime(2020, 1, 1))

# Create tabs
tab1, tab2 = st.tabs(["Forecast for the Next Year", "Predict Price for a Specific Date"])

# Fetch historical data
@st.cache_data  # Cache data to avoid reloading on every button click
def fetch_data(ticker, start_date):
    data = yf.download(ticker, start=start_date, end=datetime.now())
    data.columns = data.columns.droplevel(1)
    df = data[['Close']].reset_index()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    return df

# Tab 1: Forecast for the Next Year
with tab1:
    st.header("Forecast for the Next Year")
    
    if st.button("Generate Forecast"):
        with st.spinner(f"Fetching data and generating forecast for {selected_ticker}..."):
            # Fetch data
            df = fetch_data(selected_ticker, start_date)
            
            # Display data summary
            st.subheader("Data Summary")
            st.write(f"Historical data from {start_date} to {datetime.now().date()}")
            st.write(f"Total data points: {len(df)}")
            
            # Initialize and fit Prophet model
            model = Prophet()
            model.fit(df)
            
            # Create future dataframe for the next year
            future_year = model.make_future_dataframe(periods=180)
            
            # Make predictions for the next year
            forecast_year = model.predict(future_year)
            
            # Display forecast plot using Plotly
            st.subheader("Forecast Plot for the Next Year")
            
            # Create a Plotly figure
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['y'], mode='lines', name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add forecasted data
            fig.add_trace(go.Scatter(
                x=forecast_year['ds'], y=forecast_year['yhat'], mode='lines', name='Forecast',
                line=dict(color='orange')
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_year['ds'], y=forecast_year['yhat_upper'], fill=None, mode='lines',
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_year['ds'], y=forecast_year['yhat_lower'], fill='tonexty', mode='lines',
                line=dict(width=0), fillcolor='rgba(255, 165, 0, 0.2)', name='Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{selected_ticker} Stock Price Forecast",
                xaxis_title="Date",
                yaxis_title="Close Price (INR)",
                hovermode="x unified",
                template="plotly_white"
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display component plots
            st.subheader("Model Components")
            st.write("These plots show the trend, yearly seasonality, and weekly seasonality components of the forecast.")
            
            # Use prophet's built-in plotting functions (convert to plotly for better interactivity)
            from prophet.plot import plot_components
            import matplotlib.pyplot as plt
            
            fig2 = model.plot_components(forecast_year)
            st.pyplot(fig2)

# Tab 2: Predict Price for a Specific Date
with tab2:
    st.header("Predict Price for a Specific Date")
    
    # Specific date selection for prediction
    specific_date = st.date_input(
        "Enter a future date to predict",
        datetime.now().date() + timedelta(days=30),  # Default to 30 days in the future
        min_value=datetime.now().date() + timedelta(days=1),  # Ensure it's a future date
        max_value=datetime.now().date() + timedelta(days=365 * 5)  # Allow up to 5 years in the future
    )
    
    if st.button("Predict Price for Selected Date"):
        with st.spinner(f"Fetching data and generating prediction for {selected_ticker} on {specific_date}..."):
            # Fetch data
            df = fetch_data(selected_ticker, start_date)
            
            # Initialize and fit Prophet model
            model = Prophet()
            model.fit(df)
            
            # Create future dataframe for the selected date
            future = pd.DataFrame({'ds': [specific_date]})
            
            # Make predictions for the selected date
            forecast = model.predict(future)
            
            # Display results in a card-like format
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Price", f"₹{forecast['yhat'].values[0]:.2f}")
                
            with col2:
                st.metric("Lower Bound", f"₹{forecast['yhat_lower'].values[0]:.2f}")
                
            with col3:
                st.metric("Upper Bound", f"₹{forecast['yhat_upper'].values[0]:.2f}")
            
            # Calculate days from now and percentage change from current price
            days_from_now = (specific_date - datetime.now().date()).days
            current_price = df['y'].iloc[-1]
            predicted_price = forecast['yhat'].values[0]
            percent_change = ((predicted_price - current_price) / current_price) * 100
            
            # Display additional information
            st.write(f"Prediction for: **{specific_date}** ({days_from_now} days from today)")
            st.write(f"Current price: ₹{current_price:.2f}")
            st.write(f"Predicted change: **{percent_change:.2f}%**")
            
            # Display visual representation of current vs. predicted price
            st.subheader("Current vs. Predicted Price")
            
            # Create a simple bar chart
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=['Current Price', 'Predicted Price'],
                y=[current_price, predicted_price],
                marker_color=['blue', 'orange']
            ))
            
            fig3.update_layout(
                title=f"{selected_ticker} - Current vs. Predicted Price",
                yaxis_title="Price (INR)",
                template="plotly_white"
            )
            
            st.plotly_chart(fig3)
            
            # Display recent historical data and the prediction point
            st.subheader("Recent Historical Data with Prediction")
            
            # Show last 90 days of data with the prediction point
            recent_df = df.tail(90)
            
            fig4 = go.Figure()
            
            # Add historical data
            fig4.add_trace(go.Scatter(
                x=recent_df['ds'], y=recent_df['y'], mode='lines', name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add prediction point
            fig4.add_trace(go.Scatter(
                x=[specific_date], y=[predicted_price], mode='markers', name='Prediction',
                marker=dict(color='red', size=10)
            ))
            
            # Add confidence interval
            fig4.add_trace(go.Scatter(
                x=[specific_date, specific_date], 
                y=[forecast['yhat_lower'].values[0], forecast['yhat_upper'].values[0]],
                mode='lines', name='Confidence Interval',
                line=dict(color='green', width=2)
            ))
            
            fig4.update_layout(
                title=f"{selected_ticker} - Recent History and Prediction",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                hovermode="x unified",
                template="plotly_white"
            )
            
            st.plotly_chart(fig4)