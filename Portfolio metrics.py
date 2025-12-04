import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# --- HELPER FUNCTIONS ---

def get_portfolio_data(tickers, period_years=5):
    """Fetches weekly historical data and calculates returns/covariance, handling data failures."""
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * period_years)

    # 1. Fetch Price Data
    # Use 'group_by=False' to ensure columns are properly labeled with both metric and ticker
    # We also explicitly specify auto_adjust=True to avoid the FutureWarning
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        interval='1wk', 
        group_by='ticker', 
        auto_adjust=True
    )
    
    # 2. Extract Adjusted Close Prices (which yfinance uses for auto_adjust=True)
    # This loop is more robust for multiple tickers than data['Adj Close']
    prices = pd.DataFrame()
    for ticker in tickers:
        try:
            # Look for the adjusted close column within the data for that ticker
            # With auto_adjust=True, yfinance should use the 'Close' column adjusted prices
            prices[ticker] = data[ticker]['Close'] 
        except KeyError:
            # Skip any ticker that failed to download or doesn't have the expected column structure
            st.warning(f"Skipping ticker {ticker}: Data not found or structure invalid.")
            continue
            
    # Drop rows where any ticker has missing data and drop tickers that were skipped
    prices = prices.dropna(axis=1, how='all').dropna(axis=0, how='any')

    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Could not find valid data for at least two tickers. Check symbols or period.")

    # Update the tickers list to only include those we successfully downloaded data for
    successful_tickers = prices.columns.tolist()

    # 3. Calculate Weekly Returns
    returns = prices.pct_change().dropna()
    
    # ... (Rest of the calculation logic remains the same) ...
    
    # Weekly Expected Returns (mu_weekly) and Annualized Expected Returns (MU)
    mu_weekly = returns.mean()
    MU = mu_weekly * 52  # Annualized Return (simple average)
    
    # Weekly Covariance Matrix
    SIGMA_weekly = returns.cov()
    
    # Annualized Covariance Matrix (Sigma)
    SIGMA_annual = SIGMA_weekly * 52
    
    # We return the successful tickers to ensure W and MU have the correct dimension
    return MU.values, SIGMA_annual.values, successful_tickers

# The rest of your Python script (calculate_risk_metrics and the Streamlit layout) stays the same!

def calculate_risk_metrics(tickers, MU, SIGMA):
    """Calculates portfolio metrics (mup, sig) and risk metrics (VaR)."""
    
    N = len(tickers)
    # Use Equal Weights (W) since the user only inputs tickers, not weights
    W = np.array([1/N] * N)

    # 1. Portfolio Expected Return (mup = W' * MU)
    mup = np.dot(W, MU)
    
    # 2. Portfolio Volatility (sig = sqrt(W' * Sigma * W))
    sigma2 = W.T @ SIGMA @ W
    sig = np.sqrt(sigma2)
    
    # 3. Risk Metric Calculation (CDF and QUANTILE)
    probloss = norm.cdf(0, loc=mup, scale=sig)
    probtarget = 1 - norm.cdf(0.20, loc=mup, scale=sig)
    p95 = norm.ppf(0.05, loc=mup, scale=sig) # 95% Annual VaR

    # Collect Results
    results = pd.DataFrame({
        "Expected Annual Return": [mup],
        "Annual Volatility": [sig],
        "P(Loss < 0)": [probloss],
        "P(Return >= 20%)": [probtarget],
        "95% Annual VaR (Worst 5%)": [p95]
    })
    
    # Apply Formatting
    def percent_format(x, decimals=2):
        return f"{x * 100:.{decimals}f}%"
    
    for col in results.columns:
        if 'P(' in col:
            results[col] = results[col].apply(lambda x: percent_format(x, 1))
        else:
            results[col] = results[col].apply(lambda x: percent_format(x, 2))

    return results

# --- STREAMLIT WEB APP LAYOUT ---

st.title("Portfolio Risk Calculator (MVN Model)")
st.write("Enter ticker symbols (separated by commas) to analyze the **Equal-Weighted** portfolio risk profile based on 5 years of historical weekly data.")

st.header("Portfolio Inputs")

# The only input field is now the Tickers
ticker_input = st.text_input(
    "Ticker Symbols (e.g., AAPL, MSFT, GOOG)", 
    "BABA, CRWD, GEV, NET, NU, BIDU, CPNG, FUTU, DOCN, VST, TSM, RBRK"
)

if st.button("Calculate Risk"):
    if not ticker_input:
        st.error("Please enter at least one ticker symbol.")
    else:
        # Clean and validate input
        tickers_list = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        if not tickers_list:
             st.error("Invalid ticker list. Please check the formatting.")
             st.stop()
        
        with st.spinner(f'Fetching data and calculating for {", ".join(tickers_list)}...'):
            try:
                # PHASE 1: Data Acquisition and Matrix Calculation
                MU, SIGMA, calculated_tickers = get_portfolio_data(tickers_list)
                
                # PHASE 2: Risk Metrics Calculation
                risk_df = calculate_risk_metrics(calculated_tickers, MU, SIGMA)

                # --- Display Results ---
                st.header("📊 Portfolio Risk Summary")
                st.dataframe(risk_df.T, use_container_width=True) # Transpose for a better look

                st.success("Calculation Complete!")

            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")