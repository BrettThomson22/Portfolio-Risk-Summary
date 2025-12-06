import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import t, norm
from sklearn.covariance import LedoitWolf

# ---------------------------
# Utility & Sampling Helpers
# ---------------------------

def fetch_prices_resilient(tickers, start, end):
    """
    Fetch weekly prices for tickers, fall back to daily resample if needed.
    Corrects JSE (.JO) tickers by dividing prices by 100 (cents to Rands).
    Returns prices DataFrame of weekly 'Close' prices.
    """
    tickers = list(tickers)
    
    def process_prices(data_raw, tickers_list, interval):
        prices = pd.DataFrame()
        for tkr in tickers_list:
            series = None
            
            # Extract 'Close' price series
            if len(tickers_list) == 1 and 'Close' in data_raw.columns:
                series = data_raw['Close'].dropna()
            elif tkr in data_raw.columns and 'Close' in data_raw[tkr].columns:
                series = data_raw[tkr]['Close'].dropna()

            if series is not None:
                # --- JSE Correction Logic ---
                if tkr.upper().endswith(".JO"):
                    # Prices for JSE are often given in cents, divide by 100 to get Rands
                    series = series / 100
                # --- End Correction Logic ---
                
                if interval == '1d':
                    # Resample daily to weekly (Friday close)
                    weekly = series.resample('W-FRI').last().dropna()
                    prices[tkr] = weekly
                else:
                    prices[tkr] = series
        return prices

    # 1. Try weekly fetch (1wk)
    data_w = None
    try:
        data_w = yf.download(tickers, start=start, end=end, interval='1wk',
                             group_by='ticker', auto_adjust=True, threads=True)
    except Exception:
        pass

    prices = pd.DataFrame()
    if isinstance(data_w, pd.DataFrame) and not data_w.empty:
        prices = process_prices(data_w, tickers, '1wk')

    # 2. If insufficient rows, try daily (1d) + resample
    min_weeks = int(((end - start).days / 7) * 0.6)
    if prices.shape[0] < min_weeks or prices.shape[1] < len(tickers):
        data_d = None
        try:
            data_d = yf.download(tickers, start=start, end=end, interval='1d',
                                 group_by='ticker', auto_adjust=True, threads=True)
        except Exception:
            pass

        if isinstance(data_d, pd.DataFrame) and not data_d.empty:
            prices = process_prices(data_d, tickers, '1d')

    # Final cleanup: drop any remaining rows or columns with all NaNs
    prices = prices.dropna(axis=1, how='all').dropna(axis=0, how='any')
    return prices

def multivariate_t_rvs(mu, Sigma, df, n):
    dim = len(mu)
    v = np.random.chisquare(df, size=n)
    z = np.random.multivariate_normal(np.zeros(dim), Sigma, size=n)
    scaling = np.sqrt(df / v).reshape(-1, 1)
    return mu + z * scaling

def is_cov_ok(Sigma):
    try:
        vals = np.linalg.eigvalsh(Sigma)
        return np.all(vals > 1e-8)
    except Exception:
        return False

# ---------------------------
# Portfolio metrics & Simulation
# ---------------------------

def portfolio_metrics_and_simulation(returns_weekly_df, W, use_t=True, df=5, vol_multiplier=1.0,
                                     do_sim=False, n_sims=2000, horizon_weeks=52, seed=None):
    mu_w = returns_weekly_df.mean().values
    Sigma_w = returns_weekly_df.cov().values

    if vol_multiplier != 1.0:
        Sigma_w = Sigma_w * (vol_multiplier ** 2)

    mup_w = float(W @ mu_w)
    sigma2_w = float(W.T @ Sigma_w @ W)
    sigma_w = np.sqrt(max(sigma2_w, 0.0))

    mup_a = mup_w * 52
    sigma_a = sigma_w * np.sqrt(52)

    if use_t:
        q = t.ppf(0.05, df=df)
        dist = lambda z: t.cdf(z, df=df)
    else:
        q = norm.ppf(0.05)
        dist = norm.cdf

    z_a_loss = (0 - mup_a) / sigma_a if sigma_a > 0 else -np.inf
    prob_loss_a = dist(z_a_loss)
    z_a_target = (0.20 - mup_a) / sigma_a if sigma_a > 0 else np.inf
    prob_target_a = 1 - dist(z_a_target)

    var_annual = -(mup_a + sigma_a * q)
    var_annual = max(var_annual, 0.0)

    metrics = {
        "mup_a": mup_a, "sigma_a": sigma_a,
        "prob_loss_a": prob_loss_a,
        "prob_target_a": prob_target_a,
        "VaR95_annual": var_annual,
        "df": df if use_t else "N/A (Normal)"
    }

    sim_results = {}
    if do_sim:
        if seed is not None and seed != 0:
            np.random.seed(int(seed))
            
        max_dds = []
        annual_rets = []
        dim = len(W)
        horizon_weeks = int(horizon_weeks)
        batch = 200
        total = int(n_sims)
        
        for start in range(0, total, batch):
            run = min(batch, total - start)
            for _ in range(run):
                if use_t and df > 0:
                    samples = multivariate_t_rvs(mu_w, Sigma_w, df, horizon_weeks)
                else:
                    samples = np.random.multivariate_normal(mu_w, Sigma_w, size=horizon_weeks)
                port_rets = samples @ W
                wealth = np.cumprod(1 + port_rets)
                running_max = np.maximum.accumulate(wealth)
                drawdowns = (running_max - wealth) / running_max
                max_dds.append(float(drawdowns.max() if len(drawdowns) > 0 else 0.0))
                annual_rets.append(float(wealth[-1] - 1.0))

        sim_results['max_drawdowns'] = np.array(max_dds)
        sim_results['annual_rets'] = np.array(annual_rets)

        if len(annual_rets) > 0:
            sim_annual = np.array(annual_rets)
            sim_prob_loss = float((sim_annual < 0).mean())
            sim_prob_target = float((sim_annual >= 0.20).mean())
            ci_low, median_dd, ci_high = np.percentile(sim_results['max_drawdowns'], [2.5, 50, 97.5])
            sim_results.update({
                'sim_prob_loss': sim_prob_loss,
                'sim_prob_target': sim_prob_target,
                'dd_ci_low': ci_low, 'dd_ci_high': ci_high, 'median_dd': median_dd
            })
    return metrics, sim_results

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(layout="wide", page_title="Portfolio Risk — Robust Model")
st.title("Portfolio Risk Calculator — Robust Model (JSE Price Corrected)")

st.markdown("""
Advanced calculator with:
- Ledoit–Wolf covariance shrinkage
- Multivariate t Monte Carlo simulation
- **FIXED:** Automatically divides prices by 100 for JSE tickers (`.JO`) to correct for cents-based reporting, providing accurate metrics.
""")

with st.sidebar:
    st.header("Model Settings")
    period_years = st.number_input("Historical Lookback (years)", min_value=1, max_value=20, value=5)
    use_t = st.checkbox("Use Student's t (MV-t) for MC & Parametrics", value=True)
    df_slider = st.slider("Degrees of freedom (ν) for T-dist", min_value=3, max_value=50, value=5)
    vol_multiplier = st.slider("Volatility Stress Multiplier (1.0 = normal)", min_value=0.8, max_value=3.0, value=1.0, step=0.05)
    st.markdown("---")
    do_simulation = st.checkbox("Run Monte Carlo Simulations (Drawdown & Empirical Risk)", value=True)
    n_sims = st.number_input("Simulation Runs", min_value=500, max_value=20000, value=5000, step=500)
    seed = st.number_input("Random Seed (0 = random)", min_value=0, value=0, step=1)

st.header("Inputs")
ticker_input = st.text_input("Tickers (comma separated). Use .JO for JSE tickers.", value="AAPL, MSFT")
weights_input = st.text_input("Optional Weights (comma separated, sum to 1). Leave empty = equal weight.", value="0.50,0.50")

if st.button("Calculate Risk"):
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if len(tickers) < 1:
        st.error("Please enter at least one ticker.")
        st.stop()

    if weights_input.strip():
        try:
            ws = [float(x.strip()) for x in weights_input.split(",") if x.strip()]
            if len(ws) != len(tickers):
                st.error("Number of weights must equal number of tickers.")
                st.stop()
            W = np.array(ws) / np.sum(ws)
        except Exception as e:
            st.error(f"Could not parse weights: {e}")
            st.stop()
    else:
        W = np.array([1.0 / len(tickers)] * len(tickers))

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * period_years)
    with st.spinner("Fetching price data resiliently..."):
        try:
            prices = fetch_prices_resilient(tickers, start_date, end_date)
        except Exception as e:
            st.error(f"Error fetching price data: {e}")
            st.stop()

    if prices.empty or prices.shape[1] == 0:
        st.error("No valid price series found. Check tickers.")
        st.stop()

    tickers_found = prices.columns.tolist()
    if set(tickers_found) != set(tickers):
        st.warning(f"Some tickers were not found and removed: {set(tickers) - set(tickers_found)}")
        original_tickers = tickers
        tickers = tickers_found
        W_aligned = np.zeros(len(tickers))
        for i, tkr in enumerate(tickers):
            try:
                # Find the original weight based on the original ticker list index
                # This ensures the weights array stays aligned with the fetched prices
                original_index = [i for i, t in enumerate(original_tickers) if t == tkr][0]
                W_aligned[i] = W[original_index]
            except IndexError:
                # Fallback if alignment fails (shouldn't happen with .index() approach)
                pass 
        W = W_aligned / np.sum(W_aligned)

    returns_weekly = prices.pct_change().dropna()
    st.info(f"Historical weeks available: {returns_weekly.shape[0]}")

    # Covariance stabilization (Ledoit-Wolf)
    Sigma_w = returns_weekly.cov().values
    applied_shrink = False
    if not is_cov_ok(Sigma_w):
        st.warning("Empirical covariance matrix is ill-conditioned. Applying Ledoit-Wolf shrinkage.")
        try:
            lw = LedoitWolf().fit(returns_weekly.values)
            Sigma_w = lw.covariance_
            applied_shrink = True
        except Exception:
            # Fallback to regularization if Ledoit-Wolf fails
            Sigma_w = Sigma_w + np.eye(Sigma_w.shape[0]) * 1e-6
            applied_shrink = True

    with st.spinner("Calculating portfolio metrics and Monte Carlo..."):
        metrics, sim_results = portfolio_metrics_and_simulation(
            returns_weekly_df=returns_weekly,
            W=W,
            use_t=use_t,
            df=df_slider,
            vol_multiplier=vol_multiplier,
            do_sim=do_simulation,
            n_sims=n_sims,
            horizon_weeks=52,
            seed=seed if seed != 0 else None
        )

    def pct(x):
        try:
            if isinstance(x, (float, np.float64, np.float32)):
                return f"{float(x) * 100:.2f}%"
            return str(x)
        except Exception:
            return str(x)

    st.header("Portfolio Summary (Annualized)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Inputs")
        st.metric("Tickers Used", ", ".join(tickers))
        st.metric("Weights", ", ".join([f"{w:.3f}" for w in W]))
        st.metric("Historical Weeks", returns_weekly.shape[0])
        st.metric("Degrees of Freedom (ν)", metrics['df'])
    with col2:
        st.subheader("Parametric Risk Metrics")
        st.metric("Expected Annual Return", pct(metrics['mup_a']))
        st.metric("Annual Volatility", pct(metrics['sigma_a']))
        st.metric("P(Loss in 1yr)", pct(metrics['prob_loss_a']))
        st.metric("P(Return ≥ 20% in 1yr)", pct(metrics['prob_target_a']))
        st.metric("95% VaR (1yr)", pct(metrics['VaR95_annual']))

    if do_simulation and sim_results:
        st.header("Monte Carlo (Empirical) Results — 1 Year")
        st.write(f"Simulations run: {len(sim_results.get('annual_rets', []))}")
        mc_col1, mc_col2 = st.columns(2)
        if 'sim_prob_loss' in sim_results:
            with mc_col1:
                st.metric("Empirical P(Loss in 1yr)", pct(sim_results['sim_prob_loss']))
                st.metric("Empirical P(Return ≥ 20% in 1yr)", pct(sim_results['sim_prob_target']))
            with mc_col2:
                st.metric("Median Max Drawdown (1yr)", pct(sim_results['median_dd']))
                st.metric("95% CI Max Drawdown (1yr)", f"[{pct(sim_results['dd_ci_low'])}, {pct(sim_results['dd_ci_high'])}]")
            st.subheader("Max Drawdown Distribution (Simulated)")
            dd_ser = pd.Series(sim_results['max_drawdowns'])
            st.bar_chart(dd_ser.value_counts(bins=40).sort_index())

    if applied_shrink:
        st.success("Ledoit-Wolf shrinkage applied to stabilize the covariance matrix.")
    st.success("Calculation complete.")





