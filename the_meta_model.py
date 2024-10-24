import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from fredapi import Fred
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SlidingWindow
from gtda.diagrams import Amplitude
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Fetch Bitcoin data from Yahoo Finance
def fetch_bitcoin_data():
    btc = yf.download("BTC-USD", start="2014-01-01")["Adj Close"]
    print(f"Fetched BTC data:\n{btc.head()}")
    
    if btc.empty:
        raise ValueError("Bitcoin data fetch failed or returned no data.")
    
    btc = btc.reset_index()
    btc['Date'] = pd.to_datetime(btc['Date']).dt.tz_localize(None)
    btc.columns = ['Date', 'BTC']
    return btc

# Fetch FRED data for Gross Debt and Treasury 10Y
def fetch_fred_data(fred_api_key):
    fred = Fred(api_key=fred_api_key)
    
    treasury_10y = fred.get_series("DGS10", observation_start="2014-01-01")
    gross_debt = fred.get_series("GFDEBTN", observation_start="2014-01-01")
    
    fred_data = pd.DataFrame({
        'Date': treasury_10y.index,
        'Treasury_10Y': treasury_10y.values,
        'Gross_Debt': gross_debt.reindex(treasury_10y.index).ffill().values
    })
    
    fred_data['Date'] = pd.to_datetime(fred_data['Date'])
    print(f"Fetched FRED data:\n{fred_data.head()}")
    return fred_data

# Calculate d2 metrics using only past data
def calculate_d2_metrics(mert, debt_col='Gross_Debt', risk_free_col='Treasury_10Y', asset_col='BTC', horizon_years=1):
    mert['vol_BTC'] = mert[asset_col].pct_change().rolling(window=30, min_periods=30).std().shift(1) * np.sqrt(252)
    A = mert[asset_col].shift(1)
    D = mert[debt_col].shift(1)
    r = mert[risk_free_col].shift(1) / 100
    T = horizon_years
    mert['btc_d2'] = (np.log(A / D) + (r - 0.5 * mert['vol_BTC'] ** 2) * T) / (mert['vol_BTC'] * np.sqrt(T))
    mert['btc_d2'] = mert['btc_d2'].fillna(0)
    return mert

# Perform TDA using past-only windows
def perform_tda_on_d2(mert, window_size=30, stride=1):
    d2_values = mert[['btc_d2']].values
    num_samples = len(d2_values)
    
    # Adjust the number of windows to match the dataset length
    d2_windows = []
    for i in range(window_size, num_samples + 1, stride):
        d2_windows.append(d2_values[i - window_size:i])
    d2_windows = np.array(d2_windows)
    
    if len(d2_windows) == 0:
        raise ValueError("Insufficient data for TDA windowing.")
    
    # Apply Vietoris-Rips persistence
    VR_persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1])
    persistence_diagrams = VR_persistence.fit_transform(d2_windows)
    
    # Compute amplitude
    amplitude = Amplitude(metric='bottleneck')
    tda_features = amplitude.fit_transform(persistence_diagrams)
    
    num_tda_features = tda_features.shape[0]
    padding_length = len(mert) - num_tda_features
    
    if padding_length < 0:
        raise ValueError("More TDA features than data points available.")
    
    # Adjust padding to match the data length
    mert['tda_feature'] = [0] * padding_length + list(tda_features.flatten()[:num_tda_features])
    print(f"Sample of TDA features:\n{mert[['tda_feature']].tail()}")
    return mert

def calculate_future_moving_average_and_deciles(mert, look_ahead_days=15, ma_window=30):
    """
    Calculate target variable based on centered returns.
    At time t=0, the target is based on the moving average from t-15 to t+15.
    """
    # Calculate the centered moving average directly
    mert['future_30d_ma'] = mert['BTC'].rolling(window=ma_window, center=True).mean()
    
    # Calculate returns relative to current price
    mert['current_price'] = mert['BTC']
    mert['future_return'] = (mert['future_30d_ma'] - mert['current_price']) / mert['current_price']
    
    # Create target classes based on future returns
    mert['BTC_decile_change'] = pd.qcut(mert['future_return'].dropna(), q=3, labels=False)
    
    # Drop rows where we don't have a complete window
    mert = mert.dropna(subset=['BTC_decile_change'])
    
    # Print alignment check
    print("\nSample alignment check (showing how the centered MA is calculated):")
    debug_df = pd.DataFrame({
        'Date': mert['Date'],
        'Current_Price': mert['BTC'],
        'MA_t-15_to_t+15': mert['future_30d_ma'],
        'Return': mert['future_return'].map('{:.2%}'.format) if not mert['future_return'].isna().all() else mert['future_return'],
        'Target_Class': mert['BTC_decile_change']
    }).head(30)
    print(debug_df.to_string(index=False))
    
    print("\nTarget class distribution:")
    print(mert['BTC_decile_change'].value_counts())
    
    return mert

# Full data preparation pipeline
def prepare_data_for_model(fred_api_key):
    btc_data = fetch_bitcoin_data()
    fred_data = fetch_fred_data(fred_api_key)
    mert = pd.merge(btc_data, fred_data, on='Date', how='inner')
    mert = calculate_d2_metrics(mert)
    mert = perform_tda_on_d2(mert)
    mert = calculate_future_moving_average_and_deciles(mert)
    mert['returns'] = mert['BTC'].pct_change().shift(1)
    mert['volatility'] = mert['returns'].rolling(window=30).std().shift(1)
    features = ['BTC', 'btc_d2', 'tda_feature', 'Treasury_10Y', 'Gross_Debt', 'returns', 'volatility']
    X = mert[features].copy()
    y = mert['BTC_decile_change'].copy()
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    return X, y


# Train model with Gaussian Process optimization
def train_xgboost_model_with_gp_tuning(X, y):
    tscv = TimeSeriesSplit(n_splits=5, gap=30)
    search_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'lambda': Real(0.01, 10.0, 'log-uniform'),  # L2 regularization
        'alpha': Real(0.01, 10.0, 'log-uniform')    # L1 regularization
    }

    xgb_model = xgb.XGBClassifier(
        random_state=42,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )

    opt = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=search_space,
        n_iter=30,
        cv=tscv,
        scoring='roc_auc_ovr',
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    opt.fit(X, y)
    print("Best Parameters:", opt.best_params_)
    print("Best AUC score:", opt.best_score_)
    y_pred = opt.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy on entire dataset: {accuracy * 100:.2f}%")
    return opt

def plot_per_class_entropy(y_true, y_proba, num_classes):
    # Initialize arrays to store entropy values and class counts
    entropies = np.zeros(num_classes)
    counts = np.zeros(num_classes)

    # Calculate entropy for each sample and assign to the respective class
    for i in range(len(y_true)):
        class_index = int(y_true[i])  # Get the true class index
        entropies[class_index] += entropy(y_proba[i])  # Add entropy for this class
        counts[class_index] += 1  # Count how many samples belong to this class

    # Avoid division by zero by using only non-zero counts
    non_zero_counts = counts != 0
    entropies[non_zero_counts] /= counts[non_zero_counts]  # Mean entropy for each class

    # Plot the entropy values per class
    plt.bar(range(num_classes), entropies)
    plt.xlabel("Class")
    plt.ylabel("Mean Entropy")
    plt.title("Mean Entropy per Class")
    plt.show()


def plot_predicted_vs_actual(y_true, y_pred, time_index=None):
    # Create a time axis if none is provided
    if time_index is None:
        time_index = np.arange(len(y_true))
    
    # Plot actual vs. predicted values over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_index, y_true, label="Actual", color='blue', marker='o', linestyle='dashed')
    plt.plot(time_index, y_pred, label="Predicted", color='red', marker='x', linestyle='solid')
    
    plt.xlabel("Time")
    plt.ylabel("Class Label")
    plt.title("Predicted vs Actual Values Over Time")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots a confusion matrix using Seaborn heatmap.
    
    Parameters:
    - y_true: Array-like of shape (n_samples,)
        True labels of the dataset.
    - y_pred: Array-like of shape (n_samples,)
        Predicted labels from the model.
    - labels: List of class labels for the confusion matrix.
    """
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    fred_api_key = 'e27aeafdd08f0e92830315de82579e94'   # Replace with your FRED API key
    X, y = prepare_data_for_model(fred_api_key)

    # Train the model using GP tuning
    best_model = train_xgboost_model_with_gp_tuning(X, y)

    # Predict probabilities and class labels for the dataset
    try:
        y_proba = best_model.predict_proba(X)
    except AttributeError as e:
        print(f"Error: {e}")
        print("Ensure the model is a classifier and can generate probabilities.")
        y_proba = None

    # Predict class labels for the dataset
    y_pred = best_model.predict(X)

    # Reset the index of `y` to ensure it's properly indexed
    y_reset = y.reset_index(drop=True)

    labels = np.unique(y)

    # Plot per-class entropy if probabilities are available
    if y_proba is not None:
        plot_per_class_entropy(y_reset, y_proba, num_classes=len(np.unique(y_reset)))
    else:
        print("Unable to plot entropy, probabilities were not computed.")

    # Plot predicted vs actual class values over time
    plot_predicted_vs_actual(y_reset, y_pred)
    # Plot confusion matrix
    plot_confusion_matrix(y, y_pred, labels)
    
