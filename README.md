# Smart Event-Driven Stock Prediction System

An intelligent stock prediction platform that combines market data, financial news, social media sentiment, and machine learning to forecast short-term stock movements and generate BUY, SELL, and HOLD signals.

## Features

- Automated collection of market, news, and social media data
- MongoDB storage for scraped financial documents
- Financial sentiment analysis with FinBERT
- 28 technical and sentiment-based features
- Five-model prediction ensemble:
  - Random Forest
  - XGBoost
  - LightGBM
  - Logistic Regression
  - LSTM neural network
- Confidence-based ensemble signals
- Multi-step LSTM price forecasting
- GARCH(1,1) volatility estimation
- Monte Carlo simulations with VaR and Expected Shortfall
- SHAP explainability and LSTM saliency analysis
- Walk-forward backtesting on chronological hold-out data
- Interactive React dashboard powered by FastAPI

## Architecture

1. Financial data and documents are collected from market, news, and social media sources.
2. Documents are stored in MongoDB and scored with FinBERT.
3. Technical indicators and sentiment statistics are merged into the training dataset.
4. The trained models generate individual market predictions.
5. The ensemble combines model outputs into a final BUY, SELL, or HOLD signal.
6. Forecasting, risk analysis, explainability, and backtesting results are exposed through the API and displayed in the React dashboard.

## Methodology

The dataset uses a chronological 80/20 train-test split for each ticker. LSTM sequences are also created separately within each ticker. This prevents future information and cross-ticker history from leaking into training and provides an honest out-of-sample evaluation.

## Project Structure

```text
backend/       FastAPI backend and API endpoints
frontend/      React dashboard
src/           Data processing, scraping, training, prediction, and backtesting
models/        Local trained model artifacts (not included in Git)
data/          Local raw and processed datasets (not included in Git)
results/       Evaluation results and prediction outputs
thesis/        Dissertation chapters and figures
```

## Tech Stack

Python, FastAPI, React, MongoDB, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, Transformers (FinBERT), SHAP, GARCH, and Monte Carlo simulation.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the frontend dependencies:

```bash
cd frontend
npm install
```

Make sure MongoDB is running locally before starting the data and sentiment pipelines.

## Running the Application

Start the FastAPI backend from the project root:

```bash
uvicorn backend.main:app --reload
```

Start the React frontend in a separate terminal:

```bash
cd frontend
npm run dev
```

The dashboard is then available at the local URL shown by Vite.

## Main API Endpoints

- `GET /predict/{ticker}` - price forecast, model signals, and risk metrics
- `GET /details/{ticker}` - model accuracy, confidence, and feature factors
- `GET /signals/{ticker}` - recent source-level sentiment signals
- `GET /backtest/{ticker}` - out-of-sample walk-forward backtest
- `GET /shap/{ticker}` - SHAP-based feature contributions
- `POST /refresh/{ticker}` - refresh market data

## Notes

Local datasets, logs, virtual environments, credentials, and trained model artifacts are excluded through `.gitignore`. Model files and processed data must be generated locally before running live predictions.

This project is intended for research and educational purposes and does not constitute financial advice.
