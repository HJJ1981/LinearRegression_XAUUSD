# Gold Price Prediction using Linear Regression (XAUUSD)

![Gold Price Analysis](https://img.shields.io/badge/Analysis-Gold%20Price%20Prediction-gold)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Linear%20Regression-green)
![Data Science](https://img.shields.io/badge/Data%20Science-Time%20Series-orange)

## 🔍 Project Overview

This project implements **Linear Regression models** to predict Gold (XAU/USD) price movements using 15 carefully engineered features. The analysis focuses on predicting three key price points: **Close**, **High**, and **Low** prices based on historical weekly data spanning from January 2020 to June 2025.

## 📊 Key Features

- **Multi-target Prediction**: Separate models for Close, High, and Low price prediction
- **15 Engineered Features**: Technical indicators and market features for robust prediction
- **5+ Years of Data**: Comprehensive dataset covering 2020-2025 period
- **Interactive Analysis**: Jupyter notebooks with detailed visualizations
- **Performance Metrics**: Complete model evaluation and validation

## 🗂️ Project Structure

```
LinearRegression_XAUUSD/
├── 📓 15Features_Linear_Regression_close_price_pred.ipynb  # Close price prediction model
├── 📓 15Features_Linear_Regression_high_price_pred.ipynb   # High price prediction model  
├── 📓 15Features_Linear_Regression_low_price_pred.ipynb    # Low price prediction model
├── 📄 XAUUSD_Weekly_20200105_20250629.csv                 # Raw weekly gold price data
├── 📄 price_data.csv                                      # Processed feature data
├── 🛠️ requirements.txt                                    # Python dependencies
├── 🛠️ environment.yml                                     # Conda environment setup
├── 📁 assets/                                             # Images and visualizations
└── 📝 README.md                                           # Project documentation
```

## 📈 Dataset Information

### Raw Data (XAUUSD_Weekly_20200105_20250629.csv)
- **Timeframe**: Weekly data from January 5, 2020 to June 29, 2025
- **Columns**: DATE, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD
- **Records**: 280+ weekly observations
- **Source**: Gold/USD (XAU/USD) market data

### Features Used
The model utilizes **15 engineered features** including:
- Price-based indicators (OHLC)
- Volume metrics
- Technical indicators
- Market spread analysis
- Time-series features

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- Jupyter Lab/Notebook

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/LinearRegression_XAUUSD.git
cd LinearRegression_XAUUSD

# Create and activate environment
conda env create -f environment.yml
conda activate forecasting_env

# Launch Jupyter Lab
jupyter lab
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/LinearRegression_XAUUSD.git
cd LinearRegression_XAUUSD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## 📚 Usage

1. **Data Exploration**: Start with any of the three main notebooks
2. **Feature Engineering**: Review the 15-feature creation process
3. **Model Training**: Follow the linear regression implementation
4. **Evaluation**: Analyze model performance metrics
5. **Prediction**: Use trained models for price forecasting

### Running Individual Models

```python
# Example: Close Price Prediction
jupyter notebook 15Features_Linear_Regression_close_price_pred.ipynb
```

## 🎯 Model Performance

Each notebook contains:
- **Data preprocessing** and feature engineering
- **Exploratory Data Analysis** (EDA) with visualizations
- **Linear Regression model** training and validation
- **Performance metrics**: R², RMSE, MAE
- **Prediction visualizations** and residual analysis

## 🛠️ Technical Stack

- **Data Analysis**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Deep Learning**: `tensorflow` (for advanced features)
- **Statistical Analysis**: `scipy`
- **Interactive Notebooks**: `jupyter lab`, `ipywidgets`

## 📊 Key Insights

- **Gold price volatility** patterns during major market events (2020-2025)
- **Feature importance** analysis for price prediction
- **Model accuracy** across different price targets (Close vs High vs Low)
- **Time series trends** and seasonal patterns

## 🔮 Future Enhancements

- [ ] Implement ensemble methods (Random Forest, XGBoost)
- [ ] Add real-time data integration
- [ ] Develop web dashboard for live predictions
- [ ] Include additional technical indicators
- [ ] Implement deep learning models (LSTM, GRU)
- [ ] Add sentiment analysis from financial news

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**HuJianJin**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⭐ Show Your Support

Give a ⭐️ if this project helped you learn about financial data analysis and machine learning!

## 📞 Contact

If you have any questions or suggestions, feel free to reach out:
- Email: your.email@example.com
- Project Link: [https://github.com/yourusername/LinearRegression_XAUUSD](https://github.com/yourusername/LinearRegression_XAUUSD)

---

**Disclaimer**: This project is for educational and research purposes only. It should not be used as financial advice for trading decisions. Always consult with financial professionals before making investment decisions.
