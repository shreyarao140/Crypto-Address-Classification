## Crypto-Address-Classification
This repository contains a machine learning solution for classifying cryptocurrency addresses into different types and provides a comprehensive Python-based pipeline for data collection, preprocessing, feature engineering, model selection, training, evaluation, and deployment.

## Repository Structure

The repository is organized into two main folders:

- `assignment`: Contains the main assignment code and dataset focusing on BTC, ETH, and TRX addresses using a Random Forest classifier.
- `assignment-extra`: Contains an extended dataset including more types of cryptocurrency addresses and additional models for further analysis.

### Folder: `assignment`

#### Algorithms Used:
- Random Forest Classifier

#### Cryptocurrency Address Types:
- BTC (Bitcoin)
- ETH (Ethereum)
- TRX (Tron)

#### Files:
- `crypto_addresses1.csv`: Contains addresses for BTC, ETH, and TRX.
- `code1.ipynb`: Code for training and evaluating the Random Forest model.

### Folder: `assignment-extra`

#### Algorithms Used:
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Machines (GBM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- GaussianNB
- Decision Tree Classifier

#### Cryptocurrency Address Types:
- BTC (Bitcoin)
- ETH (Ethereum)
- TRX (Tron)
- XRP (Ripple)
- LTC (Litecoin)
- ADA (Cardano)
- DOT (Polkadot)
- BSC (Binance Smart Chain)
- SOL (Solana)
- XTZ (Tezos)
- LINK (Chainlink)
- EOS (EOS.IO)
- XLM (Stellar)
- XMR (Monero)
- ZEC (Zcash)
- ALGO (Algorand)
- AVAX (Avalanche)
- VET (VeChain)

#### Files:

- `crypto_addresses.csv`: Contains addresses for BTC, ETH, TRX, XRP, LTC, ADA, DOT, BSC, SOL, XTZ, LINK, EOS, XLM, XMR, ZEC, ALGO, AVAX, and VET.
- `code_.ipynb`: Code for implementing and evaluating multiple models including Random Forest, SVM, GBM, Logistic Regression, KNN, GaussianNB and Decision Tree Classifier.


## Python-based Pipeline

This project utilizes a comprehensive Python-based pipeline for the following tasks:

1. **Data Collection**:  a diverse dataset of cryptocurrency addresses of different blockchains.
2. **Data Preprocessing**: Cleaning and transforming the dataset to make it suitable for model training.
3. **Feature Engineering**: Extracting relevant features from the cryptocurrency addresses, such as length, character frequency, and checksum validation.
4. **Model Selection**: Choosing appropriate machine learning algorithms for classification tasks.
5. **Model Training**: Training the selected models on the dataset.
6. **Model Evaluation**: Evaluating the trained models using various metrics such as accuracy, precision, recall, and confusion matrix.
7. **Model Deployment**: Deploying the trained classification model for real-time classification of cryptocurrency addresses.

### Prerequisites

- Python 3.6 or higher
- Required libraries: `pandas`, `numpy`, `sklearn`, `joblib`, `matplotlib`.


## Detailed Analysis and Insights

- **Feature Importance**: Analyzes which features are most important in distinguishing between different types of cryptocurrency addresses.
- **Model Performance**: Compares the accuracy, precision, recall, and F1-score of various models to identify the best performing classifier.
- **Confusion Matrix**: Provides insights into the types of misclassifications occurring in each model.

## Future Improvements

- **Ensemble Methods**: Combine multiple models to improve overall classification performance.


