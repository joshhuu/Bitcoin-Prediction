# Bitcoin Price Prediction using LSTM and GRU

This project uses a **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** model to predict the future price of Bitcoin based on historical data. The model incorporates technical indicators like **SMA (Simple Moving Average)**, **EMA (Exponential Moving Average)**, and **RSI (Relative Strength Index)**, which are commonly used in financial market analysis.

### Project Overview
- **Goal**: Predict Bitcoin prices using historical data and deep learning models (LSTM, GRU).
- **Input**: Historical Bitcoin price data with features like **Price**, **SMA**, **EMA**, and **RSI**.
- **Output**: Predicts the future Bitcoin prices for the next day or week.

---

### Key Features:
- **Preprocessing**: The data is cleaned and normalized using MinMaxScaler to scale the features.
- **Technical Indicators**: The model uses **SMA**, **EMA**, and **RSI** as additional features to improve prediction accuracy.
- **Model**: A combination of **LSTM** and **GRU** layers is used to capture both long-term and short-term dependencies in the data.
- **Regularization**: Dropout layers are added to prevent overfitting.
- **Hyperparameter Tuning**: Learning rate scheduling is implemented to adjust the learning rate when the validation loss plateaus.

---

### Installation

To run this project locally, follow the steps below.

#### Prerequisites:
1. **Python 3.6+**.
2. **Install required libraries**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
   ```

---

### File Structure:
- **`Bitcoin_Prediction.ipynb`**: Jupyter notebook containing the full implementation of the model, including data preprocessing, model building, and prediction.
- **`bitcoin.csv`**: Sample historical Bitcoin price data (ensure this is present in the project directory).

---

### Steps to Run:

1. **Data Preprocessing**:
   The dataset is loaded and the **Date** column is converted to datetime format. The following technical indicators are calculated:
   - **Simple Moving Average (SMA)**
   - **Exponential Moving Average (EMA)**
   - **Relative Strength Index (RSI)**

2. **Scaling Data**:
   The dataset is normalized using **MinMaxScaler** to scale the features to the range [0, 1].

3. **Model Construction**:
   The model consists of:
   - **LSTM layer**: For capturing long-term dependencies.
   - **GRU layer**: For capturing short-term dependencies.
   - **Dropout layers**: For regularization to avoid overfitting.

4. **Model Training**:
   The model is trained using **Adam optimizer** with a **mean squared error loss function**. A **learning rate scheduler** is used to adjust the learning rate if the validation loss plateaus.

5. **Prediction**:
   Once the model is trained, predictions are made on the test set and evaluated using **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.

6. **Future Predictions**:
   The trained model can be used to predict Bitcoin prices for the **next day** or the **next 7 days** based on the last 60 days of data.

---

### Example Usage:

```python
# Example to predict the next day's Bitcoin price based on the last 60 days
last_60_days = scaled_data[-60:]
last_60_days = last_60_days.reshape(1, 60, 4)

# Predict the next day's price
next_day_price = model.predict(last_60_days)
next_day_price = scaler.inverse_transform(np.column_stack((next_day_price, np.zeros((next_day_price.shape[0], 3)))))[:, 0]
print(f'The predicted price for the next day is: {next_day_price[0]}')
```

---

### Evaluation:
The model is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the squared average of the prediction errors.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute differences between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the MSE to give an error metric in the same unit as the data.

---

### Example Output:
After training and predicting, the following example results might appear:

```
Mean Squared Error: 167251.36
Mean Absolute Error: 327.62
Root Mean Squared Error: 408.96
```

---

### Test Cases:
1. **Next Day Price Prediction**: Predict the price for the next day based on the last 60 days of data.
2. **7-Day Forecast**: Predict Bitcoin prices for the next 7 days.
3. **Model Evaluation on Test Set**: Evaluate the model’s performance on unseen test data.
4. **Major Price Movements**: Predict major surges or drops in Bitcoin prices.
5. **Out-of-Sample Prediction**: Test the model on future, unseen data.

---

### Conclusion:
This project demonstrates the use of **LSTM** and **GRU** models to predict Bitcoin prices. By incorporating technical indicators and regularization techniques, the model can make reasonably accurate predictions for short-term and medium-term forecasting. Further improvements can be made by adding more features or experimenting with different architectures.

---

Feel free to clone this repository, modify the dataset, and experiment with other configurations to enhance the model's performance.

---
