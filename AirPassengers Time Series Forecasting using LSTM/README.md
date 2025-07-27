# Deep-Learning

# AirPassengers Time Series Forecasting using LSTM

This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks to perform time series forecasting. The model is trained on the classic AirPassengers dataset, which contains monthly totals of international airline passengers from 1949 to 1960.

---

## Dataset

The dataset used is `AirPassengers.csv`, which consists of:

- **Month**: Time period from January 1949 to December 1960.
- **Passengers**: Total number of airline passengers per month.

This dataset is widely used for time series forecasting tasks due to its clear seasonality and trend characteristics.

---

## Project Files

- `AirPassengers.csv`: The input dataset file.
- `lstm_air_passengers.py`: Python script that includes data preprocessing, LSTM model building, training, and result visualization.
- `README.md`: Documentation of the project and usage instructions.

---

## Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
How to Run the Project
Clone this repository or download the files.

Ensure that AirPassengers.csv is in the same directory as lstm_air_passengers.py.

Run the Python script:

bash
Copy
Edit
python lstm_air_passengers.py
After training, the script will display a line plot comparing actual vs predicted passengers using the test dataset.

Methodology
Preprocessing:

Converted Month to datetime format and set it as index.

Normalized the passenger count using MinMaxScaler.

Created input sequences of 12 months to predict the next month's passengers.

Modeling:

Built an LSTM model with 50 units and ReLU activation.

Used Mean Squared Error (MSE) as the loss function and Adam optimizer.

Trained the model for 100 epochs.

Evaluation:

Compared predictions against actual values.

Visualized the results using a line plot.

Output
The model generates a plot with two lines:

Actual: Ground truth from the test set.

Predicted: Forecasted values from the LSTM model.

This allows you to visually assess how well the model captures seasonality and trends.
