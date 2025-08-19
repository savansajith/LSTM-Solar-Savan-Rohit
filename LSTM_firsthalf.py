import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
warnings.filterwarnings('ignore')


df = pd.read_csv('/home/ProjB2125/Desktop/Project/Winsorised/0.1per_wins.csv')
df['date'] = pd.to_datetime(df['date'])


df['SN_winsorized']=pd.to_numeric(df['SN_winsorized'], errors='coerce')

start_date='1755-02-01'
end_date='2019-12-01'
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]


df['SN_winsorized']=pd.to_numeric(df['SN_winsorized'], errors='coerce')

sn=filtered_df['SN_winsorized'].values


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = series[:, np.newaxis]
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=132, kernel_size=4, strides=1, padding="causal", activation="relu", input_shape=[None, 1]),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.LSTM(132, return_sequences=False),
        tf.keras.layers.Dense(80, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    return model


def plt_ssn_forecast(filtered_df, full_series, future_dates, future_forecast):
    historical_df = pd.DataFrame({
        'date': filtered_df['date'],
        'sunspot_number': full_series
    })

    historical_df['ma_12'] = historical_df['sunspot_number'].rolling(window=12, center=True).mean()

    trace1 = go.Scatter(
        x=historical_df['date'],
        y=historical_df['sunspot_number'],
        mode='lines+markers',
        name='Full Sunspot Series',
        opacity=0.5
    )

    trace3 = go.Scatter(
        x=historical_df['date'],
        y=historical_df['ma_12'],
        mode='lines',
        name='12-Month Moving Average',
        line=dict(color='red')
    )

    trace2 = go.Scatter(
        x=future_dates,
        y=future_forecast,
        mode='lines+markers',
        name='Future Forecast (Next 60 months)',
        line=dict(dash='dash')
    )

    layout = go.Layout(
        title="Sunspot Forecast into the Future",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Sunspot Number"),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='black',
            borderwidth=1
        )
    )

    fig = go.Figure(data=[trace1, trace3, trace2], layout=layout)
    return fig

figure_save_path = "/home/ProjB2125/Desktop/Project/LSTM100/figures/"
csv_save_path = "/home/ProjB2125/Desktop/Project/LSTM100/csvs/"
metrics_save_path = "/home/ProjB2125/Desktop/Project/LSTM100/"

os.makedirs(figure_save_path, exist_ok=True)
os.makedirs(csv_save_path, exist_ok=True)



series = sn.astype(np.float64)


time = np.arange(len(df))
split_time = int(len(series)*0.8)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


delta = 1                      # Huber loss
window_size = 60               # For dataset
batch_size = 145               # For dataset
shuffle_buffer_size= 900       # Shuffling the dataset randomly
epochs = 100                   # For optimal learning rate
train_epochs = epochs + 100    # Training epochs
momentum_sgd = 0.9             # For optimizer    


mae_lst = []
rmse_lst = []
r2_lst = []
mape1_lst = []
mape2_lst = []


for i in range(100):
	print(f'{i+1}th Run')
	tf.keras.backend.clear_session()
	tf.random.set_seed(42)

	train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
	model = build_model()

	lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20), verbose = 0)
	optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=momentum_sgd)
	model.compile(loss=tf.keras.losses.Huber(delta),
		      optimizer=optimizer,
		      metrics=["mae"])

	history = model.fit(train_set, epochs=epochs, callbacks=[lr_schedule])
	lrs = 1e-8 * (10**(np.arange(epochs)/20))
	
	min_loss = min(history.history['loss'])
	idx_min_loss = history.history['loss'].index(min_loss)
	opt_lr = lrs[idx_min_loss]
	first = str(round(float(str(opt_lr).split('e')[0])))
	second = str(opt_lr).split('e')[-1]
	final = [first, second]
	x = "e".join(final)
	x = float(x)
	
	
	tf.keras.backend.clear_session()
	tf.random.set_seed(42)
	np.random.seed(42)
	
	
	model = build_model()
	
	optimizer = tf.keras.optimizers.SGD(learning_rate=opt_lr, momentum=momentum_sgd)
	model.compile(loss=tf.keras.losses.Huber(delta),
		      optimizer=optimizer,
		      metrics=["mae"])

	history = model.fit(train_set,epochs=train_epochs)
	
	rnn_forecast = model_forecast(model, series[:, np.newaxis], window_size)
	rnn_forecast = rnn_forecast[split_time - window_size:-1, 0]

	dates = filtered_df['date'].values
	date_train = dates[:split_time]
	date_valid = dates[split_time:]
	
	future_steps = 60
	future_forecast = []

	last_date = filtered_df['date'].iloc[-1]
	future_dates = pd.date_range(start=last_date, periods=future_steps+1, freq='M')[1:]

	current_input = series[-window_size:].reshape(1, window_size, 1)
	for _ in range(future_steps):
	    next_prediction = model.predict(current_input)[0, 0]
	    future_forecast.append(next_prediction)
	    current_input = np.append(current_input[:, 1:, :], [[[next_prediction]]], axis=1)

	
	y_true = x_valid[:len(rnn_forecast)]
	y_pred = rnn_forecast
	epsilon = 1e-5
	mask = np.abs(y_true) > epsilon	
	
	mae = mean_absolute_error(y_true, y_pred)
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	r2 = r2_score(y_true, y_pred)
	mape1 = mean_absolute_percentage_error(y_true, y_pred) * 100
	mape2 = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


	mae_lst.append(mae)
	rmse_lst.append(rmse)
	r2_lst.append(r2)
	mape1_lst.append(mape1)
	mape2_lst.append(mape2)
	
	full_series = filtered_df['SN_winsorized'].astype(float).values
	full_dates = filtered_df['date'].values
	window_size = 60
	batch_size = 145
	shuffle_buffer = 900
	future_steps = 60
	
	tf.keras.backend.clear_session()
	tf.random.set_seed(42)
	np.random.seed(42)
	
	model = build_model()
	
	opt_lr = x
	optimizer = tf.keras.optimizers.SGD(learning_rate=opt_lr, momentum=0.9)
	model.compile(loss=tf.keras.losses.Huber(1.0), optimizer=optimizer, metrics=["mae"])


	train_set_full = windowed_dataset(full_series, window_size, batch_size, shuffle_buffer)
	history = model.fit(train_set_full, epochs=epochs+100)
	
	current_input = full_series[-window_size:].reshape(1, window_size, 1)
	future_forecast = []
	
	for _ in range(future_steps):
	    next_pred = model.predict(current_input, verbose=0)[0, 0]
	    future_forecast.append(next_pred)
	    current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)
	
	
	last_date = filtered_df['date'].iloc[-1]
	future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
	
	fig = plt_ssn_forecast(filtered_df, full_series, future_dates, future_forecast)
	fig_filename = f"ssn_forecast_run{i+1}.html"
	fig.write_html(os.path.join(figure_save_path, fig_filename))


	
	forecast_df = pd.DataFrame({
	    "Date": future_dates.strftime('%Y-%m'),
	    "Forecasted Sunspot Number": np.round(future_forecast, 2)
	})
	
	forecast_df.set_index('Date')
	
	df_filename = f"forecast_csv_run_{i+1}.csv"
	forecast_df.to_csv(os.path.join(csv_save_path, df_filename), index=False)

	
	
metrics_df = pd.DataFrame({
    'MAE': mae_lst,
    'RMSE': rmse_lst,
    'R2': r2_lst,
    'MAPE_1': mape1_lst,
    'MAPE_2': mape2_lst
})

metrics_df.to_csv(os.path.join(metrics_save_path, 'metrics_summary.csv'), index=False)
	
