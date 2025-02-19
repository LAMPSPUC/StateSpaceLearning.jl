import pandas as pd
import math
import statsmodels.api as sm
import numpy as np
from prophet import Prophet
from chronos import ChronosPipeline
import torch

df_train1 = pd.read_csv("paper_tests/m4_test/Monthly-train1.csv")
df_train2 = pd.read_csv("paper_tests/m4_test/Monthly-train2.csv")
df_train3 = pd.read_csv("paper_tests/m4_test/Monthly-train3.csv")
df_train4 = pd.read_csv("paper_tests/m4_test/Monthly-train4.csv")
df_train = pd.concat([df_train1, df_train2, df_train3, df_train4])
m4_info = pd.read_csv("paper_tests/m4_test/M4-info.csv")

df_test = pd.read_csv("paper_tests/m4_test/Monthly-test.csv")
ssl_init_df = pd.read_csv("paper_tests/m4_test/init_SSL/SSL_aic_0.1_false.csv")

dict_vec = []
for i in range(0, 48000):
    train_raw = df_train.iloc[i, 1:].tolist()
    index_of_nan = next((i for i, x in enumerate(train_raw) if math.isnan(x)), None)
    train = train_raw[:index_of_nan]
    test = df_test.iloc[i, 1:].tolist()
    max_train = max(train)
    min_train = min(train)
    normalized_train = [(x - min_train) / (max_train - min_train) for x in train]
    dict = {"train": train, "test": test, "normalized_train": normalized_train, "max": max_train, "min": min_train}
    dict_vec.append(dict)

def sMAPE(y_test, prediction):
    H = len(y_test)
    denominator = [abs(y) + abs(p) for y, p in zip(y_test, prediction)]
    return (200 / H) * sum([abs(y - p) / d if d != 0 else 0 for y, p, d in zip(y_test, prediction, denominator)])

def MASE(y_train, y_test, prediction, m=12):
    T = len(y_train)
    H = len(y_test)
    numerator = (1 / H) * sum([abs(y - p) for y, p in zip(y_test, prediction)])
    denominator = (1 / (T - m)) * sum([abs(y_train[j] - y_train[j - m]) for j in range(m, T)])
    return numerator / denominator # if denominator != 0 else 0

def evaluate_ss(input, sample_size, init, hyperparameters_inicialization):
    train = input["train"]
    test = input["test"]
    normalized_train = input["normalized_train"][-sample_size:]
    max_train = input["max"]
    min_train = input["min"]
    model_components = {'irregular': True, 'level': True, 'trend': True, 'freq_seasonal': [{'period': 12}], 
                                    'stochastic_level': True, 'stochastic_trend': True, 'stochastic_freq_seasonal': [True]}
    model = sm.tsa.UnobservedComponents(np.array(normalized_train), **model_components)
    if init:
        results = model.fit(start_params = hyperparameters_inicialization, disp = False, maxiter = 1e5)
    else:
        results = model.fit(disp = False, maxiter = 1e5)
    forecast = results.get_forecast(steps=18)
    normalized_forecast_values = forecast.predicted_mean
    forecast_values = [x * (max_train - min_train) + min_train for x in normalized_forecast_values]
    return sMAPE(test, forecast_values), MASE(train, test, forecast_values)


results      = []
results_init = []
for i in range(0, 48000):
    hyperparameters_inicialization = [ssl_init_df.loc[i]["ϵ"], ssl_init_df.loc[i]["ξ"],ssl_init_df.loc[i]["ζ"],ssl_init_df.loc[i]["ω_12"]]
    results.append(evaluate_ss(dict_vec[i], 2794, False, hyperparameters_inicialization))
    results_init.append(evaluate_ss(dict_vec[i], 2794, True, hyperparameters_inicialization))

smape_SS = []
mase_SS = []
smape_SS_init = []
mase_SS_init = []
for i in range(0, len(results)):
    smape_SS.append(results[i][0])
    mase_SS.append(results[i][1])
    smape_SS_init.append(results_init[i][0])
    mase_SS_init.append(results_init[i][1])

#create dataframe with mase and smape columns:
df = pd.DataFrame({'smape': smape_SS, 'mase': mase_SS})
df_init = pd.DataFrame({'smape': smape_SS_init, 'mase': mase_SS_init})
#save to csv:
df.to_csv('paper_tests/m4_test/results_SS/SS.csv')
df_init.to_csv('paper_tests/m4_test/results_SS/SS_init.csv')

#save to csv dataframe with mean of metrics:
NAIVE_sMAPE = 14.427 #M4 Paper
NAIVE_MASE = 1.063   #M4 Paper
df_mean = pd.DataFrame({'smape': [np.mean(smape_SS)], 'mase': [np.mean(mase_SS)], 'owa': [(np.mean(smape_SS) / NAIVE_sMAPE + np.mean(mase_SS) / NAIVE_MASE) / 2]})
df_mean_init = pd.DataFrame({'smape': [np.mean(smape_SS_init)], 'mase': [np.mean(mase_SS_init)], 'owa': [(np.mean(smape_SS_init) / NAIVE_sMAPE + np.mean(mase_SS_init) / NAIVE_MASE) / 2]})

df_mean.to_csv('paper_tests/m4_test/metrics_results/SS_METRICS_RESULTS.csv')
df_mean_init.to_csv('paper_tests/m4_test/metrics_results/SS_INIT_METRICS_RESULTS.csv')

def evaluate_prophet(input):
    train = input["train"]
    test = input["test"]
    timestamps = pd.date_range(start="2020-01-01", periods=len(train), freq='ME')
    #add random seed 
    df = pd.DataFrame({
        'ds': timestamps,
        'y': train
    })
    model = Prophet(interval_width=0.95)
    model.fit(df)
    future = pd.DataFrame({
        'ds': (pd.date_range(start="2020-01-01", periods=len(train) + 18, freq='ME'))[len(train):]
    })
    model_forecast = model.predict(future)
    prediction = model_forecast['yhat'].values
    return sMAPE(test, prediction), MASE(train, test, prediction)

def evaluate_chronos(input):
    train = input["train"]
    test = input["test"]
    chronos_forecast = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-large",
            device_map="mps"
        ).predict(
            context=torch.tensor(train),
            prediction_length=18,
            limit_prediction_length=False
        )
    prediction = np.quantile(chronos_forecast[0].numpy(), [0.5], axis=0)[0]
    return sMAPE(test, prediction), MASE(train, test, prediction)

smape_prophet_vec = []
mase_prophet_vec = []
smape_chronos_vec = []
mase_chronos_vec = []
for i in range(0, len(dict_vec)):
    smape_prophet, mase_prophet = evaluate_prophet(dict_vec[i])
    smape_prophet_vec.append(smape_prophet)
    mase_prophet_vec.append(mase_prophet)
    smape_chronos, mase_chronos = evaluate_chronos(dict_vec[i])
    smape_chronos_vec.append(smape_chronos)
    mase_chronos_vec.append(mase_chronos)
    # 
    print("Runningg series ", i)
    if i % 1000 == 0:
        print("Runningg series ", i)
        smape_mean_prophet = np.mean(smape_prophet_vec)
        smape_emean_chronos = np.mean(smape_chronos_vec)
        mase_mean_prophet = np.mean(mase_prophet_vec)
        mase_mean_chronos = np.mean(mase_chronos_vec)
        print("Mean sMape Prophet: ", smape_mean_prophet)
        print("Mean sMape Chronos: ", smape_emean_chronos)
        print("Mean Mase Prophet: ", mase_mean_prophet)
        print("Mean Mase Chronos: ", mase_mean_chronos)


NAIVE_sMAPE = 14.427 #M4 Paper
NAIVE_MASE = 1.063   #M4 Paper

owa_prophet = (np.mean(smape_prophet_vec) / NAIVE_sMAPE + np.mean(mase_prophet_vec) / NAIVE_MASE) / 2
owa_chronos = (np.mean(smape_chronos_vec) / NAIVE_sMAPE + np.mean(mase_chronos_vec) / NAIVE_MASE) / 2

mean_mase_prophet = np.mean(mase_prophet_vec)
mean_smape_prophet = np.mean(smape_prophet_vec)
mean_mase_chronos = np.mean(mase_chronos_vec)
mean_smape_chronos = np.mean(smape_chronos_vec)

df_results_mean = pd.DataFrame({'smape': [mean_smape_prophet, mean_smape_chronos], 'mase': [mean_mase_prophet, mean_mase_chronos], 'owa': [owa_prophet, owa_chronos]})

# save to csv

df_results_mean.to_csv('paper_tests/m4_test/metrics_results/PROPHET_CHRONOS_METRICS_RESULTS.csv')