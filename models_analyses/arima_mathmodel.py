"""
Этапы использования метода ARIMA для прогнозирования временного ряда
"""
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from utils.functions_from041124 import convert_to_eur
"""
Для начала проверяется стационарность
данных с помощью теста Дики_Фуллера
"""
def diki_fuller_arima(filtered_df):
	filtered_df = convert_to_eur(filtered_df)  # приводим все суммы в единую валюту EUR
	
	# Предположим, что у нас есть filtered_df
	df_arima = filtered_df[['contract_signing_date', 'total_contract_amount_eur']].set_index('contract_signing_date')
	
	# Выполним тест Дики-Фуллера для проверки стационарности
	result = adfuller(df_arima['total_contract_amount_eur'])
	
	# Выводим результат
	print('ADF Statistic:', result[0])
	print('p-value:', result[1])
	
	# Интерпретация
	if result[1] < 0.05:
		print("Данные стационарны (отклоняем нулевую гипотезу)")
	else:
		print("Данные не стационарны (не отклоняем нулевую гипотезу)")

	""" Построение графиков ACF и PACF"""
	
	import matplotlib.pyplot as plt
	from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
	
	# Построим графики ACF и PACF для оригинальных данных
	fig, ax = plt.subplots(2, figsize=(12, 8))
	
	# ACF
	plot_acf(df_arima['total_contract_amount_eur'], lags=20, ax=ax[0])
	ax[0].set_title('Автокорреляционная функция (ACF)')
	
	# PACF
	plot_pacf(df_arima['total_contract_amount_eur'], lags=20, ax=ax[1])
	ax[1].set_title('Частичная автокорреляционная функция (PACF)')
	
	plt.tight_layout()
	plt.show()
	
	""" построение и обучение модели ARIMA при p=1, d=0, q=1"""
	from statsmodels.tsa.arima.model import ARIMA
	
	# Построим модель ARIMA (p=1, d=0, q=1)
	model_arima = ARIMA(df_arima['total_contract_amount_eur'], order=(1, 0, 1))
	model_fit = model_arima.fit()
	
	# Выведем краткое описание модели
	print(model_fit.summary())
	
	""" Прогнозирование """
	# Прогнозируем на 12 периодов вперед
	forecast_steps = 12
	forecast = model_fit.get_forecast(steps=forecast_steps)
	
	# Получим доверительные интервалы
	forecast_df = forecast.conf_int()
	forecast_df['forecast'] = forecast.predicted_mean
	
	# Создадим индекс для прогноза
	forecast_index = pd.date_range(start=df_arima.index[-1], periods=forecast_steps + 1, freq='M')[1:]
	
	# Соединим индекс с данными прогноза
	forecast_df.index = forecast_index
	
	# Построим график
	plt.figure(figsize=(12, 6))
	plt.plot(df_arima.index, df_arima['total_contract_amount_eur'], label='Исторические данные')
	plt.plot(forecast_df.index, forecast_df['forecast'], label='Прогноз ARIMA')
	plt.fill_between(forecast_df.index, forecast_df['lower total_contract_amount_eur'],
	                 forecast_df['upper total_contract_amount_eur'], color='k', alpha=0.15)
	plt.legend()
	plt.title('Прогноз сумм контрактов с помощью ARIMA')
	plt.xlabel('Дата')
	plt.ylabel('Сумма контрактов (EUR)')
	plt.show()