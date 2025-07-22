# код SARIMA анализа временных рядов ( на примере контрактных данных

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pandas as pd
from utils.functions_from041124 import convert_to_eur


def arima_garch_model(contract_df):
	filtered_df = convert_to_eur(contract_df)
	df_arima = filtered_df[['contract_signing_date', 'total_contract_amount_eur']].set_index('contract_signing_date')
	
	# Создаем столбец для полугодий
	df_arima['Полугодие'] = df_arima.index.to_period('6M')
	
	# Группируем данные по полугодиям и суммируем контракты
	semiannual_data = df_arima.groupby('Полугодие')['total_contract_amount_eur'].sum()
	
	# Преобразуем в DataFrame для дальнейшей работы
	semiannual_df = pd.DataFrame({'Полугодие': semiannual_data.index, 'Сумма контрактов': semiannual_data.values})
	semiannual_df.set_index('Полугодие', inplace=True)
	
	# Построим модель ARIMA
	arima_model = ARIMA(semiannual_df['Сумма контрактов'], order=(1, 0, 1))
	arima_fit = arima_model.fit()
	
	# Получаем остатки (residuals) из ARIMA модели
	residuals = arima_fit.resid
	
	# Построим модель GARCH(1, 1)
	garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
	garch_fit = garch_model.fit(disp="off")
	
	# Выведем результаты модели GARCH
	print(garch_fit.summary())
	
	# Прогнозируем волатильность на следующие 3 периода
	forecast = garch_fit.forecast(horizon=3)
	
	# Визуализируем прогноз
	plt.figure(figsize=(10, 6))
	plt.plot(semiannual_df.index, residuals, label="Остатки ARIMA")
	plt.plot(forecast.variance.index[-3:], forecast.variance.values[-1, :], label="Прогноз волатильности GARCH",
	         color="red")
	plt.title("Прогноз волатильности с помощью GARCH")
	plt.xlabel("Полугодие")
	plt.ylabel("Остатки/Волатильность")
	plt.grid(True)
	plt.legend()
	plt.show()