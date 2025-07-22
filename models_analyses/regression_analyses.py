# Построение модели прогнозирования основного показателя
# методом Random Forest_Regressor
import pandas as pd
import numpy as np
from utils.functions import CurrencyConverter  # Убедитесь, что этот модуль корректен
import matplotlib.pyplot as plt
import statsmodels.api as sm  # Может пригодиться для статистического анализа
from sklearn.ensemble import RandomForestRegressor  # ИМПОРТИРУЕМ RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV  # GridSearchCV для подбора параметров
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.config import BASE_DIR
import os
import sys


def regression_analysis_month_by_month(contract_df):
	""" -- В этой модели предсказывается общая сумма контрактов
			за следующий месяц, учитывая предыдущие 2-3 месяца
	---"""
	# Определяем директорию для сохранения графиков и результатов регрессии
	REGRESSION_RESULTS_DIR = os.path.join(BASE_DIR, 'Regression_Results')
	os.makedirs(REGRESSION_RESULTS_DIR, exist_ok=True)
	
	print(f"Исходный DataFrame для регрессии. Размер: {contract_df.shape}")
	
	# Фильтруем контракты и приводим суммы к EUR
	columns_info = [('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')]
	converter = CurrencyConverter()
	filtered_contracts = converter.convert_multiple_columns(contract_df, columns_info=columns_info)
	filtered_contracts = filtered_contracts.dropna(
		subset=['contract_signing_month', 'counterparty_name', 'total_contract_amount_eur',
		        'project_name', 'discipline'])  # Добавлены project_name, discipline
	
	project_folder_name = filtered_contracts['project_name'].unique() # наименование директории названо именем проекта
	output_subfolder = project_folder_name
	if isinstance(output_subfolder, pd.arrays.StringArray):
		output_subfolder = str(output_subfolder[0])
	OUT_DIR = os.path.join(REGRESSION_RESULTS_DIR, output_subfolder)
	# Проверяем, существует ли директория, и если нет, создаем ее
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)  # os.makedirs() создает все промежуточные директории
	
	# --- Создание признаков (Feature Engineering) ---
	# 1. Агрегация по месяцам и поставщикам
	# Создадим основной DataFrame для признаков на основе месячной агрегации
	# Это будет агрегация по МЕСЯЦУ, а не по поставщикам в столбцах
	df_features = filtered_contracts.groupby('contract_signing_month').agg(
		total_monthly_amount=('total_contract_amount_eur', 'sum'),
		monthly_contract_count=('total_contract_amount_eur', 'size'),  # Количество контрактов
		monthly_avg_contract_amount=('total_contract_amount_eur', 'mean'),  # Средняя сумма контракта
		monthly_unique_counterparties=('counterparty_name', lambda x: x.nunique())  # Количество уникальных контрагентов
	).reset_index()
	
	df_features = df_features.set_index('contract_signing_month')  # Устанавливаем месяц как индекс
	
	# 2. Создание целевой переменной - суммы контрактов в следующем месяце.
	# Целевая переменная - общая сумма за следующий месяц
	Y = df_features['total_monthly_amount'].shift(-1).dropna()
	
	# Создание лаговых признаков для всех интересующих нас колонок
	features_to_lag = ['total_monthly_amount', 'monthly_contract_count', 'monthly_avg_contract_amount',
	                   'monthly_unique_counterparties']
	num_lags = 2  # Оставляем 2 лага
	
	# Создаем DataFrame для признаков Х, добавляя лаги
	X_list = []
	for feature in features_to_lag:
		for i in range(1, num_lags + 1):
			df_features[f'{feature}_lag_{i}'] = df_features[feature].shift(i)
		X_list.extend([f'{feature}_lag_{i}' for i in range(1, num_lags + 1)])
		
	# Х - теперь будет содержать только лаговые признаки
	X = df_features[X_list].dropna()
	
	# Убедитесь, что X и y имеют одинаковое количество строк
	common_index = X.index.intersection(Y.index)
	X = X.loc[common_index]
	Y = Y.loc[common_index]
	
	# --- начало логирования в файл ---
	log_file_path = os.path.join(OUT_DIR, 'analysis_log.log')
	# Сохраняем текущий stdout, чтобы потом его восстановить
	original_stdout = sys.stdout
	
	# открываем файл для записи и перенаправляем stdout
	with open(log_file_path, 'w') as f:
		sys.stdout = f # весь последующий print(), будет идти в этот файл
		
	
		if X.empty or Y.empty:
			print(
				"ВНИМАНИЕ: X или y пусты после формирования признаков/целевой переменной. Невозможно выполнить регрессию.")
			return
		
		print(f"Размер X: {X.shape}, Размер Y: {Y.shape}")
		print(f"Количество признаков (столбцов в X): {X.shape[1]}")
		print(f"Количество наблюдений (строк в X): {X.shape[0]}")
		
		# Шаг 2: Масштабирование признаков
		# Для RandomForestRegressor
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		
		# Шаг 3: Разделение данных на обучающую и тестовую выборки
		X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
		
		print(f"Количество наблюдений в обучающей выборке: {X_train.shape[0]}")
		print(f"Количество наблюдений в тестовой выборке: {X_test.shape[0]}")
		
		# Шаг 4: Применение RandomForestRegressor
		print("\n--- Обучение RandomForestRegressor ---")
		# n_estimators: количество деревьев в лесу. Чем больше, тем лучше, но дольше.
		# random_state: для воспроизводимости результатов.
		# n_jobs=-1: использовать все доступные ядра процессора для ускорения обучения.
		# max_features: количество признаков для рассмотрения на каждом расщеплении (автоматически sqrt(n_features) для регрессии).
		# min_samples_leaf: минимальное количество образцов в листе дерева.
		# max_depth: максимальная глубина каждого дерева.
		# При малом количестве данных, возможно, стоит сделать n_estimators меньше (например, 50-100)
		# и/или увеличить min_samples_leaf, чтобы деревья не были слишком сложными.
		model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,
		                              max_features=1.0,  # Можно попробовать 'sqrt' или float < 1.0
		                              min_samples_leaf=1)  # По умолчанию 1. Можно увеличить до 2-5 для большей регуляризации
		model.fit(X_train, Y_train)
		
		# Шаг 5: Предсказание на тестовых данных
		Y_pred = model.predict(X_test)
		
		# Шаг 6: Оценка модели
		r2_train = model.score(X_train, Y_train)
		r2_test = model.score(X_test, Y_test)
		mse_test = mean_squared_error(Y_test, Y_pred)
		
		print(f"RandomForest - R^2 на обучающих данных : {r2_train:.4f}")
		print(f"RandomForest - R^2 на тестовых данных : {r2_train:.4f}")
		print(f"RandomForest - Среднеквадратичная ошибка (MSE) на тестовых данных: {mse_test:.2f}")
		
		# Визуализация предсказаний vs фактических значений на тестовой выборке
		plt.figure(figsize=(12, 6))
		plt.scatter(Y_test, Y_pred, alpha=0.6)
		plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)  # Диагональная линия
		plt.xlabel('Фактические значения')
		plt.ylabel('Предсказанные значения')
		plt.title('Предсказанные vs Фактические значения на тестовой выборке')
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(os.path.join(OUT_DIR, 'predictions_vs_actual.png'))
		plt.close()
		
		# Получаем важность признаков для RandomForest
		if len(X.columns) > 0:
			feature_importances_rf = pd.DataFrame({
				'Feature': X.columns,
				'Importance': model.feature_importances_
			}).sort_values(by='Importance', ascending=False)
			print("\nВажность признаков (Random Forest):\n",
			      feature_importances_rf.to_string(index=False))
		else:
			print("Нет признаков для отображения важности")
		
		# --- Прогнозирование будущего ---
		if not X.empty:
			last_month_features_data = {}
			for feature_base in features_to_lag:
				for i in range(1, num_lags + 1):
					col_name =f'{feature_base}.lag_{i}'
					if len(df_features) >= 1:
						last_month_features_data[col_name] = df_features[feature_base].iloc[-1]
					else:
						last_month_features_data[col_name] = 0 # заполняем 0 если нет достаточной истории
			
			last_month_features_df = pd.DataFrame([last_month_features_data],
			                                      columns=X.columns)  # Убедимся, что столбцы совпадают с X
			last_month_scaled = scaler.transform(last_month_features_df)
			
			predicted_next_month_amount = model.predict(last_month_scaled)
			print(
				f"\nПредсказанная общая сумма контрактов на следующий месяц (Random Forest): {predicted_next_month_amount[0]:,.2f} EUR")
		
		print("\nАнализ регрессии завершен.")
	
	# --- Конец логирования файла ---
	sys.stdout = original_stdout  # Восстанавливаем stdout в консоль
	print(f"Отчет сохранен в: {log_file_path}")  # Этот print пойдет уже в консоль

	