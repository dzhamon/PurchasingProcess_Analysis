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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error




def regression_analysis_month_by_month(contract_df):
	""" -- В этой модели предсказывается общая сумма контрактов
			за следующий месяц, учитывая предыдущие 2-3 месяца
	---"""

	def feature_engineering(f_contracts, sub_folder):
		# --- Создание признаков (Feature Engineering) ---
		# 1. Агрегация по месяцам и поставщикам
		# Создадим основной DataFrame для признаков на основе недельной (месячной) агрегации
		# Это будет агрегация по МЕСЯЦУ, а не по поставщикам в столбцах

		# Проверяем, существует ли директория, и если нет, создаем ее
		if not os.path.exists(sub_folder):
			os.makedirs(sub_folder)  # os.makedirs() создает промежуточные директории

		df_features = f_contracts.groupby('contract_week').agg(
			avg_unit_price=('unit_price_eur', 'mean'),
			total_weekly_amount=('total_contract_amount_eur', 'sum'),
			weekly_contract_count=('total_contract_amount_eur', 'size'),  # Количество контрактов
			weekly_avg_contract_amount=('total_contract_amount_eur', 'mean'),  # Средняя сумма контракта
			weekly_unique_counterparties=('counterparty_name', lambda x: x.nunique())  # Количество уникальных контрагентов
		).reset_index()

		df_features['contract_week'] = df_features['contract_week'].apply(lambda x: x.start_time)

		df_features['week_number'] = np.arange(len(df_features)) # фактически это индексы временного ряда
		df_features['week'] = df_features['contract_week'].dt.isocalendar().week # номер недели в году (сезонность)
		df_features['total_weekly_amount_smooth'] = df_features['total_weekly_amount'].rolling(window=3, min_periods=1).mean()

		# df_features = df_features.set_index('contract_week')  # Устанавливаем месяц как индекс

		# Создание лаговых признаков для всех интересующих нас колонок
		features_to_lag = ['avg_unit_price', 'total_weekly_amount',
						   'weekly_contract_count', 'weekly_avg_contract_amount',
						   'weekly_unique_counterparties', 'week']
		# весь интервал разбит на недели
		num_lags = 4  # Оставляем 4 лагов

		# 2. Создание целевой переменной - суммы контрактов в будущем.
		# Целевая переменная - общая сумма за следующие недели, сглаженная
		Y = np.log1p(np.maximum(df_features['total_weekly_amount_smooth'].shift(-1), 1)).dropna()

		# Создаем DataFrame для признаков Х, добавляя лаги
		X_list = []
		for feature in features_to_lag:
			for i in range(1, num_lags + 1):
				df_features[f'{feature}_lag_{i}'] = df_features[feature].shift(i)
			X_list.extend([f'{feature}_lag_{i}' for i in range(1, num_lags + 1)])

		X_list.extend(['week_number'])

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

			Y_log = Y

			# Шаг 3: Разделение данных на обучающую и тестовую выборки
			# X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
			# TimeSeriesSplit обеспечивает последовательное разделение (без перемешивания)
			tscv = TimeSeriesSplit(n_splits=5)

			for train_index, test_index in tscv.split(X_scaled):
				X_train, X_test = X_scaled[train_index], X_scaled[test_index]
				Y_train, Y_test = Y_log.iloc[train_index], Y_log.iloc[test_index]

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
			model = RandomForestRegressor(
				n_estimators=200,
				random_state=42,
				n_jobs=-1,
				max_depth=8,
				max_features='sqrt',        # лучше sqrt для регрессии
				min_samples_leaf=5          # легкая регуляризация, чтобы не переобучался
			)

			# обучение модели
			model.fit(X_train, Y_train)
			Y_pred_log = model.predict(X_test)

			# Шаг 5: Предсказание на тестовых данных
			Y_pred = np.expm1(Y_pred_log)
			Y_test_orig = np.expm1(Y_test)

			# Шаг 6: Оценка модели
			r2_train = model.score(X_train, Y_train)
			r2_test = r2_score(Y_test_orig, Y_pred)
			mse_test = mean_squared_error(Y_test_orig, Y_pred)

			print(f"RandomForest - R^2 на обучающих данных : {r2_train:.4f}")
			print(f"RandomForest - R^2 на тестовых данных : {r2_test:.4f}")
			print(f"RandomForest - Среднеквадратичная ошибка (MSE) на тестовых данных: {mse_test:.2f}")

			# Визуализация предсказаний vs фактических значений на тестовой выборке
			plt.figure(figsize=(10, 6))
			plt.scatter(Y_test_orig, Y_pred, alpha=0.6, edgecolor='k')
			plt.plot([Y_test_orig.min(), Y_test_orig.max()],
					 [Y_test_orig.min(), Y_test_orig.max()], 'r--', lw=2)  # Диагональная линия
			plt.xlabel('Фактические значения')
			plt.ylabel('Предсказанные значения')
			plt.title(f'Факт vs Прогноз (R²={r2_test:.3f}, MSE={mse_test:,.0f})')
			plt.grid(True)
			plt.tight_layout()

			plot_path = os.path.join(sub_folder, 'predictions_vs_actual.png')
			plt.savefig(plot_path)
			plt.close()
			print(f"График сохранен в: {plot_path}")

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

			last_week_features_data = {}
			for feature_base in features_to_lag:
				for i in range(1, num_lags + 1):
					col_name =f'{feature_base}_lag_{i}'
					if len(df_features) >= 1:
						last_week_features_data[col_name] = df_features[feature_base].iloc[-i]
					else:
						last_week_features_data[col_name] = 0 # заполняем 0 если нет достаточной истории

					# Добавляем week_number
			last_week_features_data['week_number'] = len(df_features)
			last_week_features_df = pd.DataFrame([last_week_features_data],
											  columns=X.columns)  # Убедимся, что столбцы совпадают с X
			last_week_scaled = scaler.transform(last_week_features_df)

			# Предсказания в log-шкале
			predicted_log = model.predict(last_week_scaled)[0]

			# Обратное преобразование: из log  в обычные числа
			predicted_next_week_amount = np.expm1(predicted_log)
			print(
				f"\nПредсказанная общая сумма контрактов на следующую неделю (Random Forest): {predicted_next_week_amount:,.2f} EUR")

			print("\nАнализ регрессии завершен.")

		# --- Конец логирования файла ---
		sys.stdout = original_stdout  # Восстанавливаем stdout в консоль
		print(f"Отчет сохранен в: {log_file_path}")  # Этот print пойдет уже в консоль
		return

	# ================= Здесь начало основного модуля regression_analysis_month_by_month ===================

	# Определяем директорию для сохранения графиков и результатов регрессии
	REGRESSION_RESULTS_DIR = os.path.join(BASE_DIR, 'Regression_Results')
	os.makedirs(REGRESSION_RESULTS_DIR, exist_ok=True)
	
	print(f"Исходный DataFrame для регрессии. Размер: {contract_df.shape}")
	
	# Фильтруем контракты и приводим суммы к EUR
	columns_info = [('total_contract_amount', 'contract_currency', 'total_contract_amount_eur'),
					('unit_price', 'contract_currency', 'unit_price_eur')]
	converter = CurrencyConverter()
	filtered_contracts = converter.convert_multiple_columns(contract_df, columns_info=columns_info)

	# Шаг 1: Преобразуем столбец даты в datetime
	# 'errors='coerce'' заменит некорректные даты на NaT (Not a Time)
	filtered_contracts['contract_signing_date'] = pd.to_datetime(
		filtered_contracts['contract_signing_date'], errors='coerce'
	)

	# Удалим строки с некорректными датами
	filtered_contracts.dropna(subset=['contract_signing_date'], inplace=True)

	# Шаг 2: Создадим столбец для группировки по месяцу и году.
	# Мы используем .dt.to_period('M'), чтобы получить период год-неделя('Год-Месяц').
	filtered_contracts['contract_week'] = filtered_contracts['contract_signing_date'].dt.to_period('W')

	# Шаг 3: Группировка и агрегация
	# Мы считаем общую сумму контрактов в EUR и количество уникальных контрактов.

	weekly_aggregation = filtered_contracts.groupby('contract_week').agg(
		avg_unit_price=('unit_price_eur', 'mean'),
		total_amount_eur=('total_contract_amount_eur', 'sum'),
		number_of_contracts=('contract_number', 'nunique') # Уникальные номера контрактов
	).reset_index()

	# Преобразование столбца contract_month обратно в строку для удобства отображения/визуализации
	weekly_aggregation['contract_week'] = weekly_aggregation['contract_week'].astype(str)

	filtered_contracts = filtered_contracts.dropna(
		subset=['contract_week', 'counterparty_name', 'unit_price_eur', 'total_contract_amount_eur',
		        'project_name', 'discipline'])  # Добавлены project_name, discipline

	project_to_analyze = filtered_contracts['project_name'].unique() # для присвоения имен директорий

	if len(project_to_analyze) > 3:
		output_subfolder = 'Общий Анализ'
		OUT_DIR = os.path.join(REGRESSION_RESULTS_DIR, output_subfolder)
		feature_engineering(filtered_contracts, OUT_DIR)
	else:
		for project_name_to_analyze_ in project_to_analyze:
			# фильтруем основной DataFrame по наименованию проекта
			contracts_by_project = filtered_contracts[filtered_contracts['project_name']==project_name_to_analyze_].copy()
			project_filtered = contracts_by_project['project_name'].unique()
			output_subfolder = str(project_filtered[0])
			OUT_DIR = os.path.join(REGRESSION_RESULTS_DIR, output_subfolder)
			feature_engineering(contracts_by_project, OUT_DIR)





	