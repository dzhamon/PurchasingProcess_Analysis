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

	import numpy as np
	import pandas as pd
	import os
	import sys
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split

	def feature_engineering(f_contracts, sub_folder):
		"""
        Исправленная версия с правильной последовательностью операций
        """

		# ===== 1. СОЗДАНИЕ ПРИЗНАКОВ =====
		from models_analyses.create_new_features import create_enhanced_features
		data_dict = create_enhanced_features(f_contracts, sub_folder)

		if data_dict is None or data_dict['X'].shape[0] < 15:
			print("Недостаточно данных для построения модели")
			return

		X = data_dict['X']
		Y = data_dict['Y']
		use_log = data_dict['use_log']
		test_size = data_dict['test_size']
		df_features = data_dict['df_features']
		num_lags = data_dict['num_lags']

		print(f"\nПризнаков: {X.shape[1]}, Наблюдений: {X.shape[0]}")

		# ===== 2. МАСШТАБИРОВАНИЕ =====
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		# ===== 3. РАЗДЕЛЕНИЕ НА TRAIN/TEST =====
		X_train, X_test, Y_train, Y_test = train_test_split(
			X_scaled, Y, test_size=test_size, shuffle=False
		)

		print(f"Обучающая выборка: {X_train.shape[0]}, Тестовая выборка: {X_test.shape[0]}")

		# ===== 4. ОТБОР ПРИЗНАКОВ =====
		from models_analyses.feature_selection import smart_feature_selection

		# Определяем оптимальное количество признаков
		optimal_k = max(10, min(20, X_train.shape[0] // 15))
		print(f"\nОтбираем топ-{optimal_k} признаков из {X.shape[1]}...")

		X_train_selected, X_test_selected, selected_features = smart_feature_selection(
			X_train, Y_train, X_test, Y_test,
			X.columns, sub_folder, top_k=optimal_k
		)

		# ===== 5. ОБУЧЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ =====
		from models_analyses.compare_regression_models import compare_regression_models

		best_model_name, results_df, best_model = compare_regression_models(
			X_train_selected, X_test_selected, Y_train, Y_test,
			scaler, selected_features, sub_folder
		)

		# ===== 6. ЛОГИРОВАНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ =====
		log_file_path = os.path.join(sub_folder, 'analysis_log.log')
		original_stdout = sys.stdout

		with open(log_file_path, 'w', encoding='utf-8') as f:
			sys.stdout = f

			print("="*80)
			print("ИТОГОВЫЙ ОТЧЕТ ПО РЕГРЕССИОННОМУ АНАЛИЗУ")
			print("="*80)

			print(f"\n1. ДАННЫЕ:")
			print(f"   Период: {df_features['contract_week'].min()} - {df_features['contract_week'].max()}")
			print(f"   Всего недель: {len(df_features)}")
			print(f"   Исходных признаков: {X.shape[1]}")
			print(f"   Отобранных признаков: {len(selected_features)}")
			print(f"   Обучающая выборка: {X_train_selected.shape[0]} наблюдений")
			print(f"   Тестовая выборка: {X_test_selected.shape[0]} наблюдений")

			print(f"\n2. ОТОБРАННЫЕ ПРИЗНАКИ:")
			for i, feature in enumerate(selected_features, 1):
				print(f"   {i:2d}. {feature}")

			print(f"\n3. РЕЗУЛЬТАТЫ МОДЕЛЕЙ:")
			print(results_df.to_string(index=False))

			best_r2 = results_df.iloc[0]['R2_test']
			print(f"\n4. ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
			print(f"   R^2 на тесте: {best_r2:.4f}")

			if best_r2 < 0:
				print("\n   ВНИМАНИЕ: Отрицательный R^2 означает, что модель")
				print("   предсказывает хуже, чем простое среднее значение.")
				print("   Прогноз будет ненадежным.")
			elif best_r2 < 0.3:
				print("\n   ВНИМАНИЕ: Низкий R^2 (<0.3) - модель объясняет")
				print("   менее 30% вариации данных. Прогноз имеет высокую погрешность.")
			elif best_r2 < 0.7:
				print("\n   Модель имеет умеренное качество (R^2 < 0.7).")
			else:
				print("\n   Модель имеет хорошее качество (R^2 >= 0.7).")

			# ===== 7. ПРОГНОЗИРОВАНИЕ СЛЕДУЮЩЕЙ НЕДЕЛИ =====
			print(f"\n5. ПРОГНОЗИРОВАНИЕ СЛЕДУЮЩЕЙ НЕДЕЛИ:")

			# Получаем последние значения всех признаков
			last_row_data = {}

			# Для каждого отобранного признака берем последнее доступное значение
			for feature in selected_features:
				if feature in df_features.columns:
					# Если признак есть в df_features напрямую
					last_row_data[feature] = df_features[feature].iloc[-1]
				else:
					# Если это лаговый признак, нужно получить его правильно
					# Например, 'total_amount_lag_1' = значение total_amount на 1 неделю назад
					if '_lag_' in feature:
						base_feature = feature.rsplit('_lag_', 1)[0]
						lag_num = int(feature.rsplit('_lag_', 1)[1])

						if base_feature in df_features.columns:
							if len(df_features) >= lag_num:
								last_row_data[feature] = df_features[base_feature].iloc[-lag_num]
							else:
								last_row_data[feature] = 0
						else:
							last_row_data[feature] = 0
					else:
						last_row_data[feature] = 0

			# Создаем DataFrame для прогноза
			last_week_df = pd.DataFrame([last_row_data], columns=selected_features)

			# Масштабируем (только для отобранных признаков нужен новый scaler)
			# Создаем маску для отобранных признаков в исходном X
			selected_indices = [list(X.columns).index(f) for f in selected_features]

			# Масштабируем только нужные колонки
			X_for_scaler = X.iloc[:, selected_indices]
			scaler_selected = StandardScaler()
			scaler_selected.fit(X_for_scaler)

			last_week_scaled = scaler_selected.transform(last_week_df)

			# Предсказание
			predicted_value = best_model.predict(last_week_scaled)[0]

			# Обратное преобразование из log
			if use_log:
				predicted_next_week = np.expm1(predicted_value)
			else:
				predicted_next_week = predicted_value

			print(f"\n   Прогноз на следующую неделю: {predicted_next_week:,.2f} EUR")
			print(f"   Последняя неделя (факт): {df_features['total_amount'].iloc[-1]:,.2f} EUR")
			print(f"   Средняя за период: {df_features['total_amount'].mean():,.2f} EUR")
			print(f"   Медиана за период: {df_features['total_amount'].median():,.2f} EUR")

			# Оценка отклонения прогноза от среднего
			avg_amount = df_features['total_amount'].mean()
			deviation_pct = ((predicted_next_week - avg_amount) / avg_amount) * 100

			print(f"\n   Отклонение прогноза от среднего: {deviation_pct:+.1f}%")

			print("\n" + "="*80)
			print("АНАЛИЗ ЗАВЕРШЕН")
			print("="*80)

		# Восстанавливаем stdout
		sys.stdout = original_stdout
		print(f"\nПолный отчет сохранен: {log_file_path}")

		return {
			'best_model_name': best_model_name,
			'best_r2': best_r2,
			'prediction': predicted_next_week,
			'selected_features': selected_features
		}

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
	# Создаем столбец contract_week
	filtered_contracts['contract_week'] = filtered_contracts['contract_signing_date'].dt.to_period('W')

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





	