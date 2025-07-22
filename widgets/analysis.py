import pandas as pd
import os
import gc
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def process_data(df):
	# предобработка данных
	df['total_price'] = pd.to_numeric(df['total_price'], errors='coerce')
	df = df.dropna(subset=['close_date', 'total_price'])
	# Преобразуем 'close_date' в формат datetime
	df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
	
	# Удаляем строки с NaT в 'close_date'
	df = df.dropna(subset=['close_date'])
	return df


def group_by_currency(df):
	# Группировка данных по валютам
	grouped = df.groupby('currency')
	return grouped


def analyze_monthly_expenses(df, start_date, end_date):
	df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
	
	# Группируем данные по валютам
	grouped_by_currency = df.groupby('currency')
	
	filtered_df = df[(df['close_date'] >= start_date) & (df['close_date'] <= end_date)]
	
	# Проверка данных после фильтрации
	if filtered_df.empty:
		print("Нет данных для заданного диапазона дат.")
		return
	
	print(f"Количество записей для анализа: {len(filtered_df)}")
	
	# --- Графики по валютам ---
	for currency, group in grouped_by_currency:
		group['year_month'] = group['close_date'].dt.to_period('M')
		monthly_expenses = group.groupby('year_month')['total_price'].sum()
		
		if len(monthly_expenses) < 2:
			print(f"Недостаточно данных для валюты {currency}, чтобы построить график.")
			continue
		
		# Визуализация для каждой валюты
		print(f"Построение графика для валюты {currency}.")
		plt.figure(figsize=(12, 8))
		ax = monthly_expenses.plot(kind='line', marker='o')
		plt.title(f'Динамика затрат по месяцам за {start_date} - {end_date} (Валюта: {currency})')
		plt.xlabel('Месяц')
		plt.ylabel(f'Общие затраты ({currency})')
		plt.grid(True)
		
		# Сохраняем график в файл
		plt.savefig(f'monthly_expenses_{currency}.png')
		plt.close()
	
	# --- Общий график в EUR ---
	
	exchange_rate_to_eur = {
		'AED': 0.23,
		'CNY': 0.13,
		'EUR': 1.0,  # Базовая валюта
		'GBP': 1.13,
		'KRW': 0.00077,
		'KZT': 0.002,
		'RUB': 0.011,
		'USD': 0.83,
		'UZS': 0.000071,
		'JPY': 0.0073,
		'SGD': 0.61
	}
	
	# Применяем курс валют
	filtered_df['exchange_rate'] = filtered_df['currency'].map(exchange_rate_to_eur)
	filtered_df['total_price_eur'] = filtered_df['total_price'] * filtered_df['exchange_rate']
	
	# Проверяем, есть ли пустые значения в курсе валют
	missing_currencies = filtered_df[filtered_df['exchange_rate'].isnull()]['currency'].unique()
	if len(missing_currencies) > 0:
		print(f"Отсутствуют курсы для валют: {missing_currencies}")
	
	# Группировка по месяцу и суммирование затрат в EUR
	filtered_df['year_month'] = filtered_df['close_date'].dt.to_period('M')
	monthly_expenses_eur = filtered_df.groupby('year_month')['total_price_eur'].sum()
	
	if monthly_expenses_eur.empty:
		print("Недостаточно данных для построения графика в EUR.")
		return
	
	# Визуализация общего графика в EUR
	print("Построение общего графика в EUR.")
	plt.figure(figsize=(12, 8))
	ax = monthly_expenses_eur.plot(kind='line', marker='o')
	plt.title(f'Динамика месячных затрат в EUR за период {start_date} - {end_date}')
	plt.xlabel('Месяц')
	plt.ylabel('Общие затраты (EUR)')
	plt.grid(True)
	
	# Добавление подписей значений на график
	for x, y in zip(monthly_expenses_eur.index, monthly_expenses_eur):
		ax.text(x, y, f'{y:,.0f} EUR', ha='center', va='bottom')
	
	# Сохраняем график в файл
	plt.savefig('monthly_expenses_allcurr.png')
	plt.close()
	
	print("Графики успешно сохранены по валютам и общий график в EUR.")


def analyze_top_suppliers(df, start_date, end_date):
	# Группируем данные по валютам
	grouped_by_currency = group_by_currency(df)
	
	# Вычисляем количество лет и месяцев между датами
	delta = end_date - start_date
	num_years = delta.days // 365
	num_months = (delta.days % 365) // 30
	
	# Формируем текстовый интервал для заголовка
	interval_text = ''
	if num_years > 0:
		interval_text += f'{num_years} года' if num_years == 1 else f'{num_years} лет'
	if num_months > 0:
		if interval_text:
			interval_text += ' и '
		interval_text += f'{num_months} месяца' if num_months == 1 else f'{num_months} месяцев'
	
	# Проходим по каждой группе валют
	for currency, group in grouped_by_currency:
		# Вывод информации о текущей группе
		print(f"Валюта: {currency}, количество записей: {len(group)}")
		
		# Проверка наличия данных
		if group.empty:
			print(f"Нет данных для валюты {currency}, пропускаем...")
			continue
		
		# Группировка по поставщикам и подсчет затрат
		top_suppliers = group.groupby('winner_name')['total_price'].sum().nlargest(5)
		
		print(f"Топ-5 поставщиков для валюты {currency}:")
		print(top_suppliers)  # Проверочный вывод
		
		# Проверка наличия данных после группировки
		if top_suppliers.empty:
			print(f"Нет данных для построения графика по валюте {currency}.")
			continue
		
		# Создание фигуры для каждого графика
		fig, ax = plt.subplots(figsize=(12, 8))
		top_suppliers.plot(kind='bar', color='skyblue', ax=ax)
		
		# Добавление заголовка с указанием валюты и сумм
		ax.set_title(f'Топ-5 поставщиков по общим затратам за {interval_text} (Валюта: {currency})')
		ax.set_xlabel('Поставщик')
		ax.set_ylabel(f'Общие затраты ({currency})')
		
		# Добавление подписей значений на график
		for i, v in enumerate(top_suppliers):
			ax.text(i, v + 0.05 * v, f'{v:,.0f}', ha='center', va='bottom')
		
		ax.set_xticklabels(top_suppliers.index, rotation=20)
		ax.grid(axis='y')
		
		# Сохраняем график в файл
		plt.savefig(f'top_suppliers_{currency}.png')
		plt.close()
		gc.collect()
	
	print("Анализ топ-5 поставщиков завершен. Графики сохранены.")


""" -------------------------------------------------------------- """

""" Анализ Частота появления Поставщика"""


# DataFrame называется df и содержит столбцы 'discipline', 'actor_name', 'winner_name'

def analyze_supplier_frequency(df):
	# Группируем данные по дисциплине и исполнителю
	grouped_df = df.groupby(['discipline', 'actor_name', 'winner_name']).size().reset_index(name='win_count')
	
	# Находим топ-поставщиков по количеству выигрышей для каждой дисциплины и исполнителя
	top_suppliers = grouped_df.sort_values(by=['discipline', 'actor_name', 'win_count'], ascending=[True, True, False])
	
	# Выводим результаты
	print("Частота выигрышей поставщиков по дисциплинам и исполнителям:")
	print(top_suppliers)
	
	# Визуализация (например, гистограмма)
	import matplotlib.pyplot as plt
	plt.figure(figsize=(12, 8))
	
	for (discipline, actor_name), group in top_suppliers.groupby(['discipline', 'actor_name']):
		plt.bar(group['winner_name'], group['win_count'], label=f'{discipline} ({actor_name})')
	
	plt.xlabel('Поставщик')
	plt.ylabel('Количество выигрышей')
	plt.title('Частота выигрышей поставщиков по дисциплинам и исполнителям')
	plt.legend()
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()
	plt.close()
	return top_suppliers
	gc.collect()


""" Код программы сетевого анализа и построения графов """
""" --------------------------------------------------------- """
def network_analysis(df):
	# Фильтрация данных по выбранному проекту
	project_data = df
	
	# Извлечение уникальных значений для валют
	unique_currencies = df['currency'].unique().tolist()
	selected_project = df['project_name'].unique()[0]
	
	output_folder = 'D:/My_PyQt5_Project_app/network_graphs'
	
	# Перебираем все уникальные валюты для данного проекта
	for currency in unique_currencies:
		# Фильтрация данных по валюте
		currency_data = df[df['currency'] == currency]
		
		# Извлечение уникальных дисциплин и поставщиков для текущей валюты
		unique_disciplines = currency_data['discipline'].unique().tolist()
		unique_suppliers = currency_data['winner_name'].unique().tolist()
		
		# Создание пустого графа для текущей валюты
		G = nx.Graph()
		
		# Добавление узла для проекта (красный цвет)
		G.add_node(selected_project, type='project', color='red')
		
		# Добавление узлов для дисциплин и поставщиков только для текущей валюты
		G.add_nodes_from(unique_disciplines, type='discipline', color='green')
		G.add_nodes_from(unique_suppliers, type='supplier', color='lightblue')
		
		# Добавление связей на основе данных проекта и текущей валюты
		for _, row in currency_data.iterrows():
			discipline = row['discipline']
			supplier = row['winner_name']
			
			# Добавляем связь проект - дисциплина
			G.add_edge(selected_project, discipline)
			
			# Добавляем связь дисциплина - поставщик
			G.add_edge(discipline, supplier)
		
		# Оптимизация размещения узлов (алгоритм spring_layout)
		pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
		
		# получение цветов узлов из атрибутов
		node_colors = [data['color'] for _, data in G.nodes(data=True)]
		
		# Визуализация сети
		plt.figure(figsize=(15, 10))
		nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=8, font_color='black',
		        edge_color='gray')
		plt.title(f'Network for {selected_project} in {currency}')
		
		# Сохранение графика в файл
		file_path = os.path.join(output_folder, f'network_{selected_project}_{currency}.png')
		plt.savefig(file_path)
		plt.close('all')
		gc.collect()
		
		print(f"График для валюты {currency} с дисциплинами и поставщиками сохранен.")
	
	# Создание объединенной сети для всех валют
	try:
		G_combined = nx.Graph()
		print('Граф успешно создан')
	except Exception as e:
		print(f"Ошибка при создании графа: {e}")
		
		
		# Добавление узлов для дисциплин и поставщиков только если они есть в данных
	G_combined.add_nodes_from(project_data['discipline'].unique(), type='discipline', color='green')
	G_combined.add_nodes_from(project_data['winner_name'].unique(), type='supplier', color='lightblue')
	G_combined.add_node(selected_project, type='project', color='red')  # Узел для проекта
	
	# Добавление связей на основе всех данных проекта
	for _, row in project_data.iterrows():
		discipline = row['discipline']
		supplier = row['winner_name']
		
		# Проверка на наличие дисциплины и поставщика
		if pd.notna(discipline) and pd.notna(supplier):
			# Добавление связей (ребер) между узлами
			G_combined.add_edge(selected_project, discipline)  # Связь проект - дисциплина
			G_combined.add_edge(discipline, supplier)  # Связь дисциплина - поставщик
	
	# Список различных алгоритмов размещения
	layouts = {
		'spring': nx.spring_layout,
		'kamada_kawai': nx.kamada_kawai_layout,
	}
	
	# Перебор всех вариантов размещения
	for layout_name, layout_func in layouts.items():
		# Фиксируем random_state для воспроизводимости
		pos_combined = layout_func(G_combined, seed=42)
		
		# Получение цветов узлов из атрибутов
		node_colors_combined = [data['color'] for _, data in G_combined.nodes(data=True)]
		
		# Визуализация сети с выбранным размещением
		plt.figure(figsize=(15, 10))
		nx.draw(G_combined, pos_combined, with_labels=False, node_size=700, node_color=node_colors_combined,
		        edge_color='gray')
		nx.draw_networkx_labels(G_combined, pos_combined, font_size=5, font_color='black')
		plt.title(f'Combined Network for {selected_project} (All Currencies) - {layout_name.capitalize()} Layout')
		
		# Сохранение графика в файл в указанной папке
		file_path = os.path.join(output_folder, f'combined_network_{selected_project}_all_currencies_{layout_name}.png')
		plt.savefig(file_path)
		try:
			plt.close('all')
			print(f"График с размещением {layout_name} сохранен.")
			result = "Построение графов завершено"
			return result
		except Exception as e:
			print(f"Ошибка при завершении метода: {e}")
			
def lotcount_peryear(df):
	print('Загружается модуль подсчета количества лотов')
	# тело модуля подсчета кол-ва лотов
	# Добавляем колонку quarter
	
	project_name = df['project_name'].iloc[0] if 'project_name' in df.columns else "Неизвестный проект"
	year = df['close_date'].dt.year.iloc[0] if 'close_date' in df.columns else "Неизвестный год"
	
	df['quarter'] = df['close_date'].dt.quarter
	# Подсчитываем количество лотов по дисциплинам и кварталам, добавляя номера лотов
	df_grouped = (
		df.groupby(['discipline', 'quarter'])
		.agg(
			lot_count=('lot_number', 'nunique'),  # Подсчет уникальных lot_number
			lot_numbers=('lot_number', lambda x: ', '.join(map(str, x.unique())))  # Список уникальных номеров лотов
		)
		.reset_index()
		.rename(columns={'quarter': 'Квартал'})  # Переименование столбца
	)
	
	# Подготовим данные для визуализации
	df_pivot = df_grouped.pivot(index='discipline', columns='Квартал', values='lot_count').fillna(0)
	
	# Создаем визуализацию
	ax = df_pivot.plot(
		kind='bar',
		stacked=True,
		figsize=(10, 6),
		color=['blue', 'green', 'orange', 'red'] # Цвета для кварталов
	)
	# Добавляем подписи значений на каждом блоке
	for container in ax.containers:
		for bar in container:
			height = bar.get_height()
			if height > 0:  # Показывать только ненулевые значения
				ax.text(
					bar.get_x() + bar.get_width() / 2,  # Координата X
					bar.get_y() + height / 2,  # Координата Y
					f'{int(height)}',  # Значение (округляем до целого числа)
					ha='center',  # Горизонтальное выравнивание
					va='center',  # Вертикальное выравнивание
					fontsize=8,  # Размер шрифта
					color='white' if height > 5 else 'black'  # Цвет текста
				)
	
	# Настройка графика
	plt.title(f"Количество лотов по проекту {project_name} за {year} год")
	plt.xlabel("Дисциплина")
	plt.ylabel("Количество лотов")
	plt.legend(
		title="Квартал",
		loc="upper right",
		labels=["1 квартал", "2 квартал", "3 квартал", "4 квартал"]
	)
	plt.tight_layout()
	
	# Сохраняем результаты
	pure_project_name = project_name.replace(' ', '_').replace(':', '').replace('/', '_')
	output_folder = r"D:\Analysis-Results\Statistics"
	os.makedirs(output_folder, exist_ok=True)  # Создаем папку, если она не существует
	
	# Сохраняем график в PNG
	graph_path = os.path.join(output_folder, f"{pure_project_name}_{year}_chart.png")
	plt.savefig(graph_path, dpi=300)
	plt.close()  # Закрываем график
	
	output_path = os.path.join(output_folder, f"{pure_project_name}_{year}_statistics.xlsx")
	df_grouped.to_excel(output_path, index=False)
	print(f"Результаты сохранены в файл: {output_path}")
	
		
