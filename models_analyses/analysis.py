import pandas as pd
from datetime import datetime
import os
import gc
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox
import platform
import subprocess
import traceback

from PyQt5.QtCore import QMetaObject, Qt
from utils.vizualization_tools import save_top_suppliers_bar_chart
from utils.config import BASE_DIR
from utils.functions import CurrencyConverter
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


def analyze_monthly_cost(parent_widget, df, start_date, end_date):
	from matplotlib.ticker import FuncFormatter
	from scipy import stats
	import numpy as np
	"""
	Расширенный анализ месячных затрат с разбивкой по дисциплинам
	"""
	# Создаем папку для результатов
	OUT_DIR = os.path.join(BASE_DIR, "monthly_cost_analysis")
	os.makedirs(OUT_DIR, exist_ok=True)
	
	# Проверка наличия необходимых колонок
	required_columns = ["close_date", "discipline", "total_price", "currency"]
	missing_columns = [col for col in required_columns if col not in df.columns]
	if missing_columns:
	    QMessageBox.warning(
	        parent_widget, "Ошибка", f"Отсутствуют колонки: {', '.join(missing_columns)}"
	    )
	    return
	
	# Конвертация и фильтрация данных
	df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
	filtered_df = df[(df['close_date'] >= start_date) & (df['close_date'] <= end_date)].copy()
	
	if filtered_df.empty:
		QMessageBox.warning(parent_widget, "Ошибка", "Нет данных для заданного диапазона дат.")
		return
	
	# Добавляем столбцы для анализа
	filtered_df['year_month'] = filtered_df['close_date'].dt.to_period('M')
	filtered_df['month_name'] = filtered_df['close_date'].dt.strftime('%Y-%m')
	
	# Конвертация в EUR
	try:
		converter = CurrencyConverter()
		columns_info = [('total_price', 'currency', 'total_price_eur'),
		                ('unit_price', 'currency', 'unit_price_eur')]
		filtered_df = converter.convert_multiple_columns(
			df=filtered_df, columns_info=columns_info)
	except Exception as e:
		QMessageBox.warning(parent_widget, 'Ошибка конвертации', f"Ошибка при конвертации валют: {str(e)}")
		return
	# Константы
	MIN_TRASHOLD_PERCENT = 2 #2% минимум для отображения
	
	# 1. Анализ по дисциплинам в EUR
	discipline_analysis = filtered_df.groupby(['year_month', 'discipline'])['total_price_eur'].sum().unstack(fill_value=0)
	discipline_totals = filtered_df.groupby('discipline')['total_price_eur'].sum()
	total_sum = discipline_totals.sum()
	
	# показать отдельно только значимые дисциплины
	significant_disciplines = discipline_totals[discipline_totals / total_sum * 100 >= MIN_TRASHOLD_PERCENT]
	other_sum = discipline_totals.sum() - significant_disciplines.sum()
	
	# Создаем Series для круговой диаграммы (топ-5 + "Другие")
	pie_data = significant_disciplines.copy()
	if other_sum > 0:
		pie_data['Другие'] = other_sum
	
	# 3. Общие затраты по месяцам в EUR
	monthly_totals = filtered_df.groupby('year_month')['total_price_eur'].sum()
	
	# Статистические метрики
	monthly_stats = {
	    "Среднемесячные затраты": monthly_totals.mean(),
	    "Медиана": monthly_totals.median(),
	    "Стандартное отклонение": monthly_totals.std(),
	    "Коэффициент вариации": monthly_totals.std() / monthly_totals.mean(),
	    "Минимум": monthly_totals.min(),
	    "Максимум": monthly_totals.max(),
	}
	# Функция форматирования валют
	def format_currency(x, p):
		if x >= 1e6:
			return f'{x/1e6:.1f}M €'
		elif x >= 1e3:
			return f'{x/1e3:.0f}K €'
		else:
			return f'{x:.0f} €'
	
	# Визуализация
	plt.figure(figsize=(18, 12))
	
	# График 1: Общие затраты по месяцам (EUR)
	plt.subplot(2, 2, 1)
	monthly_totals.plot(kind='bar', color='skyblue', ax=plt.gca())
	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.title('Общие месячные затраты (EUR)')
	plt.ylabel('Сумма, EUR')
	plt.xticks(rotation=45)
	
	# добавляем линию тренда
	x_numeric = range(len(monthly_totals))
	slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, monthly_totals.values)
	trend_line = slope * np.array(x_numeric) + intercept
	plt.plot(x_numeric, trend_line, "r--", alpha=0.7, label=f"Тренд (R²={r_value**2:.3f})")
	plt.legend()
	
	# График 2: Топ-5 дисциплин по затратам (EUR)
	plt.subplot(2, 2, 2)
	discipline_analysis[significant_disciplines.index].plot(kind='bar', stacked=True, ax=plt.gca())
	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.title('Топ-5 дисциплин по затратам (EUR)')
	plt.ylabel('Сумма, EUR')
	plt.xticks(rotation=45)
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	
	# График 3: Доля дисциплин (круговая диаграмма с категорией "Другие")
	plt.subplot(2, 2, 3)
	colors = plt.cm.tab20.colors  # Используем цветовую палитру
	pie_data.plot(
		kind='pie',
		autopct=lambda p: f'{p:.1f}%\n({p * pie_data.sum() / 100:,.0f} EUR)',
		colors=colors[:len(pie_data)],  # Берем нужное количество цветов
		startangle=90,
		counterclock=False,
		ax=plt.gca()
	)
	plt.title('Распределение затрат (топ-5 дисциплин)')
	plt.ylabel('')
	
	# График 4: Нормализованная динамика по дисциплинам
	plt.subplot(2, 2, 4)
	normalized = discipline_analysis.apply(lambda x: (x / x.max()) * 100)
	for discipline in significant_disciplines.index:
		plt.plot(normalized.index.astype(str), normalized[discipline], marker='o', label=discipline)
	plt.title('Нормализованная динамика (100% = максимум для дисциплины)')
	plt.ylabel('Процент от максимального значения')
	plt.xticks(rotation=45)
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.grid(True)
	
	plt.tight_layout()
	
	# Сохранение графиков
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	plt.savefig(os.path.join(OUT_DIR, f'monthly_cost_eur_{timestamp}.png'), dpi=300, bbox_inches='tight')
	plt.close()
	
	# Экспорт данных в Excel
	with pd.ExcelWriter(os.path.join(OUT_DIR, f'cost_analysis_eur_{timestamp}.xlsx')) as writer:
		# Сводная таблица по дисциплинам
		pivot_eur = filtered_df.pivot_table(
			index='year_month',
			columns='discipline',
			values='total_price_eur',
			aggfunc='sum',
			fill_value=0
		)
		pivot_eur.to_excel(writer, sheet_name='По дисциплинам (EUR)')
		
		# Итоговая статистика
		summary = filtered_df.groupby('discipline').agg({
			'total_price_eur': ['sum', 'mean', 'count'],
			'total_price': 'sum'
		})
		summary.columns = ['Сумма (EUR)', 'Среднее (EUR)', 'Кол-во закупок', 'Сумма (ориг валюта)']
		summary['Доля, %'] = (summary['Сумма (EUR)'] / summary['Сумма (EUR)'].sum()) * 100
		summary.to_excel(writer, sheet_name='Итоги')
	
	QMessageBox.information(
		parent_widget,
		"Анализ завершен",
		f"Анализ месячных затрат в EUR сохранен в:\n{OUT_DIR}"
	)

def analyze_top_suppliers(parent_widget, df):
	"""
	Расширенный анализ топ-10 поставщиков с исправленными ошибками и улучшенной обработкой
	"""
	try:
		project_name = str(df['project_name'].unique())
		# Создаем папку для результатов (с проверкой имени проекта)
		OUT_DIR = os.path.join(BASE_DIR, "top10_suppliers_analysis", project_name)
		os.makedirs(OUT_DIR, exist_ok=True)
		
		# Конвертация дат и фильтрация
		df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
		start_dt = pd.to_datetime(df['close_date'].min())
		end_dt = pd.to_datetime(df['close_date'].max())
		filtered_df = df[(df['close_date'] >= start_dt) & (df['close_date'] <= end_dt)].copy()
		
		if filtered_df.empty:
			QMessageBox.warning(parent_widget, "Ошибка", "Нет данных для заданного диапазона дат.")
			return QMessageBox.warning(parent_widget, "Ошибка", "Проверьте правильнось даты закрытия Лота")
		
		# Конвертация в EUR
		converter = CurrencyConverter()
		columns_info = [('total_price', 'currency', 'total_price_eur')]
		filtered_df = converter.convert_multiple_columns(df=filtered_df, columns_info=columns_info)
		
		# Рассчитываем период анализа
		delta = end_dt - start_dt
		years = delta.days // 365
		months = (delta.days % 365) // 30
		interval_text = f"{years} г. {months} мес." if years > 0 else f"{months} мес."
		
		# Создаем уникальное имя файла
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		file_name = f"suppliers_analysis_{project_name}_{timestamp}.xlsx"
		file_path = os.path.join(OUT_DIR, file_name)
		
		# Анализ топ поставщиков
		top_suppliers = filtered_df.groupby('winner_name')['total_price_eur'].agg(
			['sum', 'count', 'mean'])
		top_suppliers = top_suppliers.nlargest(10, 'sum')
		top_suppliers.columns = ['Общая сумма (EUR)', 'Кол-во закупок', 'Средняя сумма']
		
		index = top_suppliers.index
		index.name = 'winner_name'
		top_suppliers.index = index
			
		# Проверяем, есть ли данные для анализа
		if top_suppliers.empty:
			QMessageBox.warning(parent_widget, "Ошибка", "Нет данных о поставщиках для анализа.")
			return
		
		# Сохраняем в Excel с дополнительными листами
		with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
			# 1. Основной лист с топ поставщиками
			top_suppliers.to_excel(writer, sheet_name='Топ-10 поставщиков')
			
			# 2. Анализ по валютам
			if 'currency' in filtered_df.columns:
				currency_df = filtered_df[filtered_df['winner_name'].isin(top_suppliers.index)]
				currency_analysis = currency_df.groupby(['winner_name', 'currency'])['total_price'].sum().unstack()
				currency_analysis.to_excel(writer, sheet_name='По валютам')
			
			# 3. Динамика по месяцам
			monthly_df = filtered_df[filtered_df['winner_name'].isin(top_suppliers.index)].copy()
			monthly_df['month'] = monthly_df['close_date'].dt.to_period('M').astype(str)
			monthly_sum = monthly_df.pivot_table(
				index='month',
				columns='winner_name',
				values='total_price_eur',
				aggfunc='sum').fillna(0)
			monthly_sum.to_excel(writer, sheet_name='Динамика по месяцам')
			
			# 4. Общая статистика
			stats = pd.DataFrame({
				'Всего поставщиков': [filtered_df['winner_name'].nunique()],
				'Всего закупок': [len(filtered_df)],
				'Общая сумма (EUR)': [filtered_df['total_price_eur'].sum()],
				'Средняя сумма закупки (EUR)': [filtered_df['total_price_eur'].mean()],
				'Период анализа': [interval_text],
				'Дата анализа': [datetime.now().strftime('%Y-%m-%d %H:%M')]
			}).T
			stats.columns = ['Значение']
			stats.to_excel(writer, sheet_name='Статистика')
			
			# Форматирование Excel
			workbook = writer.book
			header_format = workbook.add_format({
				'bold': True,
				'text_wrap': True,
				'valign': 'top',
				'fg_color': '#4F81BD',
				'font_color': 'white',
				'border': 1
			})
			
			num_format = workbook.add_format({'num_format': '#,##0.00'})
			
			# Для каждого листа получаем соответствующий датафрейм
			sheets_data = {
				'Топ-10 поставщиков': top_suppliers,
				'По валютам': currency_analysis if 'currency' in filtered_df.columns else pd.DataFrame(),
				'Динамика по месяцам': monthly_sum,
				'Статистика': stats
			}
			for sheet_name, data in sheets_data.items():
				worksheet = writer.sheets[sheet_name]
				for col_num, value in enumerate(data.columns.values):
					worksheet.write(0, col_num, value, header_format)
					
					# Числовое форматирование
					if sheet_name != 'Статистика':
						worksheet.set_column(1, len(data.columns)-1, 15, num_format)
						
					# Автоширина столбцов
					for col_num, column in enumerate(data.columns):
						max_len = max(
							len(str(column)),
							data[column].astype(str).str.len().max()
						)
						worksheet.set_column(col_num, col_num, max_len + 2)
					
			# Визуализация ===============================
			plt.figure(figsize=(18, 12))
			
			# График 1: Топ-10 поставщиков (EUR)
			plt.subplot(2, 2, 1)
			top_suppliers['Общая сумма (EUR)'].sort_values().plot(kind='barh', color='steelblue')
			plt.title(f'Топ-10 поставщиков ({interval_text})')
			plt.xlabel('Сумма закупок, EUR')
			plt.grid(axis='x')
			
			# График 2: Соотношение суммы и количества закупок (исправленный)
			plt.subplot(2, 2, 2)
			for i in range(len(top_suppliers)):
				plt.scatter(
					top_suppliers.iloc[i]['Кол-во закупок'],
					top_suppliers.iloc[i]['Общая сумма (EUR)'],
					s=top_suppliers.iloc[i]['Средняя сумма'] * 0.1,
					alpha=0.6
				)
				plt.text(
					top_suppliers.iloc[i]['Кол-во закупок'],
					top_suppliers.iloc[i]['Общая сумма (EUR)'],
					top_suppliers.index[i],
					fontsize=8,
					ha='center',
					va='bottom'
				)
			plt.title('Соотношение количества и суммы закупок')
			plt.xlabel('Количество закупок')
			plt.ylabel('Общая сумма, EUR')
			plt.grid(True)
			
			# График 3: Доля топ-10 поставщиков
			plt.subplot(2, 2, 3)
			total_sum = filtered_df['total_price_eur'].sum()
			top_sum = top_suppliers['Общая сумма (EUR)'].sum()
			other_sum = total_sum - top_sum
			plt.pie([top_sum, other_sum],
			        labels=['Топ-10 поставщиков', 'Остальные'],
			        autopct=lambda p: f'{p:.1f}%\n({p * total_sum / 100:,.0f} EUR)',
			        colors=['lightcoral', 'lightgray'])
			plt.title('Доля топ-10 поставщиков в общих затратах')
			
			# График 4: Динамика по месяцам для топ-3 поставщиков
			plt.subplot(2, 2, 4)
			top_3_suppliers = top_suppliers.index[:3]
			for supplier in top_3_suppliers:
				if supplier in monthly_sum.columns:
					plt.plot(monthly_sum.index, monthly_sum[supplier], marker='o', label=supplier)
					plt.title('Динамика топ-3 поставщиков по месяцам')
					plt.xlabel('Месяц')
					plt.ylabel('Сумма, EUR')
					plt.xticks(rotation=45)
					plt.legend()
					plt.grid(True)
			
			plt.tight_layout()
			
			# Сохраняем графики
			chart_path = os.path.join(OUT_DIR, f'suppliers_visualization_{timestamp}.png')
			plt.savefig(chart_path, dpi=300, bbox_inches='tight')
			plt.close() # ====================================
					
			# Показываем сообщение об успехе
			QMessageBox.information(
				parent_widget,
				"Анализ завершен",
				f"Анализ поставщиков успешно сохранен:\n\n"
				f"Excel-файл: {file_path}\n"
				f"Графики: {chart_path}"
			)
	
	except Exception as e:
		(
			QMessageBox.critical(
				parent_widget,
				"Ошибка",
				f"Произошла ошибка при анализе поставщиков:\n\n{str(e)}"
			))
	print(f"Error in analyze_top_suppliers: {traceback.format_exc()}")


# -----------------------------------------------------

""" Анализ Частота появления Поставщика"""


# DataFrame называется df и содержит столбцы 'discipline', 'actor_name', 'winner_name'

# Добавление процентного соотношения
def analyze_supplier_frequency(df, output_dir="D:/Analysis-Results/Supplier-Frequency", threshold=1):
	import os
	import matplotlib.pyplot as plt
	os.makedirs(output_dir, exist_ok=True)
	
	# Группировка данных
	grouped_df = df.groupby(['discipline', 'actor_name', 'winner_name']).size().reset_index(name='win_count')
	
	# Добавляем общее количество закупок по (discipline, actor_name)
	total_counts = grouped_df.groupby(['discipline', 'actor_name'])['win_count'].transform('sum')
	grouped_df['win_percentage'] = (grouped_df['win_count'] / total_counts * 100).round(2)
	
	# Сортировка данных
	top_suppliers = grouped_df.sort_values(by=['discipline', 'actor_name', 'win_count'], ascending=[True, True, False])
	
	# Сохранение в Excel с дополнительными метриками
	grouped_data = top_suppliers.groupby('discipline')
	for discipline, group_discipline in grouped_data:
		discipline_dir = os.path.join(output_dir, discipline.replace(" ", "_"))
		os.makedirs(discipline_dir, exist_ok=True)
		
		excel_path = os.path.join(discipline_dir, f"{discipline}_supplier_frequency.xlsx")
		with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
			group_discipline.to_excel(writer, index=False, sheet_name='Supplier Frequency')
		
		# Дополнительный график: топ-10 поставщиков в дисциплине
		top_10 = group_discipline.groupby('winner_name')['win_count'].sum().nlargest(10)
		if not top_10.empty:
			fig, ax = plt.subplots(figsize=(10, 6))
			top_10.plot(kind='bar', ax=ax, color='skyblue')
			ax.set_title(f'Top 10 Suppliers in {discipline}')
			ax.set_ylabel('Total Wins')
			plt.tight_layout()
			plt.savefig(os.path.join(discipline_dir, f"top_10_suppliers_{discipline}.png"))
			plt.close()
	
	return top_suppliers


def network_analysis(parent_widget, df):
	"""
	:param df: отфильтрованный датафрейм по одному проекту
	:return: None
	Использованы алгоритмы Fruchterman-Reingold Algorithm и kamada_kawai
	"""
	print('Запускается метод сетевого анализа и построения графов')
	
	# Преобразуем значения project_name в строки
	df['project_name'] = df['project_name'].astype(str)
	
	# Извлечение уникальных значений для валют
	unique_currencies = df['currency'].unique().tolist()
	selected_project = df['project_name'].unique()[0]
	
	output_folder = 'D:/Analysis-Results/network_graphs'
	os.makedirs(output_folder, exist_ok=True)
	
	# Список алгоритмов размещения
	layouts = {
		'spring': nx.spring_layout,
		'kamada_kawai': nx.kamada_kawai_layout,
	}
	
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
			
			if pd.notna(discipline) and pd.notna(supplier):
				# Добавляем связь проект - дисциплина
				G.add_edge(selected_project, discipline)
				# Добавляем связь дисциплина - поставщик
				G.add_edge(discipline, supplier)
		
		# Перебираем все алгоритмы размещения
		for layout_name, layout_func in layouts.items():
			print(f"Построение графика для {currency} с размещением: {layout_name}")
			
			# Оптимизация размещения узлов
			try:
				pos = layout_func(G, seed=42) if layout_name == 'spring' else layout_func(G)
			except Exception as e:
				print(f"Ошибка при вычислении layout {layout_name}: {e}")
				continue
			
			# Получение цветов узлов из атрибутов
			node_colors = [data['color'] for _, data in G.nodes(data=True)]
			
			# Визуализация сети
			plt.figure(figsize=(15, 10))
			nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=6, font_color='black',
			        edge_color='gray')
			# Заголовок
			title = f'Network for {selected_project} in {currency} - {layout_name.capitalize()} Layout'
			plt.title(title, fontsize=14)
			
			# Расширение области для добавления пояснения
			plt.subplots_adjust(bottom=0.2)
			
			# Добавление пояснения
			description = f"Project: {selected_project}, Currency: {currency}, Layout: {layout_name.capitalize()}"
			plt.figtext(0.5, 0.02, description, wrap=True, horizontalalignment='center', fontsize=10)
			
			# Сохранение графика в файл
			file_path = os.path.join(output_folder, f'network_{selected_project}_{currency}_{layout_name}.png')
			try:
				plt.savefig(file_path)
				print(f"График с размещением {layout_name} сохранен: {file_path}")
			except Exception as error:
				print(f"Ошибка при сохранении графика {layout_name}: {error}")
			finally:
				plt.close('all')
				gc.collect()
	
	QMessageBox.information(parent_widget, "Сообщение",
	                        f"Метод сетевого анализа завершен. Файлы сохранены в папке {output_folder}")
	return

def network_analysis_improved(parent_widget, df):
    """
    Улучшенный сетевой анализ для одного проекта с визуализацией
    """
    print("Запускается метод сетевого анализа и построения графов")

    # Проверка наличия необходимых колонок
    required_columns = ["project_name", "currency", "discipline", "winner_name"]
    if not all(col in df.columns for col in required_columns):
        QMessageBox.warning(parent_widget, "Ошибка", "Отсутствуют необходимые колонки.")
        return

    # Подготовка данных и папок
    selected_project = df["project_name"].iloc[0]
    output_folder = os.path.join(
        os.getcwd(), "network_graphs"
    )  # Использование относительного пути
    os.makedirs(output_folder, exist_ok=True)

    # Список алгоритмов размещения
    layouts = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }

    # Создание графа
    G = nx.Graph()

    # Добавляем узлы и связи
    for _, row in df.iterrows():
        project = row["project_name"]
        discipline = row["discipline"]
        supplier = row["winner_name"]

        # Добавляем узлы, если их нет
        G.add_node(project, type="project", color="red")
        G.add_node(discipline, type="discipline", color="green")
        G.add_node(supplier, type="supplier", color="lightblue")

        # Добавляем связи
        G.add_edge(project, discipline)
        G.add_edge(discipline, supplier)

    # Получение цветов узлов
    node_colors = [G.nodes[node]["color"] for node in G.nodes()]

    # Перебираем алгоритмы и строим графики
    for layout_name, layout_func in layouts.items():
        print(f"Построение графика с размещением: {layout_name}")

        try:
            pos = layout_func(G, seed=42) if layout_name == "spring" else layout_func(G)
        except Exception as e:
            print(f"Ошибка при вычислении layout {layout_name}: {e}")
            continue

        plt.figure(figsize=(15, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=700,
            node_color=node_colors,
            font_size=6,
            font_color="black",
            edge_color="gray",
        )

        # Добавляем легенду
        legend_labels = {
            "project": "Проект",
            "discipline": "Дисциплина",
            "supplier": "Поставщик",
        }
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markersize=10,
                markerfacecolor=color,
            )
            for label, color in [
                ("Проект", "red"),
                ("Дисциплина", "green"),
                ("Поставщик", "lightblue"),
            ]
        ]
        plt.legend(handles=legend_handles, title="Тип узла", loc="upper left")

        # Заголовок и пояснения
        title = f"Сетевой анализ проекта {selected_project} ({layout_name.capitalize()} Layout)"
        plt.title(title, fontsize=14)

        file_path = os.path.join(
            output_folder, f"network_{selected_project}_{layout_name}.png"
        )
        plt.savefig(file_path, dpi=300)
        print(f"График с размещением {layout_name} сохранен: {file_path}")
        plt.close()

    QMessageBox.information(
        parent_widget,
        "Сообщение",
        f"Метод сетевого анализа завершен. Файлы сохранены в папке {output_folder}",
    )

def find_common_suppliers_between_disciplines(df):
	"""
	Проверяет, имеют ли поставщики одной дисциплины общих поставщиков с другой дисциплиной
	и возвращает номера лотов для этих поставщиков.
	Параметры:
		df (DataFrame): Данные с колонками ['discipline', 'winner_name', 'lot_number'].
	Возвращает:
		DataFrame: Таблица с парами дисциплин, списком общих поставщиков и номерами лотов.
	"""
	# Группировка данных по дисциплинам
	discipline_suppliers = df.groupby('discipline')['winner_name'].apply(set)
	
	# Список всех дисциплин
	disciplines = discipline_suppliers.index.tolist()
	
	# Список для результатов
	results = []
	
	# Перебор всех пар дисциплин
	for i, discipline1 in enumerate(disciplines):
		for discipline2 in disciplines[i + 1:]:
			# Найдем общих поставщиков между дисциплинами
			common_suppliers = discipline_suppliers[discipline1] & discipline_suppliers[discipline2]
			
			# Если есть общие поставщики, формируем результирующий список
			if common_suppliers:
				results.append({
					'discipline1': discipline1,
					'discipline2': discipline2,
					'common_suppliers': list(common_suppliers)
				})
	
	# Преобразование результатов в DataFrame
	return pd.DataFrame(results)


def compare_materials_and_prices(df, common_suppliers_df):
	from utils.functions import CurrencyConverter, check_file_access
	
	converter = CurrencyConverter()
	df_converted = converter.convert_column(df, amount_column='unit_price', currency_column='currency',
	                                        result_column='amount_eur')
	
	results = []
	
	# Перебор всех строк в common_suppliers_df
	for _, row in common_suppliers_df.iterrows():
		discipline1 = row['discipline1']
		discipline2 = row['discipline2']
		common_suppliers = row['common_suppliers']
		
		for supplier in common_suppliers:
			# Фильтруем данные для поставщика в обеих дисциплинах
			discipline1_data = df_converted[
				(df_converted['discipline'] == discipline1) & (df_converted['winner_name'] == supplier)]
			discipline2_data = df_converted[
				(df_converted['discipline'] == discipline2) & (df_converted['winner_name'] == supplier)]
			
			for good_name in set(discipline1_data['good_name']).intersection(set(discipline2_data['good_name'])):
				discipline1_goods = discipline1_data[discipline1_data['good_name'] == good_name]
				discipline2_goods = discipline2_data[discipline2_data['good_name'] == good_name]
				
				price1 = discipline1_goods['amount_eur'].mean()
				price2 = discipline2_goods['amount_eur'].mean()
				
				# Извлекаем номера лотов
				lot_numbers_discipline1 = discipline1_goods['lot_number'].unique()
				lot_numbers_discipline2 = discipline2_goods['lot_number'].unique()
				
				persent_of_difference = (price1 - price2) * 100 / (price1 + price2)
				
				if persent_of_difference > 10 or persent_of_difference < -10:
					results.append({
						'supplier': supplier,
						'good_name': good_name,
						'discipline1': discipline1,
						'discipline2': discipline2,
						'price_discipline1': price1,
						'price_discipline2': price2,
						'persent_of_diff': persent_of_difference,
						'lot_numbers_discipline1': lot_numbers_discipline1.tolist(),
						'lot_number_discipline2': lot_numbers_discipline2.tolist()
					})
	# Преобразуем results в DataFrame
	results_df = pd.DataFrame(results)
	
	# Указание папки и имени файла
	output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
	os.makedirs(output_folder, exist_ok=True)
	file_path = os.path.join(output_folder, "suppliers_analysis.xlsx")
	
	if check_file_access(file_path):
		# Сохраняем DataFrame в Excel
		results_df.to_excel(file_path, index=False)
		print(f"Файл успешно сохранён: {file_path}")
	else:
		print("Файл занят, программа не может продолжить работу")
	
	return results_df


def matches_results_stat(comparison_results):
	# общее количество совпадений
	total_matches = len(comparison_results)
	unique_suppliers = comparison_results['supplier'].nunique()
	
	# Средний процент расхождения цен
	average_difference = comparison_results['persent_of_diff'].mean()
	
	# Топ-10 поставщиков по количеству совпадений
	top_suppliers = comparison_results['supplier'].value_counts().head(10)
	
	# Вывод статистики
	print(f"Общее количество совпадений: {total_matches}")
	print(f"Уникальные поставщики: {unique_suppliers}")
	print(f"Средний процент расхождения цен: {average_difference:.2f}%")
	print("Топ-10 поставщиков по количеству совпадений:")
	print(top_suppliers)
