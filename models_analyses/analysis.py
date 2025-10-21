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
	import os
	from datetime import datetime
	"""
	Расширенный анализ месячных затрат с разбивкой по дисциплинам
	"""
	# Создаем папку для результатов
	OUT_DIR = os.path.join(BASE_DIR, "monthly_cost_analysis")
	os.makedirs(OUT_DIR, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
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

	# Преобразуем  Series в DataFrame для Z-Score
	monthly_series_df = monthly_totals.rename('total_price_eur').reset_index()

	# Z-Score (Поиск месяцев, которые сильно отклоняются от CРЕДНЕГО)
	monthly_series_df['Z_Score'] = stats.zscore(monthly_series_df['total_price_eur'])
	outliers_zscore = monthly_series_df[abs(monthly_series_df['Z_Score']) >= 2] # Порог Z >= 2

	# 2. IQR (Поиск месяцев, которые сильно отклоняются от МЕДИАНЫ)
	Q1 = monthly_series_df['total_price_eur'].quantile(0.25)
	Q3 = monthly_series_df['total_price_eur'].quantile(0.75)
	IQR = Q3 - Q1
	# Порог: 1.5 * IQR
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	outliers_iqr = monthly_series_df[
		(monthly_series_df['total_price_eur'] < lower_bound) |
		(monthly_series_df['total_price_eur'] > upper_bound)
		].sort_values(by='total_price_eur', ascending=False)

	
	# Статистические метрики
	monthly_stats = {
	    "Среднемесячные затраты": monthly_totals.mean(),
	    "Медиана": monthly_totals.median(),
	    "Стандартное отклонение": monthly_totals.std(),
	    "Коэффициент вариации": monthly_totals.std() / monthly_totals.mean(),
	    "Минимум": monthly_totals.min(),
	    "Максимум": monthly_totals.max(),
	}

	# 1. Коэффициент вариации по дисциплинам
	cv_by_discipline = filtered_df.groupby('discipline')['total_price_eur'].agg(
		cv=lambda x: x.std() / x.mean() if x.mean() != 0 else 0
	).sort_values(by='cv', ascending=False)
	cv_by_discipline.columns = ['Коэффициент вариации']



	# Функция форматирования валют
	def format_currency(x, p):
		if x >= 1e6:
			return f'{x/1e6:.1f}M €'
		elif x >= 1e3:
			return f'{x/1e3:.0f}K €'
		else:
			return f'{x:.0f} €'
	
	# Визуализация

	# 1. График: Общие затраты по месяцам (EUR) с трендом
	# =======================================================
	plt.figure(figsize=(10, 6)) # Создаем новую фигуру
	monthly_totals.plot(kind='bar', color='skyblue', ax=plt.gca(), label='Месячные затраты')

	# --- НОВЫЙ КОД: Добавление 3-х месячного Скользящего среднего (Moving Average) ---
	window_size = 3 # Окно в 3 месяца (можно попробовать 4 или 6)
	ma_line = monthly_totals.rolling(window=window_size, center=False).mean()
	ma_line.plot(kind='line', color='darkgreen', linewidth=3, label=f'{window_size}-мес. Скользящее среднее', ax=plt.gca())
	# -------------------------------------------------------------------------------

	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.title('1. Общие месячные затраты (EUR), Тренд и Скользящее среднее')
	plt.ylabel('Сумма, EUR')
	plt.xticks(rotation=45)

	# Добавляем линию тренда (уже есть, но убедитесь, что она ниже ma_line)
	x_numeric = range(len(monthly_totals))
	slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, monthly_totals.values)
	trend_line = slope * np.array(x_numeric) + intercept
	plt.plot(x_numeric, trend_line, "r--", alpha=0.7, label=f"Линейный Тренд (R²={r_value**2:.3f})")

	plt.legend(loc='upper left') # Поместите легенду в более удобное место
	plt.grid(axis='y', linestyle='--') # Добавим сетку для читаемости
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'1_monthly_totals_trend_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 2. График: Топ-5 дисциплин по затратам (Stacked Bar)
	# =======================================================
	plt.figure(figsize=(12, 7)) # Создаем новую фигуру
	# Используем только значимые дисциплины
	discipline_analysis[significant_disciplines.index].plot(kind='bar', stacked=True, ax=plt.gca())
	plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.title('2. Топ-дисциплины по месячным затратам (EUR)')
	plt.ylabel('Сумма, EUR')
	plt.xticks(rotation=45)
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'2_discipline_stacked_bar_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 3. График: Доля дисциплин (круговая диаграмма)
	# =======================================================
	plt.figure(figsize=(9, 9)) # Круговая диаграмма лучше смотрится в квадрате
	colors = plt.cm.tab20.colors
	pie_data.plot(
		kind='pie',
		autopct=lambda p: f'{p:.1f}%\n({p * pie_data.sum() / 100:,.0f} EUR)',
		colors=colors[:len(pie_data)],
		startangle=90,
		counterclock=False,
		ax=plt.gca()
	)
	plt.title('3. Распределение общих затрат (Топ-дисциплины + Другие)')
	plt.ylabel('')
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'3_discipline_pie_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 4. График: Нормализованная динамика по дисциплинам
	# =======================================================
	plt.figure(figsize=(12, 7)) # Создаем новую фигуру
	normalized = discipline_analysis.apply(lambda x: (x / x.max()) * 100)
	for discipline in significant_disciplines.index:
		plt.plot(normalized.index.astype(str), normalized[discipline], marker='o', label=discipline)
	plt.title('4. Нормализованная динамика по дисциплинам (100% = макс. для дисциплины)')
	plt.ylabel('Процент от максимального значения')
	plt.xticks(rotation=45)
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'4_discipline_normalized_{timestamp}.png'), dpi=300)
	plt.close()


	# =======================================================
	# 5. График: Box Plot для анализа выбросов (Новый график)
	# =======================================================
	plt.figure(figsize=(8, 5)) # Создаем новую фигуру
	plt.boxplot(monthly_totals.values, vert=False)
	plt.title('5. Box Plot месячных затрат (Визуализация выбросов)')
	plt.yticks([1], ['Всего (EUR)'])
	plt.gca().xaxis.set_major_formatter(FuncFormatter(format_currency))
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, f'5_monthly_boxplot_{timestamp}.png'), dpi=300)
	plt.close()


	# Экспорт данных в Excel
	with pd.ExcelWriter(os.path.join(OUT_DIR, f'cost_analysis_eur_{timestamp}.xlsx')) as writer:

		# Экспортируем результаты анализа выбросов
		outliers_zscore.to_excel(writer, sheet_name='Выбросы (Z-Score)', index=False)
		outliers_iqr.to_excel(writer, sheet_name='Выбросы (IQR)', index=False)
		# --------------------------------------------------------

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

		# 2. Общая статистика (по месяцам)
		monthly_stats_df = pd.DataFrame.from_dict(monthly_stats, orient='index', columns=['Значение (EUR)'])
		# Добавим Коэффициент вариации из monthly_stats и переформатируем
		if 'Коэффициент вариации' in monthly_stats_df.index:
			monthly_stats_df.loc['Коэффициент вариации', 'Значение (EUR)'] *= 100

		monthly_stats_df.to_excel(writer, sheet_name='Общая статистика')

		# 3. CV по дисциплинам
		cv_by_discipline.to_excel(writer, sheet_name='CV по дисциплинам')
	
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
				# Создаем список заголовков, включая имя столбца с поставщиками
				if sheet_name in ["Топ-10 поставщиков", "По валютам", "Динамика по месяцам", "Статистика"]:
				    headers = ["Поставщик"] + list(data.columns)
				else:
				    headers = list(data.columns)
				
				# Записываем все заголовки
				for col_num, value in enumerate(headers):
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

# -----------------------------------------------------

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
    OUTPUT_DIR = os.path.join(BASE_DIR, "network_graphs") # Использование относительного пути
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        file_path = os.path.join(OUTPUT_DIR, f"network_{selected_project}_{layout_name}.png"
        )
        plt.savefig(file_path, dpi=300)
        plt.close()

    QMessageBox.information(
        parent_widget,
        "Сообщение",
        f"Метод сетевого анализа завершен. Файлы сохранены в папке {OUTPUT_DIR}",
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
