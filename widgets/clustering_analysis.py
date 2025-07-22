import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def convert_to_eur(data_df):
	"""
	Функция для конвертации валют в EUR.
	"""
	exchange_rate_to_eur = {
		"AED": 0.27,
		'CNY': 0.14,
		'EUR': 1.2,
		'GBP': 1.32,
		'KRW': 0.00076,
		'KZT': 0.0021,
		'RUB': 0.0113,
		'USD': 1,
		'UZS': 0.000078,
		'JPY': 0.0069,
		'SGD': 0.78
	}
	
	# Преобразуем стоимость в EUR
	data_df['exchange_rate'] = data_df['currency'].map(exchange_rate_to_eur)
	data_df['total_price_eur'] = data_df['total_price'] * data_df['exchange_rate']
	
	return data_df


def iterative_clustering(data_df, pdf):
	"""
	Функция для выполнения иерархической кластеризации и сохранения дендрограммы в PDF.
	"""
	data_df = convert_to_eur(data_df)
	
	grouped_data = data_df.groupby('winner_name').agg({
		'total_price_eur': 'sum',
		'unit_price': 'sum',
		'supplier_qty': 'sum',
		'good_count': 'sum'
	}).reset_index()
	
	features = ['total_price_eur', 'unit_price', 'supplier_qty', 'good_count']
	pairwise_distances = pdist(grouped_data[features], metric='euclidean')
	
	Z = linkage(pairwise_distances, method='ward') # Метод Уорда расчета расстояний
	
	# Страница 1: Дендрограмма
	plt.figure(figsize=(10, 7))
	fig_dendro = plt.gcf()  # Сохраняем ссылку на текущую фигуру (дендрограмму)
	dendrogram(Z, labels=grouped_data['winner_name'].values, leaf_rotation=90, leaf_font_size=6)
	plt.title('Кластеризация. Метод ближайшего соседа')
	plt.xlabel('Поставщики')
	plt.ylabel('Расстояние')
	pdf.savefig(fig_dendro)  # Сохраняем страницу в PDF
	plt.close(fig_dendro)
	
	return Z, grouped_data


def analyze_clusters_at_iteration(Z, grouped_data, num_clusters, pdf):
	"""
	Функция для анализа кластеров и сохранения результатов в PDF.
	"""
	clusters = fcluster(Z, num_clusters, criterion='maxclust')
	grouped_data['cluster'] = clusters
	
	# Страница 2: Список поставщиков и их кластеры
	fig, ax = plt.subplots(figsize=(10, 7))
	ax.axis('off')  # Отключаем оси
	text = ""
	for cluster_num, suppliers in grouped_data.groupby('cluster')['winner_name']:
		text += f"Кластер {cluster_num}: {', '.join(suppliers)}\n"
	
	ax.text(0.5, 0.5, text, fontsize=8, ha='center', wrap=True)
	pdf.savefig(fig)  # Добавляем новую страницу с результатами кластеров
	plt.close(fig)
	
	print(f"Кластеризация завершена, результаты сохранены в PDF")
	return grouped_data
