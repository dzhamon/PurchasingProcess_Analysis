from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
from utils.vizualization_tools import cluster_results_visualize
import matplotlib.pyplot as plt
from utils.config import BASE_DIR
from datetime import datetime
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Регистрация шрифта
def register_fonts():
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        logging.info("Шрифт DejaVuSans зарегистрирован успешно.")
    except Exception as e:
        logging.error(f"Ошибка при регистрации шрифта: {e}")

register_fonts()

def export_to_excel(df, save_path):
    """
    Экспортирует DataFrame в Excel.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_excel(save_path, index=False)
        logging.info(f"Данные успешно экспортированы в {save_path}")
    except Exception as e:
        logging.error(f"Ошибка при экспорте данных в Excel: {e}")
        
def save_plot(plot_func, save_path, **kwargs):
    """
    Сохраняет график, вызывая переданную функцию построения.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_func(**kwargs)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"График сохранен: {save_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении графика: {e}")


class SeedKMeansClustering:
    def __init__(self, kpi_analyzer):
        """
        Инициализация с экземпляром MyLotAnalyzeKPI.
        """
        self.kpi_analyzer = kpi_analyzer
        self.full_data_with_clusters = None

    def perform_clustering(self):
        """
        Выполняет кластеризацию с использованием Seed Points.
        """
        try:
            # Шаг 1: Рассчитать KPI
            df_kpi = self.kpi_analyzer.calculate_kpi(self.kpi_analyzer.df)
    
            # Шаг 2: Получить Seed Data и выборку обучения
            seed_data = self.kpi_analyzer.get_seed_data(df_kpi)
            remaining_data = df_kpi[~df_kpi.index.isin(seed_data.index)].copy()  # Создаем копию!
    
    
            # Шаг 3: Подготовить данные
            seed_points = seed_data[['total_lots', 'avg_time_to_close', 'avg_lot_value', 'sum_lot_value']].values
            n_clusters = len(seed_data['discipline'].unique())
            features = remaining_data[['total_lots', 'avg_time_to_close', 'avg_lot_value', 'sum_lot_value']].values
            
            # Кластеризация
            kmeans = KMeans(n_clusters=n_clusters, init=seed_points, n_init=10)
            kmeans.fit(features)
    
            # Добавить метки кластеров к данным
            remaining_data['cluster_label'] = kmeans.labels_
            seed_data['cluster_label'] = range(len(seed_data))  # Метки кластеров для Seed Data
    
            # Шаг 4: Оценка результатов кластеризации
            silhouette_avg = silhouette_score(features, kmeans.labels_)
            logging.info(f"Silhouette Score: {silhouette_avg}")
    
            # Шаг 5: Объединить данные
            self.full_data_with_clusters = pd.concat([seed_data, remaining_data])
            return self.full_data_with_clusters, kmeans
        except Exception as error:
            logging.error(f"Ошибка кластеризации: {error}")
            return None, None
        

    def plot_cluster_distribution(self, df_clusters, save_path):
        print('Мы в Гистограмме?')
        """
        Строит гистограмму распределения по кластерам.
        :param df_clusters: DataFrame с данными кластеризации, содержащий столбец 'cluster_label'.
        """
        def plot_histogram():
            # Получение распределения
            cluster_distribution = df_clusters['cluster_label'].value_counts()
            cluster_distribution.sort_index().plot(kind='bar', color='skyblue')
            plt.title('Распределение по кластерам')
            plt.xlabel('Кластеры')
            plt.ylabel('Количество исполнителей')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        save_plot(plot_histogram, save_path)
        
def analysis_df_clusters(df):
    from sklearn.cluster import KMeans
    import pandas as pd
    
    OUT_DIR = os.path.join(BASE_DIR, "Cluster_Analysis")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    df_cluster_old = df.copy()
    # создадим словарь, для хранения датафреймов с кластерами
    clusters_data = {}
    # итерируемся по уникальным значениям кластеров
    for cluster_id in range(len(df['cluster_label'].unique())):
        # фильтруем датафрейм, оставляя только строки с текущим номером кластера
        cluster_df = df[df['cluster_label'] ==  cluster_id]
        # сохраняем полученный датафрейм в словарь
        clusters_data[f"cluster{cluster_id}"] = cluster_df
        
    # выведем все кластера с их метриками в разные вкладки одного excel файла
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"discip_actor_clusters_{timestamp}.xlsx"
    file_path = os.path.join(OUT_DIR, file_name)
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for cluster_dict, df_dict in clusters_data.items():
            # преобразуем метку кластера в строку для имени листа
            sheet_name = str(cluster_dict)
            df_dict.to_excel(writer, sheet_name=sheet_name, index=False)

    # рассчитываем средние значения выбранных метрик
    metrics = ['total_lots', 'sum_lot_value', 'avg_time_to_close']
    
    # создадим пустой датафрейм для хранения средних значений по кластерам
    cluster_means = pd.DataFrame(index=range(len(df['cluster_label'].unique())), columns=metrics)
    
    # итерируемся по всем уникальным значениям кластеров
    for cluster_id in range(len(df['cluster_label'].unique())):
        # Фильтруем DataFrame для текущего кластера
        cluster_df = df[df['cluster_label'] == cluster_id]
        # Рассчитываем средние значения выбранных показателей для текущего кластера
        means = cluster_df[metrics].mean()
        # Записываем полученные средние значения в итоговый DataFrame
        cluster_means.loc[cluster_id] = means
    
    # Выводим DataFrame со средними значениями по кластерам
    print(cluster_means)
    
    # Инициализируем модель KMeans с 3 кластерами
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # Обучаем модель на данных о средних значениях кластеров
    kmeans.fit(cluster_means)
    
    # Получаем метки кластеров, к которым был отнесен каждый исходный кластер
    efficiency_labels = kmeans.labels_
    
    # Создаем новый DataFrame, чтобы сопоставить исходные номера кластеров с их метками эффективности
    efficiency_df = pd.DataFrame({'original_cluster': cluster_means.index, 'efficiency_group': efficiency_labels})
    
    # # занимаемся определением групп эффективности -----------------------------------
    # # Создадим временный DataFrame для анализа
    # analysis_df = cluster_means.copy()
    # analysis_df['original_cluster'] = analysis_df.index  # Добавляем original_cluster как столбец для мерджа
    #
    # # Объединяем analysis_df с efficiency_df, чтобы получить группы эффективности
    # analysis_df = pd.merge(analysis_df, efficiency_df, on='original_cluster', how='left')
    #
    # # Группируем по efficiency_group и вычисляем средние значения метрик для каждой группы
    # group_characteristics = analysis_df.groupby('efficiency_group')[metrics].mean()
    #
    # group_characteristics.plot(kind='bar', subplots=True, layout=(1, len(metrics)), figsize=(15, 5), sharey=False)
    # plt.suptitle('Средние значения метрик по группам эффективности', y=1.02)
    # plt.tight_layout()
    # plt.show()
    #
    
    # Выводим центроиды KMeans, чтобы посмотреть на них напрямую
    print("\nЦентроиды KMeans (средние значения признаков для каждой из 3 групп):")
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=metrics, index=[f'Group {i}' for i in range(3)])
    print(centroids_df)
    # ----------------------------------
    
    # Определяем метрики, где больше - лучше, и где меньше - лучше
    positive_metrics = ['total_lots', 'sum_lot_value']
    negative_metrics = ['avg_time_to_close']
    # Создаем копию для нормализации
    normalized_centroids = centroids_df.copy()
    
    # 1. Нормализация данных
    for col in positive_metrics:
        min_val = centroids_df[col].min()
        max_val = centroids_df[col].max()
        if (max_val - min_val) == 0:  # Избегаем деления на ноль, если все значения одинаковые
            normalized_centroids[col] = 0.5  # Или любое среднее значение, если нет вариации
        else:
            normalized_centroids[col] = (centroids_df[col] - min_val) / (max_val - min_val)
    
    for col in negative_metrics:
        min_val = centroids_df[col].min()
        max_val = centroids_df[col].max()
        if (max_val - min_val) == 0:  # Избегаем деления на ноль
            normalized_centroids[col] = 0.5
        else:
            # Инвертируем шкалу: 1 - (значение - мин) / (макс - мин)
            normalized_centroids[col] = 1 - (centroids_df[col] - min_val) / (max_val - min_val)
    
    print("\nНормализованные центроиды (все метрики теперь: больше - лучше):\n", normalized_centroids)
    
    # 2. Расчет сводного показателя (Score)
    # Здесь мы просто суммируем нормализованные значения.
    # Например: score = (norm_total_lots * 0.4) + (norm_sum_lot_value * 0.4) + (norm_avg_time_to_close * 0.2)
    # Для простоты сейчас используем равные веса.
    normalized_centroids['efficiency_score'] = normalized_centroids[positive_metrics].sum(axis=1) + \
                                               normalized_centroids[negative_metrics].sum(axis=1)
    
    print("\nЦентроиды со сводным показателем эффективности:\n", normalized_centroids)
    
    # Ранжирование групп по сводному показателю
    ranked_groups = normalized_centroids.sort_values(by='efficiency_score', ascending=False)
    
    print("\nРанжирование групп по эффективности:")
    print(ranked_groups[['efficiency_score']])
    
    # Определение категорий
    category_names = ['Эффективные менеджеры', 'Средние менеджеры', 'Малоэффективные менеджеры']
    group_categories = {ranked_groups.index[i]: category_names[i] for i in range(len(category_names))}
    
    # --- Создание словаря с данными по центроидам ---
    
    # Инициализируем пустой словарь для хранения данных
    group_analysis_results = {}
    
    # итерируемся по каждой группе в исходном датафрейме с центроидами
    for group_name in centroids_df.index:
        # извлекаем оригинальные значения метрик для текущей группы
        original_metrics = centroids_df.loc[group_name].to_dict()
        
        # Извлекаем нормализованные значения метрик и efficiency_score
        normalized_metrics = normalized_centroids.loc[group_name][positive_metrics + negative_metrics].to_dict()
        efficiency_score = normalized_centroids.loc[group_name]['efficiency_score']
        
        # Извлекаем категорию эффективности для текущей группы
        efficiency_category = group_categories[group_name]
        
        # Добавляем данные в основной словарь
        group_analysis_results[group_name] = {
            "efficiency_category": efficiency_category,
            "original_metrics": original_metrics,
            "normalized_metrics": normalized_metrics,
            "efficiency_score": efficiency_score
        }
    
    # Выведем полученный словарь для проверки
    print("\n--- Словарь с данными по группам эффективности ---")
    import json
    print(json.dumps(group_analysis_results, indent=4, ensure_ascii=False))
    
    
    # визуализируем полученные результаты
    cluster_results_visualize(df_cluster_old, efficiency_df)

   