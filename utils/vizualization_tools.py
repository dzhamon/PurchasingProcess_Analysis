import os
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from utils.functions import clean_filename
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem

def create_plot_graf(good_name, filtered_data, output_folder):
    print("\n--- Начало создания графика ---")
    print(f"Товар: {good_name}")
    print(f"Данные:\n{filtered_data[['winner_name', 'avg_unit_price']]}")
    
    try:
        # 1. Проверка данных
        if filtered_data.empty:
            raise ValueError("Пустой DataFrame")
        
        if 'winner_name' not in filtered_data.columns or 'avg_unit_price' not in filtered_data.columns:
            raise ValueError("Отсутствуют необходимые столбцы")
        
        # 2. Очистка данных
        plot_data = filtered_data.dropna(subset=['winner_name', 'avg_unit_price']).copy()
        plot_data['avg_unit_price'] = pd.to_numeric(plot_data['avg_unit_price'], errors='coerce')
        plot_data = plot_data.dropna(subset=['avg_unit_price'])
        
        if plot_data.empty:
            raise ValueError("Нет валидных данных после очистки")
        
        # 3. Подготовка пути
        cleaned_good_name = re.sub(r'[\\/*?:"<>|]', "_", good_name)  # Более строгая очистка
        os.makedirs(output_folder, exist_ok=True)
        png_file = os.path.join(output_folder, f"{cleaned_good_name}_price_analysis.png")
        print(f"Путь для сохранения: {png_file}")
        
        # 4. Создание графика
        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            x=plot_data['winner_name'],
            height=plot_data['avg_unit_price'],
            color='skyblue'
        )
        
        # Добавление значений на столбцы
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
        
        plt.title(f'Средняя цена за единицу: {good_name[:50]}...', pad=20)
        plt.xlabel('Поставщик')
        plt.ylabel('Цена (EUR)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 5. Сохранение и отображение
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        # print(f"График сохранён: {png_file}")
        # # plt.show()  # Отображение графика
        plt.close()
    
    except Exception as e:
        print(f"!!! Ошибка при создании графика для '{good_name}': {str(e)}")
    return


def plot_bar_chart(x, y, title, x_label, y_label, output_file):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
def preprocess_supplier_names(supplier_names):
    # Удаляем лидирующие пробелы
    supplier_names = supplier_names.str.strip()
    # Удаляем приставку "не использовать дубль"
    supplier_names = supplier_names.str.replace(r'^Не использовать дубль\s*', '', regex=True)
    # Ограничиваем длину до 15 символов
    supplier_names = supplier_names.str[:15]
    
    return supplier_names

def save_top_suppliers_bar_chart(top_suppliers, currency, interval_text, output_dir):
    """
    Создание и сохранение графиков топ-10 поставщиков
    """
    top_suppliers.index = preprocess_supplier_names(top_suppliers.index)
    
    # Создаем фигуру для графика
    fig, ax = plt.subplots(figsize=(12, 8))
    top_suppliers.plot(kind='bar', color='skyblue', ax=ax)
    
    # Обновляем метки оси X
    ax.set_xticks(range(len(top_suppliers)))
    ax.set_xticklabels(top_suppliers.index, rotation=30)

    # Add title and labels
    ax.set_title(f'Top-10 Suppliers by Total Costs for {interval_text} (Currency: {currency})')
    ax.set_xlabel('Supplier')
    ax.set_ylabel(f'Total Costs ({currency})')

    # добавляем значения на столбцы
    for i, v in enumerate(top_suppliers):
        ax.text(i, v + 0.07 * v, f'{v:,.0f}', ha='center', va='bottom')

    ax.grid(axis='y')

    # сохраняем графики в указанную директорию
    file_path = os.path.join(output_dir, f'top_suppliers_{currency}.png')
    plt.savefig(file_path)
    plt.close()
    gc.collect()


def visualize_price_differences(df):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Создаем папку для сохранения графиков, если её нет
    output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
    os.makedirs(output_folder, exist_ok=True)
    unique_disciplines = df['discipline1'].unique()
    for discipline in unique_disciplines:
        filtered_df = df[df['discipline1'] == discipline]
        materials = filtered_df['good_name']
        prices_discipline1 = filtered_df['price_discipline1']
        prices_discipline2 = filtered_df['price_discipline2']
        
        x = np.arange(len(materials))  # Positions for bars
        bar_width = 0.35  # ширина каждого столбца
        
        # создаем график
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bar1 = ax.bar(
            x - bar_width / 2,
            prices_discipline1,
            width=bar_width,
            label='Discipline 1',
            color='skyblue'
        )
        bar2 = ax.bar(
            x + bar_width / 2,
            prices_discipline2,
            width=bar_width,
            label='Discipline 2',
            color='orange'
        )
        
        ax.set_xlabel('Materials', fontsize=12)
        ax.set_ylabel('Unit Price', fontsize=12)
        ax.set_title(f'Comparison of Unit Prices for Discipline: {discipline}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(materials, rotation=45, ha='right', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend()
        
        plt.tight_layout()
        # Сохраняем график
        output_file = os.path.join(output_folder, f"{discipline}_price_comparison.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)  # Закрываем график, чтобы не занимать память
    
    print(f"Графики успешно сохранены в папку: {output_folder}")
    
    return


def heatmap_common_suppliers(comparison_results):
    print('Запускается метод heatmap_common_suppliers')
    
    # Папка для сохранения тепловой карты
    output_folder = r"D:\Analysis-Results\suppliers_between_disciplines"
    os.makedirs(output_folder, exist_ok=True)  # Создаем папку, если она не существует
    
    # Создаем матрицу пересечений
    disciplines = set(comparison_results['discipline1']).union(set(comparison_results['discipline2']))
    # heatmap_data = pd.DataFrame(index=list(disciplines), columns=list(disciplines)).fillna(0)
    # heatmap_data = pd.DataFrame(index=list(disciplines), columns=list(disciplines)).fillna(0).infer_objects(copy=False)
    heatmap_data = pd.DataFrame(index=list(disciplines), columns=list(disciplines), dtype=float).fillna(0)
    
    for _, row in comparison_results.iterrows():
        discip1, discip2 = row['discipline1'], row['discipline2']
        heatmap_data.at[discip1, discip2] += 1
        heatmap_data.at[discip2, discip1] += 1
    
    # Визуализация тепловой карты
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        fmt="g",
        cbar=True,
        square=True,
        linewidths=0.5
    )
    
    # Добавляем отступы
    plt.title("Количество общих товаров между дисциплинами", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2.0)  # Устанавливаем дополнительные отступы
    
    # Сохраняем тепловую карту
    output_file = os.path.join(output_folder, "heatmap_common_suppliers.png")
    plt.savefig(output_file, dpi=300)
    plt.close()  # Закрываем график, чтобы не занимать память
    
    print(f"Тепловая карта успешно сохранена в файл: {output_file}")
    
    return


def visualize_isolation_forest(analyzed_df):
    # Проверка наличия необходимых столбцов
    if 'unit_price_in_eur' not in analyzed_df.columns or 'total_price_in_eur' not in analyzed_df.columns:
        print("Для визуализации нужны столбцы 'unit_price_in_eur' и 'total_price_in_eur'")
        return
    
    # Используем только два признака для 2D визуализации
    data = analyzed_df[['unit_price_in_eur', 'total_price_in_eur']].dropna()
    
    # Создаём модель Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    data['is_anomaly'] = model.predict(data)  # 1 - нормальные данные, -1 - аномалии
    
    # Создаём сетку для визуализации границ
    xx, yy = np.meshgrid(
        np.linspace(data['unit_price_in_eur'].min(), data['unit_price_in_eur'].max(), 100),
        np.linspace(data['total_price_in_eur'].min(), data['total_price_in_eur'].max(), 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid_points).reshape(xx.shape)
    
    # Директория для сохранения графика
    output_dir = r"D:\Analysis-Results\efficient_analyses"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "isolation_forest_visualization.png")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    
    # Границы разделения
    plt.contourf(xx, yy, scores, levels=50, cmap=plt.cm.RdYlBu_r, alpha=0.6)
    
    # Точки данных
    plt.scatter(
        data['unit_price_in_eur'], data['total_price_in_eur'],
        c=data['is_anomaly'].map({1: 'blue', -1: 'red'}),
        edgecolors='k', alpha=0.8, label='Data Points'
    )
    
    plt.title('Isolation Forest - Visualization of Anomalies')
    plt.xlabel('Unit Price (EUR)')
    plt.ylabel('Total Price (EUR)')
    plt.colorbar(label='Anomaly Score')
    plt.legend(['Normal', 'Anomalies'], loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # Сохранение графика в файл PNG
    plt.savefig(output_file_path, format='png', dpi=300)
    plt.show()
    
    print(f"График успешно сохранён в файл: {output_file_path}")


def save_herfind_hirshman_results(supplier_stats, hhi):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Создаем директорию для результатов
    output_dir = r"D:\Analysis-Results\hirshman_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем supplier_stats и hhi в Excel
    supplier_stats_path = os.path.join(output_dir, "supplier_stats.xlsx")
    supplier_stats.to_excel(supplier_stats_path, index=False)
    
    hhi_path = os.path.join(output_dir, "hhi_index.xlsx")
    hhi.to_excel(hhi_path, index=False)
    
    # DataFrame для накопления всех major_suppliers
    all_major_suppliers = pd.DataFrame()
    
    # Генерация круговых диаграмм для каждой дисциплины
    for discipline in supplier_stats['discipline'].unique():
        # Фильтруем данные по дисциплине
        filtered = supplier_stats[supplier_stats['discipline'] == discipline]
        
        # Разделяем на поставщиков с долей >= 8% и "других"
        major_suppliers = filtered[filtered['supp_share'] >= 8]
        
        # Добавляем колонку 'discipline' для идентификации дисциплины
        major_suppliers = major_suppliers.copy()
        major_suppliers['discipline'] = discipline
        
        # Накапливаем major_suppliers в общий DataFrame
        all_major_suppliers = pd.concat([all_major_suppliers, major_suppliers], ignore_index=True)
        
        # Рассчитываем суммарную долю "других"
        other_suppliers_share = filtered[filtered['supp_share'] < 8]['supp_share'].sum()
        
        # Подготовка данных для диаграммы
        labels = list(major_suppliers['counterparty_name'])
        sizes = list(major_suppliers['supp_share'])
        
        if other_suppliers_share > 0:
            labels.append('Другие')
            sizes.append(other_suppliers_share)
        
        # Построение диаграммы
        plt.figure(figsize=(12, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f"Доли поставщиков для дисциплины: {discipline}")
        
        # Сохранение графика
        pie_chart_path = os.path.join(output_dir, f"{discipline}_supplier_shares_pie_chart.png")
        plt.savefig(pie_chart_path)
        plt.close()
    
    # Сохраняем итоговый DataFrame major_suppliers
    major_suppliers_path = os.path.join(output_dir, "all_major_suppliers.xlsx")
    all_major_suppliers.to_excel(major_suppliers_path, index=False)
    
    print("Результаты сохранены.")
    return all_major_suppliers

def cluster_results_visualize(df_clusters, efficiency_df):
    print("--- Дополнительная проверка данных для объединения ---")
    
    # Проверка типов данных столбцов объединения
    print("\nТипы данных в df_clusters['cluster_label']:", df_clusters['cluster_label'].dtype)
    print("Типы данных в efficiency_df['original_cluster']:", efficiency_df['original_cluster'].dtype)
    
    # Преобразование типов данных, если они различаются (например, в int)
    # Важно: если есть NaN, это может вызвать ошибку.
    # Лучше сначала преобразовать в числовой тип, который может содержать NaN (например, Int64 из pandas)
    # или отфильтровать NaN, если они не нужны для объединения.
    if df_clusters['cluster_label'].dtype != efficiency_df['original_cluster'].dtype:
        print("\nТипы данных различаются. Попытка преобразовать их в 'int'.")
        try:
            df_clusters['cluster_label'] = df_clusters['cluster_label'].astype(int)
            efficiency_df['original_cluster'] = efficiency_df['original_cluster'].astype(int)
            print("Преобразование типов данных выполнено успешно.")
        except ValueError as e:
            print(f"Ошибка при преобразовании типов данных: {e}")
            print("Проверьте, нет ли нечисловых значений в 'cluster_label' или 'original_cluster'.")
    
    # Проверка уникальных значений в столбцах объединения
    print("\nУникальные cluster_label в df_clusters (первые 20):")
    print(df_clusters['cluster_label'].unique()[:20])  # Выводим только первые 20, чтобы не загромождать вывод
    
    print("\nУникальные original_cluster в efficiency_df (первые 20):")
    print(efficiency_df['original_cluster'].unique()[:20])
    
    # Проверка совпадения уникальных значений
    missing_in_efficiency = set(df_clusters['cluster_label'].unique()) - set(efficiency_df['original_cluster'].unique())
    missing_in_clusters = set(efficiency_df['original_cluster'].unique()) - set(df_clusters['cluster_label'].unique())
    
    if missing_in_efficiency:
        print(
            f"\nCluster_label из df_clusters, которых нет в efficiency_df['original_cluster']: {missing_in_efficiency}")
    if missing_in_clusters:
        print(f"\nOriginal_cluster из efficiency_df, которых нет в df_clusters['cluster_label']: {missing_in_clusters}")
    
    print("--- Конец дополнительной проверки ---")
    
    # --- Теперь повторим объединение и визуализации ---
    
    # 2. Соединение информации о кластерах и группах эффективности
    # Убедимся, что типы данных совпадают перед мержем
    # Если преобразование выше не сработало, попробуйте force-cast, если уверены в данных
    # df_clusters['cluster_label'] = df_clusters['cluster_label'].astype(str) # Пример, если числа в разных форматах
    # efficiency_df['original_cluster'] = efficiency_df['original_cluster'].astype(str)
    
    cluster_efficiency = pd.merge(df_clusters[['cluster_label', 'actor_name']],
                                  efficiency_df,
                                  left_on='cluster_label',
                                  right_on='original_cluster',
                                  how='inner')  # Изменим how='left' на how='inner' для начала.
    # Это гарантирует, что останутся только строки,
    # для которых есть совпадение в обоих DataFrame.
    # Если после этого появятся данные, то проблема была в несовпадении ключей.
    # Если по-прежнему пусто - значит совпадений очень мало или нет.
    
    print("\n--- Проверка объединенного DataFrame после исправления ---")
    print("Первые 5 строк объединенного DataFrame (cluster_efficiency) после исправления:")
    print(cluster_efficiency.head())
    print("Уникальные группы эффективности в объединенном DataFrame после исправления:")
    print(cluster_efficiency['efficiency_group'].unique())
    print("Количество NaN значений в efficiency_group после исправления:")
    print(cluster_efficiency['efficiency_group'].isnull().sum())
    print("Количество столбцов в cluster_efficiency_counts после исправления:")
    cluster_efficiency_counts = cluster_efficiency.groupby(['cluster_label', 'efficiency_group'])[
        'actor_name'].nunique().unstack(fill_value=0)
    print(cluster_efficiency_counts.shape[1])
    print("--- Конец проверки исправленного DataFrame ---")
    
    # --- Визуализации (после проверки можно их снова запустить) ---
    
    # 1. Распределение исполнителей по кластерам (этот график обычно не должен быть одноцветным, если кластеров больше одного)
    cluster_counts = df_clusters.groupby('cluster_label')['actor_name'].nunique().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    plt.title('Распределение количества исполнителей по кластерам')
    plt.xlabel('Номер кластера')
    plt.ylabel('Количество исполнителей')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 2. Стекированная столбчатая диаграмма распределения групп эффективности внутри кластеров
    # (исправленная версия с учетом, что кластеры по X, группы эффективности по цветам)
    if not cluster_efficiency_counts.empty and cluster_efficiency_counts.shape[1] > 1:
        cluster_efficiency_counts.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
        plt.title('Распределение групп эффективности внутри кластеров')
        plt.xlabel('Номер кластера')
        plt.ylabel('Количество исполнителей')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Группа эффективности')
        plt.tight_layout()
        plt.show()
    else:
        print("Недостаточно данных для стекированной столбчатой диаграммы по группам эффективности.")
    
    # 3. Круговая диаграмма распределения кластеров по группам эффективности
    efficiency_group_counts = efficiency_df['efficiency_group'].value_counts()
    if not efficiency_group_counts.empty and len(efficiency_group_counts) > 1:  # Проверка, что групп больше одной
        plt.figure(figsize=(8, 8))
        plt.pie(efficiency_group_counts, labels=efficiency_group_counts.index, autopct='%1.1f%%', startangle=140,
                colors=sns.color_palette('pastel'))
        plt.title('Распределение кластеров по группам эффективности')
        plt.tight_layout()
        plt.show()
    else:
        print("Недостаточно групп эффективности для круговой диаграммы.")
    
    # 4. Столбчатая диаграмма "Распределение кластеров по группам эффективности" (как вы просили)
    cluster_distribution_by_efficiency = efficiency_df.groupby('efficiency_group')['original_cluster'].nunique()
    if not cluster_distribution_by_efficiency.empty and len(cluster_distribution_by_efficiency) > 1:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=cluster_distribution_by_efficiency.index,
                    y=cluster_distribution_by_efficiency.values,
                    palette='viridis')
        plt.title('Распределение кластеров по группам эффективности')
        plt.xlabel('Группа эффективности')
        plt.ylabel('Количество кластеров')
        plt.tight_layout()
        plt.show()
    else:
        print("Недостаточно групп эффективности для столбчатой диаграммы распределения кластеров.")
    
    # 5. Среднее количество исполнителей в кластерах по группам эффективности
    merged_df = pd.merge(df_clusters, efficiency_df, left_on='cluster_label', right_on='original_cluster',
                         how='inner')  # Также используем inner
    if not merged_df.empty:
        actors_per_cluster = merged_df.groupby(['original_cluster', 'efficiency_group'])[
            'actor_name'].nunique().reset_index()
        if not actors_per_cluster.empty:
            avg_actors_per_cluster_in_efficiency_group = actors_per_cluster.groupby('efficiency_group')[
                'actor_name'].mean()
            if not avg_actors_per_cluster_in_efficiency_group.empty and len(
                    avg_actors_per_cluster_in_efficiency_group) > 1:
                plt.figure(figsize=(8, 6))
                sns.barplot(x=avg_actors_per_cluster_in_efficiency_group.index,
                            y=avg_actors_per_cluster_in_efficiency_group.values,
                            palette='plasma')
                plt.title('Среднее количество исполнителей в кластерах по группам эффективности')
                plt.xlabel('Группа эффективности')
                plt.ylabel('Среднее количество исполнителей')
                plt.tight_layout()
                plt.show()
            else:
                print(
                    "Недостаточно данных для столбчатой диаграммы среднего количества исполнителей по группам эффективности.")
        else:
            print("После агрегации 'actors_per_cluster' пуст.")
    else:
        print("Объединенный DataFrame (merged_df) для среднего количества исполнителей пуст.")
