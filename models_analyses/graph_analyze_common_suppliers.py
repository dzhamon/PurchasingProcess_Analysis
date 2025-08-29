import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
from PyQt5.QtWidgets import QMessageBox


def analyze_and_visualize_suppliers(parent_widget, df):
    """
    Анализирует и визуализирует связи между дисциплинами и общими поставщиками,
    а также оценивает ценовую волатильность для каждого проекта в DataFrame.

    :param df: Исходный DataFrame со всеми данными, может содержать несколько проектов.
    """
    if df.empty:
        QMessageBox.warning(
            parent_widget, "Ошибка", "DataFrame пуст. Нет данных для анализа."
        )
        return

    # 1. Получение списка уникальных проектов
    unique_projects = df["project_name"].unique()
    if len(unique_projects) == 0:
        QMessageBox.warning(parent_widget, "Ошибка", "Не найдено проектов в DataFrame.")
        return

    output_folder = os.path.join(os.getcwd(), "network_graphs")
    os.makedirs(output_folder, exist_ok=True)
    
    # Создаем файл excel для записи результатов
    excel_output_path = os.path.join(os.getcwd(), "project_analysis_results.xlsx")
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        for project_name in unique_projects:
            print(f"Анализ проекта: {project_name}")
    
            # 2. Фильтрация данных по текущему проекту
            project_df = df[df["project_name"] == project_name].copy()
    
            # 3. Выявление общих поставщиков внутри проекта
            discipline_supplier_counts = project_df.groupby("winner_name")[
                "discipline"
            ].nunique()
            common_suppliers = discipline_supplier_counts[
                discipline_supplier_counts > 1
            ].index.tolist()
    
            if not common_suppliers:
                print(
                    f"В проекте '{project_name}' не найдено поставщиков, работающих более чем с одной дисциплиной."
                )
                continue
    
            # 4. Расчет волатильности цен по исходной валюте
            supplier_volatility = {}
            for supplier in common_suppliers:
                supplier_data = project_df[project_df["winner_name"] == supplier]
                volatility_by_currency = (
                    supplier_data.groupby("currency")["total_price"]
                    .agg(["mean", "std"])
                    .fillna(0)
                )
    
                volatility_by_currency["cv"] = np.where(
                    volatility_by_currency["mean"] != 0,
                    volatility_by_currency["std"] / volatility_by_currency["mean"],
                    0,
                )
                supplier_volatility[supplier] = volatility_by_currency["cv"].max()
            
            # Сбор данных для Excel
            project_analysis_data = []
            for supplier in common_suppliers:
                # Находим все дисциплины для данного поставщика в проекте
                disciplines = project_df[project_df['winner_name'] == supplier]['discipline'].unique().tolist()
                project_analysis_data.append({
                    'Поставщик': supplier,
                    'Коэффициент волатильности (CV)': supplier_volatility.get(supplier, 0),
                    'Связанные дисциплины': ", ".join(disciplines) # Соединяем дисциплины в одну строку
                })
            # создаем DataFrame из собранных данных
            if project_analysis_data:
                analysis_df = pd.DataFrame(project_analysis_data)
                # Записываем DataFrame на новый лист в Excel-файле
                save_sheet_name = project_name[:31]
                analysis_df.to_excel(writer, sheet_name=save_sheet_name, index=False)
    
            # 5. Построение графа только для общих связей
            G = nx.Graph()
            for _, row in project_df.iterrows():
                discipline = row["discipline"]
                supplier = row["winner_name"]
    
                if supplier in common_suppliers:
                    G.add_node(discipline, type="discipline", color="green")
                    G.add_node(
                        supplier,
                        type="supplier",
                        volatility=supplier_volatility.get(supplier, 0),
                    )
                    G.add_edge(discipline, supplier)
                    
            # Constants for vizualization
            FONT_SIZE = 6
            NODE_SIZE_DISCIPLINE = 600
            NODE_SIZE_SUPPLIER_BASE = 150
            NODE_SIZE_SUPPLIER_SCALE = 1500
            MAX_VOLATILITY_FOR_SCALING = 0.5  # Cap the volatility scale to prevent huge nodes
    
            # 6. Визуализация графа
            plt.figure(figsize=(20, 15))
            try:
                pos = nx.spring_layout(G, k=0.15, iterations=100)
            except Exception as e:
                print(f"Ошибка при вычислении spring_layout: {e}. Переключаемся на kamada_kawai.")
                pos = nx.kamada_kawai_layout(G)
    
            # Определение цветов и размеров узлов
            node_colors = []
            node_sizes = []
            labels = {}
            
            for node, data in G.nodes(data=True):
                if data['type'] == 'discipline':
                    node_colors.append('#00A86B') # "lightgreen"
                    node_sizes.append(NODE_SIZE_DISCIPLINE)
                    labels.update({node: node})
                elif data["type"] == "supplier":
                    volatility_value = data.get("volatility", 0)
                    scaled_volatility = min(volatility_value, MAX_VOLATILITY_FOR_SCALING)
                    node_sizes.append(NODE_SIZE_SUPPLIER_BASE + scaled_volatility * NODE_SIZE_SUPPLIER_SCALE)
                
                    # Цвета в зависимости от волатильности (от синего к красному
                    color_map = plt.cm.get_cmap("YlOrRd")
                    node_colors.append(color_map(scaled_volatility))
                    labels[node] = f'{node}\nCV: {volatility_value:.2f}'
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
            nx.draw_networkx_edges(G, pos, edge_color="#606060", alpha=0.8)
            nx.draw_networkx_labels(
                G, pos, labels=labels, font_size=FONT_SIZE, font_color="#222222"
            )
            
            # Добавляем легенду
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Дисциплина",
                    markersize=15,
                    markerfacecolor="#00A86B",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Поставщик (размер и цвет - волатильность)",
                    markersize=15,
                    markerfacecolor="yellow",
                )
            ]
            plt.legend(handles=legend_handles, title="Тип узла", loc="upper right", fontsize=12)
            
            plt.title(
                f"Анализ связей и волатильности у общих поставщиков для проекта: {project_name}",
                fontsize=18,
                pad=20,
            )
            
            file_name = f"common_suppliers_network_{project_name}.png"
            file_path = os.path.join(output_folder, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            
    QMessageBox.information(
        parent_widget,
        "Сообщение",
        f"Анализ завершен. Графики для каждого проекта сохранены в папке:\n{output_folder}",
    )
    return
