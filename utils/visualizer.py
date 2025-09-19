# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class KPIVisualizer:
    def __init__(self, df_kpi_normalized, df_kpi_monthly, OUT_DIR):
        self.df_kpi_normalized = df_kpi_normalized
        self.df_kpi_monthly = df_kpi_monthly
        self.OUT_DIR = OUT_DIR
        print('DF_KPY ', df_kpi_normalized.describe())
        print(self.df_kpi_normalized.columns)

    def plot_bar_chart(self):
        """
        Бар-чарт для сравнения KPI сотрудников по дисциплинам.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='actor_name', y='kpi_score', hue='discipline', data=self.df_kpi_normalized)
        plt.title('KPI сотрудников по дисциплинам')
        plt.xlabel('Сотрудник')
        plt.ylabel('KPI Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_pie_chart(self):
        """
        Круговая диаграмма для распределения суммарных KPI по дисциплинам.
        """
        discipline_kpi = self.df_kpi_normalized.groupby('discipline')['kpi_score'].sum().reset_index()
        plt.figure(figsize=(8, 8))
        plt.pie(discipline_kpi['kpi_score'], labels=discipline_kpi['discipline'], autopct='%1.1f%%', startangle=140)
        plt.title('Распределение KPI по дисциплинам')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
    
    def plot_heatmap(self):
        """
        Тепловая карта для анализа KPI сотрудников по дисциплинам.
        """
        output_dir = os.path.join(self.OUT_DIR, 'heatmaps_by_discipline')
        os.makedirs(output_dir, exist_ok=True)
        print('Мы в методе plot_heatmap')
        
        # Получаем уникальные дисциплины
        disciplines = self.df_kpi_normalized['discipline'].unique()
        
        # Итерация по дисциплинам
        for discipline in disciplines:
            # Фильтруем данные для текущей дисциплины
            df_discipline = self.df_kpi_normalized[self.df_kpi_normalized['discipline'] == discipline]
            
            # Пивотируем данные для тепловой карты
            heatmap_data = df_discipline.pivot(index='actor_name', columns='discipline', values='kpi_score')
            
            # Проверяем, есть ли данные для текущей дисциплины
            if heatmap_data.empty:
                print(f"Нет данных для дисциплины: {discipline}")
                continue
            
            # Настройка размера под A4
            plt.figure(figsize=(20, 10))  # Пропорции A4
            ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar=True, linewidths=0.5)
            plt.title(f"Тепловая карта KPI для дисциплины: {discipline}", fontsize=12, pad=20)
            plt.xlabel("Дисциплина", fontsize=12)
            plt.ylabel("Сотрудник", fontsize=10)
            
            # Добавление kpi_score внутри строк графика
            for y, actor_name in enumerate(heatmap_data.index):
                kpi_value = df_discipline.loc[df_discipline['actor_name'] == actor_name, 'kpi_score'].values[0]
                ax.text(
                    0.5, y + 0.5,  # Координаты: центр строки
                    f"{kpi_value:.2f}",
                    ha='center', va='center', fontsize=10, color="red", weight="normal"
                )
            
            # Сохраняем график
            output_path = os.path.join(output_dir, f"heatmap_{discipline}.png")
            plt.tight_layout()  # Устраняет проблемы с выравниванием
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Тепловая карта для дисциплины '{discipline}' сохранена в {output_dir}.")
    
    def plot_line_chart(self):
        """
        Создает отдельные линейные графики для каждой дисциплины и сохраняет их в PNG.
        """
        output_dir = os.path.join(self.OUT_DIR, "line_charts_by_discipline")
        os.makedirs(output_dir, exist_ok=True)

        # Получаем список уникальных дисциплин
        disciplines = self.df_kpi_monthly['discipline'].unique()

        if len(disciplines) > 1:
            print(f"Обнаружено {len(disciplines)} дисциплин. Будут созданы отдельные графики.")
        
        # Создаем и сохраняем график для каждой дисциплины
        for discipline in disciplines:
            # Фильтруем DataFrame по текущей дисциплине
            df_discipline = self.df_kpi_monthly[self.df_kpi_monthly['discipline'] == discipline]
            
            # Если в дисциплине больше 1 сотрудника, строим график
            if len(df_discipline['actor_name'].unique()) > 1:
                plt.figure(figsize=(15, 8))
                
                sns.lineplot(
                    x="month",
                    y="kpi_score",
                    hue="actor_name",
                    data=df_discipline,
                    marker="o",
                )

                plt.title(f"Динамика KPI по месяцам: {discipline}")
                plt.xlabel("Месяц")
                plt.ylabel("KPI Score")
                plt.xticks(rotation=45)
                plt.legend(title="Сотрудник", bbox_to_anchor=(1.05, 1), loc="upper left")
                
                plt.tight_layout()
                
                # Формируем имя файла (например, "kpi_reports/Дисциплина_Инженерный.png")
                filename = os.path.join(output_dir, f"{discipline}.png")
                
                # Сохраняем график в файл PNG
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close() # Закрываем фигуру, чтобы не перегружать память
                
                print(f"График для дисциплины '{discipline}' сохранен в файл: {filename}")
            else:
                print(f"В дисциплине '{discipline}' недостаточно данных для линейного графика (меньше 2 сотрудников).")
        
        