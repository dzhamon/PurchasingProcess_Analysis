# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class KPIVisualizer:
    def __init__(self, df_kpi):
        self.df_kpi = df_kpi
        print('DF_KPY ', df_kpi.describe())
        print(self.df_kpi.columns)

    def plot_bar_chart(self):
        """
        Бар-чарт для сравнения KPI сотрудников по дисциплинам.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='actor_name', y='kpi_score', hue='discipline', data=self.df_kpi)
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
        discipline_kpi = self.df_kpi.groupby('discipline')['kpi_score'].sum().reset_index()
        plt.figure(figsize=(8, 8))
        plt.pie(discipline_kpi['kpi_score'], labels=discipline_kpi['discipline'], autopct='%1.1f%%', startangle=140)
        plt.title('Распределение KPI по дисциплинам')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
    
    def plot_heatmap(self):
        """
        Тепловая карта для анализа KPI сотрудников по дисциплинам.
        """
        output_dir = 'D:\Analysis-Results\heatmaps_by_discipline'
        os.makedirs(output_dir, exist_ok=True)
        print('Мы в методе plot_heatmap')
        
        # Получаем уникальные дисциплины
        disciplines = self.df_kpi['discipline'].unique()
        
        # Итерация по дисциплинам
        for discipline in disciplines:
            # Фильтруем данные для текущей дисциплины
            df_discipline = self.df_kpi[self.df_kpi['discipline'] == discipline]
            
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
        Линейный график для отображения динамики KPI.
        Для этого необходимо иметь временную компоненту в данных. Если у вас нет такой информации, этот график может быть не применим.
        """
        # Строим линейный график
        if 'close_date' in self.df_kpi.columns:
            print('SECOND ', self.df_kpi)
            # Конвертируем 'close_date' в формат datetime, если это не так
            self.df_kpi['close_date'] = pd.to_datetime(self.df_kpi['close_date'])
            
            # Создаем новый столбец 'month' на основе даты закрытия
            self.df_kpi['month'] = self.df_kpi['close_date'].dt.to_period('M').astype(str)
            
            # Группируем данные по месяцам и сотрудникам
            self.df_grouped = self.df_kpi.groupby(['month', 'actor_name'])['kpi_score'].sum().reset_index()
            print(self.df_grouped.head())
            
            # Выведем часть данных для проверки перед построением графика
            print('Часть сгруппированных данных ', self.df_grouped.head())
            
            # Построение линейного графика с группировкой по сотрудникам
            # Построение графика с линиями между точками
            sns.lineplot(x='month', y='kpi_score', hue='actor_name', data=self.df_grouped, marker='o',
                         style='actor_name', legend='full')
            
            plt.title('Динамика KPI по месяцам')
            plt.xlabel('Месяц')
            plt.ylabel('KPI Score')
            plt.xticks(rotation=45)
            
            # Располагаем легенду сверху справа
            plt.legend(title='Сотрудник', loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.show()
        else:
            print("Не хватает необходимых столбцов для построения графика.")