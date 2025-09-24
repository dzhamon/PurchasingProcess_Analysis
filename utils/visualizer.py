# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from PyQt5.QtWidgets import QMessageBox

class KPIVisualizer:
    def __init__(self, df_kpi_normalized, df_kpi_monthly, report_dir):
        self.df_kpi_normalized = df_kpi_normalized
        self.df_kpi_monthly = df_kpi_monthly
        self.report_dir = report_dir
        print('DF_KPY ', df_kpi_normalized.describe())
        print(self.df_kpi_normalized.columns)
    
    def plot_bar_chart(self):
        """
        Создает бар-чарт для сравнения KPI сотрудников.
        Отображает топ-10 лучших и топ-10 худших исполнителей.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
    
        # Сортируем DataFrame по kpi_score
        df_sorted = self.df_kpi_normalized.sort_values(by="kpi_score", ascending=False)
    
        # Выбираем топ-10 лучших и топ-10 худших
        df_top_10 = df_sorted.head(10)
        df_bottom_10 = df_sorted.tail(10)
    
        # Объединяем их в один DataFrame для построения графика
        df_to_plot = pd.concat([df_top_10, df_bottom_10])
    
        plt.figure(figsize=(15, 10))
    
        # Строим бар-чарт
        sns.barplot(
            x="actor_name",
            y="kpi_score",
            hue="discipline",
            data=df_to_plot,
            palette="viridis",  # Используем более приятную палитру
        )
    
        plt.title(
            f"Топ-10 лучших и худших исполнителей по общему KPI"
        )
        plt.xlabel("Сотрудник")
        plt.ylabel("KPI Score (нормализованное значение)")
        plt.xticks(rotation=45, ha="right")  # Улучшаем читаемость подписей
    
        # Размещаем легенду за пределами графика
        plt.legend(title="Дисциплина", bbox_to_anchor=(1.05, 1), loc="upper left")
    
        plt.tight_layout()
        plt.show()
    
        # --- Сохраняем график в файл ---
        output_dir = os.path.join(self.report_dir, "overall_kpi_charts")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"Overall_KPI_Chart_xxx.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Общий график KPI сохранён в файл: {filename}")


    def plot_heatmap(self):
        """
        Тепловая карта для анализа KPI сотрудников по дисциплинам.
        """
        print('Мы в методе plot_heatmap')
        
        out_dir = os.path.join(self.report_dir, 'Тепловые карты по дисциплинам')
        os.makedirs(out_dir, exist_ok=True)
        
        # Получаем уникальные дисциплины
        disciplines = self.df_kpi_normalized['discipline'].unique()
        
        # Итерация по дисциплинам
        for discipline in disciplines:
            # Фильтруем данные для текущей дисциплины
            df_discipline = self.df_kpi_normalized[self.df_kpi_normalized['discipline'] == discipline]
            
            # Пивотируем данные для тепловой карты
            heatmap_data = df_discipline.pivot_table(
                 index="actor_name",
                 columns="discipline",
                 values="kpi_score",
                 aggfunc="mean",
             )
            
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
            file_name = os.path.join(out_dir, f"heatmap_{discipline}.png")
            plt.tight_layout()  # Устраняет проблемы с выравниванием
            plt.savefig(file_name, dpi=300)
            plt.close()
            print(f"Тепловая карта для дисциплины '{discipline}' сохранена в {out_dir}.")
    
    def plot_line_chart(self):
        """
        Создает отдельные линейные графики для каждой дисциплины и сохраняет их в PNG.
        """
        out_dir = os.path.join(self.report_dir, "Линейные графики по дисциплинам по дисциплинам")
        os.makedirs(out_dir, exist_ok=True)
        
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

                plt.title(f" - Динамика KPI по месяцам: {discipline}")
                plt.xlabel("Месяц")
                plt.ylabel("KPI Score")
                plt.xticks(rotation=45)
                plt.legend(title="Сотрудник", bbox_to_anchor=(1.05, 1), loc="upper left")
                
                plt.tight_layout()
                
                # Формируем имя файла
                filename = os.path.join(out_dir, f"{discipline}.png")
                
                # Сохраняем график в файл PNG
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close() # Закрываем фигуру, чтобы не перегружать память
                
                print(f"График для дисциплины '{discipline}' сохранен в файл: {out_dir}")
            else:
                print(f"В дисциплине '{discipline}' недостаточно данных для линейного графика (меньше 2 сотрудников).")
        
   