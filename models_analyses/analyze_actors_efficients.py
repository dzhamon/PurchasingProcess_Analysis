""" Модуль Анализ активности и эффективности исполнителей Лотов """

import os
import pandas as pd
from utils.config import BASE_DIR
import matplotlib.pyplot as plt

class AnalyzeActorsEfficients:
    def __init__(self, df):
        self.df = df.copy()
        self.analysis_df = None

        # Создаем единую базовую директорию для всех результатов анализа
        self.BASE_OUT_DIR = os.path.join(BASE_DIR, 'Supplier-Frequency')
        os.makedirs(self.BASE_OUT_DIR, exist_ok=True)

        self.SUPPLIER_BEHAVIOR_DIR = os.path.join(self.BASE_OUT_DIR, 'Supplier-Behavior')
        os.makedirs(self.SUPPLIER_BEHAVIOR_DIR, exist_ok=True)

    
    def analyze_supplier_frequency(self):
        # Выводим в список уникальные наименования Проектов
        set_projects = self.df['project_name'].unique()
        
        # Строим цикл по наименованиям Проектов
        for project in set_projects:
            # фильтруем DataFrame по текущему проекту
            df_project = self.df[self.df['project_name'] == project]
            # Группировка данных
            grouped_df = (
                df_project.groupby(["discipline", "actor_name", "winner_name"])
                .size()
                .reset_index(name="win_count")
            )
        
            # Добавляем общее количество закупок по (discipline, actor_name)
            total_counts = grouped_df.groupby(["discipline", "actor_name"])[
                "win_count"
            ].transform("sum")
            grouped_df["win_percentage"] = (grouped_df["win_count"] / total_counts * 100).round(
                2
            )
        
            # Сортировка данных
            top_suppliers = grouped_df.sort_values(
                by=["discipline", "actor_name", "win_count"], ascending=[True, True, False]
            )
            
            # Создадим директорию для результатов по текущему Проекту
            project_dir = os.path.join(self.BASE_OUT_DIR, f"{project}")
            os.makedirs(project_dir, exist_ok=True)
        
            # Сохранение в Excel с дополнительными метриками
            grouped_data = top_suppliers.groupby("discipline")
            for discipline, group_discipline in grouped_data:
                discipline_dir = os.path.join(project_dir, discipline.replace(" ", "_"))
                os.makedirs(discipline_dir, exist_ok=True)
        
                excel_path = os.path.join(
                    discipline_dir, f"{discipline.replace(' ', '_')}_suppl_fcy.xlsx"
                )
                
                # Сохраняем данные конкретной дисциплины в свой файл
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    group_discipline.to_excel(writer, index=False)
        
                # Дополнительный график: топ-10 поставщиков в дисциплине
                top_10 = group_discipline.groupby("winner_name")["win_count"].sum().nlargest(10)
                if not top_10.empty:
                    fig, ax = plt.subplots(figsize=(14, 10))
                    top_10.plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_title(f"Top 10 Suppliers in {discipline}")
                    ax.set_ylabel("Total Wins")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(discipline_dir, f"top_10_suppliers_{discipline}.png")
                    )
                    plt.close()
        return None

    def analyze_supplier_behavior(self):
    
        # Убедимся, что столбцы с датами имеют правильный формат
        self.df["open_date"] = pd.to_datetime(self.df["open_date"])
        self.df["close_date"] = pd.to_datetime(self.df["close_date"])
    
        # Рассчитываем продолжительность торгов в днях
        self.df["duration_days"] = (self.df["close_date"] - self.df["open_date"]).dt.days
    
        # Создаем уникальное имя файла
        file_path = os.path.join(self.SUPPLIER_BEHAVIOR_DIR, "bidding_duration_analysis.xlsx")
    
        # Группируем данные, чтобы получить среднюю продолжительность для каждой пары (actor, winner)
        grouped_duration = (
            self.df.groupby(["discipline", "actor_name", "winner_name"])["duration_days"]
            .agg(["count", "mean"])
            .reset_index()
        )
        grouped_duration.columns = [
            "discipline",
            "actor_name",
            "winner_name",
            "win_count",
            "average_duration",
        ]
    
        # Рассчитываем общую среднюю продолжительность торгов по каждой дисциплине
        discipline_avg = self.df.groupby("discipline")["duration_days"].mean().reset_index()
        discipline_avg.columns = ["discipline", "overall_avg_duration"]
    
        # Объединяем два датафрейма для сравнения
        analysis_df = pd.merge(grouped_duration, discipline_avg, on="discipline")
    
        # Рассчитываем разницу в продолжительности торгов
        analysis_df["duration_difference"] = (
            analysis_df["average_duration"] - analysis_df["overall_avg_duration"]
        )
    
        # Сортируем по разнице, чтобы найти самые большие отклонения
        analysis_df = analysis_df.sort_values(by="duration_difference", ascending=True)
    
        # Сохраняем результат в Excel
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            analysis_df.to_excel(writer, index=False, sheet_name="Duration Analysis")
    
            # Форматирование для удобства
            workbook = writer.book
            worksheet = writer.sheets["Duration Analysis"]
            header_format = workbook.add_format({"bold": True, "text_wrap": True})
    
            for col_num, value in enumerate(analysis_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
    
        print(f"Анализ продолжительности торгов сохранен в {file_path}")
        return None

