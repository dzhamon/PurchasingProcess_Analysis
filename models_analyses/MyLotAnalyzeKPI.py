import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.functions import CurrencyConverter


class LotAnalyzeKPI:
    def __init__(self, df, weights, OUT_DIR, analysis_type):
        """
        Инициализация с DataFrame, содержащим данные по лотам.
        """
        self.df = df
        self.unique_disciplines = self.df["discipline"].unique()
        self.weights = weights
        self.OUT_DIR = OUT_DIR
        self.analysis_type = analysis_type
        self.successful_statuses = ["Конкурс завершен. Направлен в отдел АД"]
        

    def calculate_kpi(self):
        """
        Рассчитывает итоговый KPI для каждого исполнителя по дисциплинам с использованием pandas.
        """
        # 1. Пересчет стоимости лотов в EUR
        converter = CurrencyConverter()
        columns_info = [("total_price", "currency", "total_price_eur")]
        df_converted = converter.convert_multiple_columns(
            self.df, columns_info=columns_info
        )

        self.df["total_price_eur"] = df_converted["total_price_eur"].copy()

        # 2. Группировка данных по дисциплине и исполнителю
        if self.analysis_type == 'single_project':
            # группируем только по исполнителю и дисциплине
            grouped = self.df.groupby(['discipline', 'actor_name'])
        else:
            # Для нескольких проектов добавляем 'project_name' в группировку
            grouped = self.df.groupby(["project_name", "discipline", "actor_name"])

        # Преобразуем даты в формат datetime
        self.df["open_date"] = pd.to_datetime(self.df["open_date"])
        self.df["close_date"] = pd.to_datetime(self.df["close_date"])

        # Рассчитаем время закрытия  Лотов в днях
        self.df["time_to_close"] = (
            self.df["close_date"] - self.df["open_date"]
        ).dt.days

        # 3. Агрегация данных для расчета KPI
        kpi_data = grouped.agg(
            total_lots=("lot_number", "count"),
            sum_lot_value=("total_price_eur", "sum"),
            avg_time_to_close=("time_to_close", "mean"),
            close_date=("close_date", "first"),  # Берем первую дату закрытия для группы
        ).reset_index()

        kpi_data["normalized_lots"] = (
            kpi_data["total_lots"] - kpi_data["total_lots"].mean()
        ) / kpi_data["total_lots"].std()
        kpi_data["normalized_value"] = (
            kpi_data["sum_lot_value"] - kpi_data["sum_lot_value"].mean()
        ) / kpi_data["sum_lot_value"].std()
        kpi_data["normalized_time"] = (
            -(kpi_data["avg_time_to_close"] - kpi_data["avg_time_to_close"].mean())
            / kpi_data["avg_time_to_close"].std()
        )
        # Заменим NaN на 0 в нормализованных столбцах, если стандартное отклонение равно 0
        kpi_data.fillna(0, inplace=True)

        # 4. Расчет итогового KPI на основе нормализованных данных
        # # Используем только 3 показателя, избегая двойного учёта стоимости
        kpi_data["kpi_score"] = (
            self.weights.get("lots", 0) * kpi_data["normalized_lots"]
            + self.weights.get("value", 0) * kpi_data["normalized_value"]
            + self.weights.get("time", 0) * kpi_data["normalized_time"]
        )

        # 5. Финальная нормализация для удобства представления (например, 0 до 1)
        df_kpi_normalized = self.normalize_kpi_table(kpi_data)

        print(df_kpi_normalized)
        max_kpi_score = df_kpi_normalized["kpi_score"].max()
        print(f"Максимальный KPI: {max_kpi_score}")
        min_kpi_score = df_kpi_normalized["kpi_score"].min()
        print(f"Минимальный KPI: {min_kpi_score}")

        return df_kpi_normalized

    def normalize_kpi_table(self, df):
        # Эта функция будет нормировать kpi_score к диапазону 0-1
        max_val = df["kpi_score"].max()
        min_val = df["kpi_score"].min()
        if max_val == min_val:
            df["kpi_score_normalized"] = 0
        else:
            df["kpi_score_normalized"] = (df["kpi_score"] - min_val) / (
                max_val - min_val
            )
        return df

    def calculate_monthly_kpi(self):
        """
        Рассчитывет ежемесячный KPI для каждого исполнителя
        """
        # 1. Подготовка данных: создаем '' и 'is_successful'
        self.df["open_date"] = pd.to_datetime(self.df["open_date"])
        self.df["close_date"] = pd.to_datetime(self.df["close_date"])
        self.df["month"] = self.df["close_date"].dt.to_period("M").astype(str)
        self.df["time_to_close"] = (self.df["close_date"] - self.df["open_date"]).dt.days
        self.df["is_successful"] = (
            self.df["lot_status"].isin(self.successful_statuses).astype(int)
        )

        # 2. Пересчет стоимости лотов в EUR
        converter = CurrencyConverter()
        columns_info = [("total_price", "currency", "total_price_eur")]
        df_converted = converter.convert_multiple_columns(
            df=self.df, columns_info=columns_info
        )
        self.df["total_price_eur"] = df_converted["total_price_eur"].copy()

        # 3. Группировка данных
        if self.analysis_type == 'single_project':
            grouped = self.df.groupby(["month", "discipline", "actor_name"])
        else:
            grouped = self.df.groupby(["month", "project_name", "discipline", "actor_name"])

        # 4. Агрегация данных
        kpi_data = grouped.agg(
            total_lots=("lot_number", "count"),
            avg_lot_value=("total_price_eur", "mean"),
            avg_time_to_close=("time_to_close", "mean"),
            lot_success_rate=("is_successful", "mean"),
        ).reset_index()

        # 5. Нормализация и расчет KPI
        # Нормализация Z-score для каждого показателя
        kpi_data["normalized_lots"] = (
            kpi_data["total_lots"] - kpi_data["total_lots"].mean()
        ) / kpi_data["total_lots"].std()
        kpi_data["normalized_value"] = (
            kpi_data["avg_lot_value"] - kpi_data["avg_lot_value"].mean()
        ) / kpi_data["avg_lot_value"].std()
        kpi_data["normalized_time"] = (
            -(kpi_data["avg_time_to_close"] - kpi_data["avg_time_to_close"].mean())
            / kpi_data["avg_time_to_close"].std()
        )
        kpi_data["normalized_success"] = (
            kpi_data["lot_success_rate"] - kpi_data["lot_success_rate"].mean()
        ) / kpi_data["lot_success_rate"].std()

        kpi_data.fillna(0, inplace=True)

        kpi_data["kpi_score"] = (
            self.weights.get("lots", 0) * kpi_data["normalized_lots"]
            + self.weights.get("value", 0) * kpi_data["normalized_value"]
            + self.weights.get("time", 0) * kpi_data["normalized_time"]
            + self.weights.get("success", 0) * kpi_data["normalized_success"]
        )

        return kpi_data

    def lots_per_actor(self):
        """
        Возвращает количество лотов, обработанных каждым исполнителем внутри своей Дисциплины.
        """
        # 1. Создаем словарь, который будет хранить информацию - Дисциплина - исполнитель- его количество Лотов
        lots_per_actor = {}

        # 2. Проходим по каждой дисциплине
        for discipline in self.unique_disciplines:
            # Фильтруем DataFrame по текущей дисциплине
            df_filtered = self.df[self.df["discipline"] == discipline].copy()

            if not df_filtered.empty:
                # Группировка по 'actor_name', подсчет лотов для каждого исполнителя
                lots_count = (
                    df_filtered.groupby("actor_name")["lot_number"].count().to_dict()
                )

                # Добавляем информацию в словарь, где ключ — дисциплина, а значение — словарь с количеством лотов по исполнителям
                lots_per_actor[discipline] = lots_count
            else:
                lots_per_actor[discipline] = {}

        # Теперь lots_per_actor содержит данные для визуализации
        return lots_per_actor

    def avg_time_to_close(self):
        """
        Рассчитывает среднее время обработки лота для каждого исполнителя.
        """
        avg_time_per_discipline = {}  # Инициализация словаря для сохранения данных

        for discipline in self.unique_disciplines:
            # Фильтруем по дисциплине
            df_filtered = self.df[self.df["discipline"] == discipline].copy()

            # Проверяем, что фильтр не вернул пустой DataFrame
            if not df_filtered.empty:
                # Вычисляем время обработки
                df_filtered.loc[:, "processing_time"] = pd.to_datetime(
                    df_filtered.loc[:, "close_date"]
                ) - pd.to_datetime(df_filtered.loc[:, "open_date"])

                # Группируем по исполнителю и вычисляем среднее время обработки
                avg_processing_time = df_filtered.groupby("actor_name")[
                    "processing_time"
                ].mean()
                # Сохраняем результаты в словарь
                avg_time_per_discipline[discipline] = avg_processing_time
            else:
                avg_time_per_discipline[discipline] = {}
        return avg_time_per_discipline

    def get_seed_data(self, df_kpi):
        """
        Выбирает лучших исполнителей (Seed Data) на основе максимального KPI по каждой дисциплине.
        """
        # Группируем данные по дисциплинам и выбираем исполнителей с максимальным KPI
        seed_data = df_kpi.loc[df_kpi.groupby("discipline")["kpi_score"].idxmax()]
        print("Это seed_data")
        print(seed_data.columns)
        print(seed_data)
        print("Количество уникальных дисциплин в seed_data")
        print(
            seed_data["discipline"].nunique()
        )  # Количество уникальных дисциплин в seed_data
        return seed_data
