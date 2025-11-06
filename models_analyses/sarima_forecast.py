import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
from scipy.stats import zscore

def run_sarima_forecast(df_pivot_eur: pd.DataFrame, category: str, OUT_DIR: str, timestamp: str, forecast_months: int = 12) -> tuple[pd.DataFrame, str]:
    """
    Выполняет прогнозирование месячных затрат для одной дисциплины с помощью SARIMA.

    Аргументы:
        df_pivot_eur (pd.DataFrame): DataFrame с затратами по дисциплинам (индекс - year_month).
        category (str): Название дисциплины для анализа.
        OUT_DIR (str): Путь к папке для сохранения результатов.
        timestamp (str): Временная метка для имен файлов.
        forecast_months (int): Количество месяцев для прогнозирования.

    Возвращает:
        tuple[pd.DataFrame, str]: DataFrame с прогнозом и путь к сохраненному графику.
    """

    # Отфильтровываем только нужную категорию (ts --> TimeSeries, внутренний параметр SARIMA)
    ts = df_pivot_eur[category].dropna()
    ts_prov = ts.copy()

    # Находим индекс последнего ненулевого значения
    last_valid_index = ts[ts != 0].index.max()

    if last_valid_index is None or pd.isna(last_valid_index):
        print(f"❌ Ряд {category} содержит только нулевые значения. Прогноз невозможен.")
        return pd.DataFrame(), ""

    ts_smooth = ts.loc[:last_valid_index].copy() # рабочая копия ряда

    # Расчет целевых показателей
    non_zero_values = ts_smooth[ts_smooth != 0]

    if non_zero_values.empty:
        print(f"❌ Ряд {category} не содержит ненулевых значений для сглаживания.")
        return pd.DataFrame(), ""

    # Целевое значение для заполнения нулевых месяцев
    target_allocation = non_zero_values.median()

    # Порог для определения крупного "выброса" (например, в 3 раза больше медианы)
    surplus_threshold = target_allocation * 2
    #surplus_threshold = target_allocation *

    # Выполнение самосохраняющего сглаживания
    # Создаем "пул" для распределения
    allocation_pool = 0.0

    # Проход 1: Создание пула избытка (Сглаживание пиков)
    for i in range(len(ts_smooth)):
        value = ts_smooth.iloc[i]

        # Сценарий А. Если это месяц с крупной закупкой (создаем избыток)
        if value > surplus_threshold:
            # Избыток: разница между фактическим значением и порогом
            surplus = value - target_allocation
            allocation_pool += surplus
            ts_smooth.iloc[i] = target_allocation # Уменьшаем это значение до целевого

    print(f"✅ Проход 1 завершен. Общий пул для заполнения: {allocation_pool:.2f} EUR.")

    # Проход 2: Рандомизированное заполнение нулей из пула

    # идентификация месяцев для перераспределения
    zero_mask = (ts_smooth == 0)
    zero_count = zero_mask.sum()

    if allocation_pool > 0 and zero_count > 0:
        # 1. Определяем интервал случайных значений
        # Минимум: 1 (чтобы избежать log(0) и сделать месяц ненулевым)
        # Максимум: Медиана, или максимум, который можно выделить из пула

        # Максимальное значение, которое может получить один месяц,
        # чтобы остальные получили хотя бы 1 EUR, или просто Медиана.
        max_fill_value = min(target_allocation, allocation_pool - (zero_count - 1))

        # Если пул слишком мал, ставим минимум 1 EUR.
        if max_fill_value <= 1:
            max_fill_value = 1.0

        # 2. Генерируем случайные доли: мы распределяем весь allocation_pool
        # Создаем N (zero_count) случайных чисел, их сумма будет ~N/2 * max_fill_value.

        # Генерируем N случайных чисел в интервале [1, max_fill_value]
        random_fills = [random.uniform(1.0, max_fill_value) for _ in range(zero_count)]

        # 3. Нормализация (Масштабирование) для сохранения суммы!
        # Суммируем сгенерированные случайные числа
        sum_random_fills = sum(random_fills)

        # Коэффициент, который гарантирует, что сумма новых значений точно равна allocation_pool
        scale_factor = allocation_pool / sum_random_fills

        # Применяем коэффициент к случайным числам
        final_fills = [val * scale_factor for val in random_fills]

        print(f"ℹ️ Пул {allocation_pool:.2f} EUR распределен между {zero_count} нулями.")

        # 4. Вставляем значения обратно в ts_smooth
        ts_smooth[zero_mask] = final_fills

        # Пул израсходован
        allocation_pool = 0.0

    else:
        print("ℹ️ Пул пуст или нулевых месяцев для заполнения нет.")


    # Проверка сохранения суммы
    original_sum = ts.sum()
    imputed_sum = ts_smooth.sum()

    print(f"✅ Суммосохраняющее сглаживание завершено для {category}. Сумма сохранена.")

    # Логарифмирование сглаженного ряда (ts_smooth)
    ts_clean = ts_smooth.replace(0, 1) # Заменяем оставшиеся нули на 1 для логарифма
    ts_log = np.log(ts_clean)

    # Используем автоматический подбор параметров (pmdarima)
    from pmdarima import auto_arima

    print(f"⌛ Запуск автоподбора SARIMA для {category}...")

    # Рассчитываем Z-счет для логарифмированного ряда
    z_scores = zscore(ts_log)
    # Используем порог 3.0 (стандартный)
    threshold = 3.0
    outliers = np.abs(z_scores) > threshold

    if outliers.any():
        print(f"⚠️ В обрезанном ряду {category} найдено {outliers.sum()} выбросов (Z-счет > {threshold}).")
        # Заменяем выбросы медианным значением ряда
        median_val = ts_log[~outliers].median()
        ts_log[outliers] = median_val
        print("✅ Выбросы заменены медианным значением для стабилизации прогноза.")

    # 1. Автоматический подбор модели
    # m=12 для месячной сезонности
    # suppress_warnings=True для чистого вывода
    model_fit = auto_arima(
        ts_log,
        start_p=1, start_q=1,
        max_p=1, max_q=1,
        m=12,               # Сезонность (12 месяцев)
        d=0, D=0,           # Дифференцирование (можно оставить 1)
        start_P=0, start_Q=0,
        max_P=1, max_Q=1,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,      # Ускоряет процесс
        seasonal=True
    )

    print(f"✅ Оптимальная модель для {category}: {model_fit.get_params()['order']}x{model_fit.get_params()['seasonal_order']}")

    # 3. Генерация прогноза и доверительный интервал
    forecast_values, conf_int = model_fit.predict(
        n_periods=forecast_months,
        return_conf_int=True,
        alpha=0.05
    )

    # Создание DataFrame прогноза с датами
    last_date = ts_log.index.max()
    forecast_index = pd.date_range(start=last_date, periods=forecast_months + 1, freq='MS')[1:]

    forecast_df = pd.DataFrame(conf_int, index=forecast_index, columns=['Нижняя граница 95%', 'Верхняя граница 95%'])
    forecast_df['Прогноз (EUR)'] = forecast_values.values
    # Порядок столбцов
    forecast_df = forecast_df[['Прогноз (EUR)', 'Нижняя граница 95%', 'Верхняя граница 95%']]

    # Делогарифмирование исторических данных
    ts_exp = np.exp(ts_log)

    # Делогарифмирование прогнозных данных и доверительного интервала
    forecast_df_exp = np.exp(forecast_df)


    # Функция для форматирования оси Y
    def format_currency(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f}M €'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K €'
        else:
            return f'{x:.0f} €'

    # Визуализация и сохранение графика

    plot_filename = os.path.join(OUT_DIR, f"6_forecast_plot_{category.replace(' ', '_').replace('/', '_')}_{timestamp}.png")
    plt.figure(figsize=(10, 6))

    # Исторические данные
    plt.plot(ts_exp.index, ts_exp.values, label='Исторические данные', color='blue')

    # Прогнозные данные
    plt.plot(forecast_df_exp.index, forecast_df_exp['Прогноз (EUR)'], label='Прогноз SARIMA', color='red', linestyle='--')

    # Доверительный интервал
    plt.fill_between(forecast_df_exp.index,
                     forecast_df_exp['Нижняя граница 95%'],
                     forecast_df_exp['Верхняя граница 95%'],
                     color='pink', alpha=0.3, label='95% Доверительный интервал')

    plt.title(f'6. Прогноз месячных затрат для: {category}')
    plt.xlabel('Дата')
    plt.ylabel('Сумма контрактов (EUR)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
    # Устанавливаем максимальное значение, например, в 2 или 3 раза больше максимального прогноза
    max_visible_y = forecast_df_exp['Прогноз (EUR)'].max() * 3
    # Ограничиваем ось Y:
    plt.ylim(bottom=0, top=max_visible_y)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()


    return forecast_df_exp[['Прогноз (EUR)', 'Нижняя граница 95%', 'Верхняя граница 95%']], os.path.abspath(plot_filename)