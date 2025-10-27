import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

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

    # 1. Подготовка данных: Преобразование PeriodIndex в DatetimeIndex для SARIMA
    df_series = df_pivot_eur[category].rename('y').to_frame().copy()
    # to_timestamp(how='start') преобразует '2024-01' в '2024-01-01 00:00:00'
    df_series.index = df_series.index.to_timestamp(how='start')

    # 2. Моделирование SARIMA (Стандартные параметры для месячных данных)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    try:
        model = SARIMAX(
            df_series['y'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        print(f"✅ Модель SARIMA обучена для {category}.")

    except Exception as e:
        print(f"❌ Ошибка при обучении модели SARIMA для {category}: {e}")
        return pd.DataFrame(), ""

    # 3. Генерация прогноза
    forecast = results.get_forecast(steps=forecast_months)
    forecast_df = forecast.summary_frame(alpha=0.05)

    forecast_df.columns = ['Прогноз (EUR)', 'Ошибка (std err)', 'Нижняя граница 95%', 'Верхняя граница 95%']

    # 4. Визуализация и сохранение графика
    plot_filename = os.path.join(OUT_DIR, f"6_forecast_plot_{category.replace(' ', '_').replace('/', '_')}_{timestamp}.png")

    # Функция для форматирования оси Y
    def format_currency(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f}M €'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K €'
        else:
            return f'{x:.0f} €'

    plt.figure(figsize=(10, 5))

    # Исторические данные
    plt.plot(df_series.index, df_series['y'], label='Исторические данные', color='blue')

    # Прогноз
    plt.plot(forecast_df.index, forecast_df['Прогноз (EUR)'], label='Прогноз SARIMA', color='red', linestyle='--')

    # Доверительный интервал
    plt.fill_between(forecast_df.index,
                     forecast_df['Нижняя граница 95%'],
                     forecast_df['Верхняя граница 95%'],
                     color='pink', alpha=0.3, label='95% Доверительный интервал')

    plt.title(f'6. Прогноз месячных затрат для: {category}')
    plt.xlabel('Дата')
    plt.ylabel('Сумма контрактов (EUR)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    return forecast_df[['Прогноз (EUR)', 'Нижняя граница 95%', 'Верхняя граница 95%']], os.path.abspath(plot_filename)