# код SARIMA анализа временных рядов (на примере контрактных данных)

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from utils.functions import CurrencyConverter
from PyQt5.QtWidgets import QMessageBox


def arima_garch_model(parent_widget, contract_df):

    # Конвертация в EUR
    try:
        converter = CurrencyConverter()
        columns_info = [('total_contract_amount', 'contract_currency', 'total_contract_amount_eur')]
        filtered_df_eur = converter.convert_multiple_columns(
            df=contract_df, columns_info=columns_info)
    except Exception as e:
        QMessageBox.warning(parent_widget, 'Ошибка конвертации', f"Ошибка при конвертации валют: {str(e)}")
        return

    df_arima = filtered_df_eur[['contract_signing_date', 'total_contract_amount_eur']].set_index('contract_signing_date')
    
    # Создаем столбец для полугодий
    df_arima['Полугодие'] = df_arima.index.to_period('6M')
    
    # Группируем данные по полугодиям и суммируем контракты
    semiannual_data = df_arima.groupby('Полугодие')['total_contract_amount_eur'].sum()
    
    # Преобразуем в DataFrame для дальнейшей работы
    semiannual_df = pd.DataFrame({
        'Полугодие': semiannual_data.index, 
        'Сумма контрактов': semiannual_data.values
    })
    semiannual_df.set_index('Полугодие', inplace=True)
    
    # Построим модель ARIMA
    arima_model = ARIMA(semiannual_df['Сумма контрактов'], order=(1, 0, 1))
    arima_fit = arima_model.fit()
    
    # Получаем остатки (residuals) из ARIMA модели
    residuals = arima_fit.resid
    
    # Построим модель GARCH(1, 1)
    garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp="off")
    
    # Выведем результаты модели GARCH
    print(garch_fit.summary())
    
    # Прогнозируем волатильность на следующие 3 периода
    forecast = garch_fit.forecast(horizon=3)
    forecast_variance = forecast.variance.values[-1, :]
    
    # === УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ ===
    
    # Настройка стиля и размера
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Преобразуем Period в datetime для лучшего отображения
    residuals_dates = residuals.index.to_timestamp()
    forecast_dates = pd.date_range(
        start=residuals_dates[-1], 
        periods=4, 
        freq='6M'
    )[1:]  # Следующие 3 полугодия
    
    # 1. ИСТОРИЧЕСКИЕ ОСТАТКИ
    ax.plot(residuals_dates, residuals.values,
            color='#1f77b4', linewidth=2, 
            label='Остатки ARIMA', alpha=0.8, zorder=3)
    
    # 2. ВЕРТИКАЛЬНАЯ ЛИНИЯ РАЗДЕЛЕНИЯ
    split_date = residuals_dates[-1]
    ax.axvline(x=split_date, color='#7f7f7f', 
               linestyle=':', linewidth=2.5, 
               label='Начало прогноза', alpha=0.7, zorder=2)
    
    # 3. ПРОГНОЗ ВОЛАТИЛЬНОСТИ (более заметный)
    ax.plot(forecast_dates, forecast_variance,
            color='#d62728', linewidth=3.5, linestyle='--',
            label='Прогноз волатильности GARCH', 
            marker='o', markersize=8,
            markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor='#d62728', zorder=4)
    
    # 4. ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ для волатильности (±2 std)
    forecast_std = forecast_variance ** 0.5
    upper_bound = forecast_variance + 2 * forecast_std
    lower_bound = forecast_variance - 2 * forecast_std
    
    ax.fill_between(forecast_dates, lower_bound, upper_bound,
                    color='#ff7f0e', alpha=0.25,
                    label='95% Доверительный интервал', zorder=1)
    
    # 5. ФОРМАТИРОВАНИЕ ОСИ Y
    def format_large_numbers(x, pos):
        if abs(x) >= 1e6:
            return f'{x/1e6:.1f}M €'
        elif abs(x) >= 1000:
            return f'{x/1000:.0f}K €'
        return f'{x:.0f} €'
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))
    
    # 6. УЛУЧШЕННАЯ СЕТКА
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='major')
    ax.set_axisbelow(True)
    
    # 7. СТАТИСТИЧЕСКИЙ БЛОК
    avg_volatility = forecast_variance.mean()
    trend = "↗️ рост" if forecast_variance[-1] > forecast_variance[0] else "↘️ снижение"
    max_vol = forecast_variance.max()
    min_vol = forecast_variance.min()
    
    textstr = f'📊 Прогноз волатильности:\n'
    textstr += f'Средняя: {avg_volatility:,.0f} €²\n'
    textstr += f'Тренд: {trend}\n'
    textstr += f'Диапазон: {min_vol:,.0f} - {max_vol:,.0f} €²'
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat',
                     alpha=0.85, edgecolor='gray', linewidth=1.5),
            zorder=5, family='monospace')
    
    # 8. УЛУЧШЕННАЯ ЛЕГЕНДА
    legend = ax.legend(loc='upper right', frameon=True, shadow=True,
                      fontsize=11, fancybox=True, framealpha=0.95,
                      edgecolor='gray', facecolor='white')
    legend.set_zorder(6)
    
    # 9. ЗАГОЛОВКИ И ПОДПИСИ
    ax.set_title('Прогноз волатильности с помощью ARIMA-GARCH\n'
                 '(Остатки ARIMA и прогноз условной дисперсии)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Полугодие', fontsize=12, fontweight='bold')
    ax.set_ylabel('Остатки / Волатильность (EUR²)', fontsize=12, fontweight='bold')
    
    # 10. ДОПОЛНИТЕЛЬНЫЕ УЛУЧШЕНИЯ
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Показать график
    plt.show()
    
    # Возвращаем результаты для дальнейшего использования
    return {
        'arima_fit': arima_fit,
        'garch_fit': garch_fit,
        'forecast': forecast,
        'residuals': residuals
    }