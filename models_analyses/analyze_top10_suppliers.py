import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_top10_suppliers(
    df,
    supplier_col="supplier_name",
    value_col="contract_sum",
    time_col="date",
    product_col="product_name",
    discipline_col="discipline",
):
    """
    Анализирует топ-10 поставщиков по объему закупок.

    df : pd.DataFrame
            Данные (обязательно должны содержать supplier_col, value_col).
    supplier_col : str
            Название колонки с поставщиками.
    value_col : str
            Название колонки с объемом закупки (сумма контракта или цена).
    time_col : str
            Дата/период для динамики.
    product_col : str
            Название товара (для анализа ассортимента).
    discipline_col : str
            Дисциплина/категория (для анализа диверсификации).
    """

    # --- 1. Топ-10 по сумме ---
    top10 = df.groupby(supplier_col)[value_col].sum().nlargest(10).reset_index()
    top_suppliers = top10[supplier_col].tolist()
    top10_total = top10[value_col].sum()
    total = df[value_col].sum()

    print("=== ТОП-10 поставщиков ===")
    print(top10)
    print(f"\nДоля ТОП-10 от всех закупок: {top10_total/total:.2%}")

    # --- 2. Коэффициент Джини по долям ---
    shares = top10[value_col] / total
    shares_sorted = np.sort(shares)
    cum_shares = np.cumsum(shares_sorted)
    gini = 1 - 2 * np.trapz(cum_shares, dx=1 / len(cum_shares))
    print(f"Коэффициент Джини (только топ-10): {gini:.3f}")

    # --- 3. Динамика по времени ---
    df_time = (
        df[df[supplier_col].isin(top_suppliers)]
        .groupby([pd.Grouper(key=time_col, freq="Q"), supplier_col])[value_col]
        .sum()
        .reset_index()
    )

    # --- 4. Средние цены ---
    df_prices = (
        df[df[supplier_col].isin(top_suppliers)]
        .groupby([supplier_col, product_col])[value_col]
        .mean()
        .reset_index()
    )

    # --- 5. Диверсификация ассортимента ---
    df_div = (
        df[df[supplier_col].isin(top_suppliers)]
        .groupby(supplier_col)[[product_col, discipline_col]]
        .nunique()
        .reset_index()
        .rename(
            columns={
                product_col: "unique_products",
                discipline_col: "unique_disciplines",
            }
        )
    )

    print("\n=== Диверсификация поставщиков ===")
    print(df_div)

    # === ВИЗУАЛИЗАЦИИ ===

    # Pie chart: доля топ-10
    plt.figure(figsize=(6, 6))
    plt.pie(top10[value_col], labels=top10[supplier_col], autopct="%1.1f%%")
    plt.title("Доли ТОП-10 поставщиков")
    plt.show()

    # Line chart: динамика по кварталам
    plt.figure(figsize=(10, 6))
    for sup in top_suppliers:
        supplier_data = df_time[df_time[supplier_col] == sup]
        plt.plot(supplier_data[time_col], supplier_data[value_col], label=sup)
    plt.title("Динамика закупок по кварталам (ТОП-10)")
    plt.xlabel("Период")
    plt.ylabel("Сумма закупок")
    plt.legend()
    plt.show()

    # Boxplot: цены по товарам
    plt.figure(figsize=(10, 6))
    df_box = df[df[supplier_col].isin(top_suppliers)]
    df_box.boxplot(column=value_col, by=supplier_col, rot=45)
    plt.title("Разброс цен у ТОП-10 поставщиков")
    plt.suptitle("")
    plt.ylabel("Цена")
    plt.show()

    return {
        "top10": top10,
        "gini": gini,
        "diversification": df_div,
        "time_dynamics": df_time,
        "price_analysis": df_prices,
    }
