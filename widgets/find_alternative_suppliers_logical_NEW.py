import pandas as pd
from typing import Dict, Any


class AlternativeSuppliersAnalyzer:
    def __init__(self):
        self.all_contracts_data = (
            None  # здесь будет храниться принятый DataFrame self.base_contract_df
        )

    # Это метод-слот, который будет принимать DataFrame
    def receive_contract_data(self, df: pd.DataFrame):
        print("Данные контрактов получены через сигнал!")
        self.all_contracts_data = df


def find_alternative_suppliers_logical(
    current_project_data, all_contracts_data, discipline
):
    """
    Логический поиск альтернативных поставщиков на основе всех контрактов организации

    Args:
                    current_project_data: Данные по текущему проекту/дисциплине
                    all_contracts_data: ВСЕ контракты организации (исторические данные)
                    discipline: Анализируемая дисциплина

    Returns:
                    dict: Альтернативные поставщики с детальной информацией
    """

    print(f"🔍 ПОИСК АЛЬТЕРНАТИВ для дисциплины: {discipline}")
    print("=" * 60)

    alternatives_by_product = {}

    # Получаем список товаров текущего проекта
    current_products = current_project_data["product_name"].unique()

    for product in current_products:
        print(f"\n📦 Анализ продукта: {product}")

        # Текущие поставщики этого товара
        # ИСПОЛЬЗУЕМ 'supplier_name'
        current_suppliers = (
            current_project_data[current_project_data["product_name"] == product][
                "supplier_name"
            ]
            .unique()
            .tolist()
        )
        print(f"Текущие поставщики: {', '.join(current_suppliers)}")

        # Ищем всех поставщиков, которые когда-либо поставляли этот товар
        # Исключаем текущих поставщиков из списка потенциальных альтернатив,
        # если они уже являются текущими для данного проекта/дисциплины

        # Фильтруем все контракты по текущему продукту
        product_contracts = all_contracts_data[
            all_contracts_data["product_name"] == product
        ]

        # Находим всех уникальных поставщиков этого продукта
        all_product_suppliers = product_contracts["supplier_name"].unique()

        # Исключаем текущих поставщиков
        alternative_suppliers_names = [
            supplier
            for supplier in all_product_suppliers
            if supplier not in current_suppliers
        ]

        alternative_suppliers_details = []
        if alternative_suppliers_names:
            print(
                f"Потенциальные альтернативы: {', '.join(alternative_suppliers_names)}"
            )

            # Для каждой альтернативы собираем детали: среднюю цену, количество контрактов
            for alt_supplier in alternative_suppliers_names:
                supplier_contracts = product_contracts[
                    product_contracts["supplier_name"] == alt_supplier
                ]
                avg_price = supplier_contracts["unit_price_eur"].mean()
                num_contracts = supplier_contracts.shape[0]

                # Простая метрика рекомендации: чем больше контрактов и ниже средняя цена, тем лучше
                # Можно усложнить: учесть дату последнего контракта, объем, отзывы и т.д.
                recommendation_score = 0
                if num_contracts > 0:
                    # Например, 100 / avg_price для инверсии, чтобы меньшая цена давала больший балл
                    # + баллы за количество контрактов
                    recommendation_score = (1000 / avg_price) + (num_contracts * 5)

                alternative_suppliers_details.append(
                    {
                        "supplier_name": alt_supplier,
                        "avg_price": avg_price,
                        "num_contracts": num_contracts,
                        "recommendation_score": recommendation_score,
                    }
                )
            # Сортируем альтернативы по убыванию рекомендационного балла
            alternative_suppliers_details.sort(
                key=lambda x: x["recommendation_score"], reverse=True
            )
        else:
            print("Альтернативы не найдены.")

        # Простой анализ рынка (можно расширить)
        market_analysis_text = "Рынок: "
        if len(all_product_suppliers) <= 2:
            market_analysis_text += "Высококонцентрированный."
        elif len(all_product_suppliers) <= 5:
            market_analysis_text += "Умеренно концентрированный."
        else:
            market_analysis_text += "Конкурентный."

        alternatives_by_product[product] = {
            "current_suppliers": current_suppliers,
            "alternative_suppliers": alternative_suppliers_details,
            "market_analysis": market_analysis_text,
            "alternatives_found": len(
                alternative_suppliers_details
            ),  # <-- Этот ключ должен быть здесь!
        }
    return alternatives_by_product


def analyze_alternative_suppliers(current_project_data, all_contracts_data, discipline):
    """
    Выполняет анализ альтернативных поставщиков и форматирует результаты.
    """
    alternatives = find_alternative_suppliers_logical(
        current_project_data, all_contracts_data, discipline
    )

    # Преобразуем в формат, совместимый с существующим кодом
    formatted_results = {}  # Возвращаем словарь, а не список
    for product, product_alternatives in alternatives.items():
        formatted_results[product] = {  # Сохраняем по ключу продукта
            "current_suppliers": product_alternatives["current_suppliers"],
            "alternatives_found": len(  # <-- И здесь он также формируется
                product_alternatives["alternative_suppliers"]
            ),
            "top_alternatives": product_alternatives["alternative_suppliers"][
                :5
            ],  # Топ-5
            "market_analysis": product_alternatives["market_analysis"],
            "recommendation": generate_product_recommendation(product_alternatives),
            "alternative_suppliers": product_alternatives[
                "alternative_suppliers"
            ],  # Добавлено для полного списка
        }

    return formatted_results


def generate_product_recommendation(product_alternatives):
    """
    Генерирует рекомендацию по продукту на основе найденных альтернатив
    """
    alternatives_count = len(product_alternatives["alternative_suppliers"])

    if alternatives_count == 0:
        return "Критический риск: альтернативы не найдены. Требуется расширение поиска поставщиков."
    elif alternatives_count <= 2:
        return f"Ограниченный выбор: {alternatives_count} альтернатив. Рекомендуется активный поиск новых поставщиков."
    else:
        return f"Хорошие возможности диверсификации: {alternatives_count} альтернатив доступно. Рекомендуется рассмотреть диверсификацию поставщиков."
