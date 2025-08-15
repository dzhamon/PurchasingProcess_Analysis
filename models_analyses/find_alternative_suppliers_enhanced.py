import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
from utils.functions import CurrencyConverter

# Вспомогательные функции, которые не зависят от состояния класса, остаются вне его.
def calculate_years_experience(supplier_contracts):
    """
    Рассчитывает количество лет опыта поставщика.
    """
    if "contract_signing_date" not in supplier_contracts.columns or supplier_contracts.empty:
        return 0

    try:
        dates = pd.to_datetime(supplier_contracts["contract_signing_date"], errors="coerce").dropna()
        if dates.empty or len(dates) < 1:
            return 0
        first_date = dates.min()
        last_date = dates.max()
        years_diff = (last_date - first_date).days / 365.25
        return max(years_diff, 1.0) if len(dates) > 0 else 0
    except Exception as e:
        print(f"Ошибка расчета опыта поставщика: {str(e)}")
        return 0


def calculate_recommendation_score(supplier_info, current_avg_price):
    """
    Рассчитывает интегральный рейтинг поставщика (0-100).
    """
    score = 50
    experience_score = min(supplier_info["years_experience"] * 5, 25)
    score += experience_score
    contracts_score = min(supplier_info["contracts_count"] * 2, 15)
    score += contracts_score
    if current_avg_price > 0 and supplier_info["avg_price"] > 0:
        price_diff = (
            current_avg_price - supplier_info["avg_price"]
        ) / current_avg_price
        if price_diff > 0:
            price_score = min(price_diff * 100, 20)
        else:
            price_score = max(price_diff * 50, -10)
        score += price_score
    if supplier_info["price_std"] > 0:
        stability_score = max(
            10 - supplier_info["price_std"] / supplier_info["avg_price"] * 100, 0
        )
        score += stability_score
    diversification_score = min(supplier_info["projects_count"] * 2, 10)
    score += diversification_score
    return min(max(score, 0), 100)


def identify_supplier_advantages(supplier_info, current_avg_price):
    """
    Определяет конкурентные преимущества поставщика.
    """
    advantages = []
    if supplier_info["years_experience"] >= 3:
        advantages.append(
            f"Большой опыт работы ({supplier_info['years_experience']:.1f} лет)"
        )
    if supplier_info["contracts_count"] >= 10:
        advantages.append(
            f"Высокая контрактная активность ({supplier_info['contracts_count']} контрактов)"
        )
    # Используем current_avg_price (который будет unit_price_eur)
    if current_avg_price > 0 and supplier_info["avg_price"] < current_avg_price * 0.9:
        savings = (1 - supplier_info["avg_price"] / current_avg_price) * 100
        advantages.append(f"Ценовое преимущество (экономия {savings:.1f}%)")
    if supplier_info["projects_count"] > 1:
        advantages.append(
            f"Работа на разных проектах ({supplier_info['projects_count']} проектов)"
        )
    if supplier_info["price_std"] / supplier_info["avg_price"] < 0.15:
        advantages.append("Стабильные цены")
    return advantages


def assess_supplier_risks(supplier_info):
    """
    Оценивает потенциальные риски работы с поставщиком.
    """
    risks = []
    if supplier_info["years_experience"] < 1:
        risks.append("Недостаточный опыт работы")
    if supplier_info["contracts_count"] < 3:
        risks.append("Малое количество контрактов")
    if supplier_info["price_std"] / supplier_info["avg_price"] > 0.3:
        risks.append("Нестабильные цены")
    if supplier_info["last_contract"]:
        last_contract_date = pd.to_datetime(supplier_info["last_contract"])
        if datetime.now() - last_contract_date > timedelta(days=365):
            risks.append("Длительный перерыв в работе")
    return risks


def generate_supplier_recommendation(supplier_info, current_avg_price):
    """
    Генерирует текстовую рекомендацию по поставщику.
    """
    if supplier_info["contracts_count"] >= 5 and supplier_info["years_experience"] >= 2:
        if current_avg_price > 0 and supplier_info["avg_price"] < current_avg_price:
            return "⭐ РЕКОМЕНДУЕТСЯ: Опытный поставщик с привлекательными ценами"
        else:
            return "✅ ПОДХОДИТ: Надежный поставщик с хорошей репутацией"
    elif supplier_info["contracts_count"] >= 2:
        return "⚠️ ОСТОРОЖНО: Ограниченный опыт, требует дополнительной проверки"
    else:
        return "❌ НЕ РЕКОМЕНДУЕТСЯ: Недостаточный опыт работы"


def analyze_market_position(current_suppliers, alternative_suppliers, all_suppliers):
    """
    Анализирует рыночную позицию и конкуренцию.
    """
    total_suppliers = len(all_suppliers)
    current_count = len(current_suppliers)
    alternative_count = len(alternative_suppliers)

    market_shares = {}
    total_contracts = sum(info["contracts_count"] for info in all_suppliers.values())
    for supplier, info in all_suppliers.items():
        market_shares[supplier] = (
            info["contracts_count"] / total_contracts if total_contracts > 0 else 0
        )
    current_concentration = sum(
        market_shares.get(supplier, 0) for supplier in current_suppliers
    )

    return {
        "total_market_suppliers": total_suppliers,
        "current_suppliers_count": current_count,
        "alternative_suppliers_count": alternative_count,
        "market_coverage_current": current_concentration,
        "diversification_potential": (
            alternative_count / total_suppliers if total_suppliers > 0 else 0
        ),
        "market_competitiveness": (
            "Высокая"
            if alternative_count > current_count
            else "Средняя" if alternative_count > 0 else "Низкая"
        ),
    }


def generate_product_recommendation_summary(product_alternatives):
    """
    Генерирует рекомендацию по продукту на основе найденных альтернатив.
    """
    alternatives_count = len(product_alternatives["alternative_suppliers"])
    if alternatives_count == 0:
        return "Критический риск: альтернативы не найдены. Требуется расширение поиска поставщиков."
    elif alternatives_count <= 2:
        return f"Ограниченный выбор: {alternatives_count} альтернатив. Рекомендуется активный поиск новых поставщиков."
    else:
        return f"Хорошие возможности диверсификации: {alternatives_count} альтернатив доступно."

class AlternativeSuppliersAnalyzer:
    def __init__(self):
        self.all_contracts_data = None
        print("AlternativeSuppliersAnalyzer: Инициализирован.")

    def receive_contract_data(self, df: pd.DataFrame):
        print("AlternativeSuppliersAnalyzer: Данные контрактов получены через сигнал!")
        self.all_contracts_data = df
        print(
            f"AlternativeSuppliersAnalyzer: DataFrame содержит {len(df)} строк и {len(df.columns)} столбцов."
        )

    def run_analysis(
        self, current_project_data: pd.DataFrame,
        target_disciplines: list = None,
        target_supplier: str = None,
    ) -> Dict[str, Any]:
        """
        Запускает полный цикл поиска альтернативных поставщиков для одной или нескольких дисциплин.
        Возвращает агрегированные результаты для экспорта в Excel.

        Args:
            current_project_data (pd.DataFrame): Отфильтрованные данные по текущему проекту/лотам/контрактам.

            target_disciplines (list, optional): Список дисциплин для анализа. Если None, анализируются все
            дисциплины в current_project_data.

        Returns:
            Dict[str, Any]: Словарь, где ключ - это имя дисциплины, а значение -
                            отформатированные результаты анализа для этой дисциплины.
                            Формат для каждой дисциплины:
                            {
                                "product_name": {
                                    "current_suppliers": [...],
                                    "alternatives_found": int,
                                    "recommendation": "text",
                                    "alternative_suppliers": [...] # полный список
                                },
                                ...
                            }
        """
        if self.all_contracts_data is None:
            print(
                "ОШИБКА: Нет полных данных контрактов (self.all_contracts_data) для запуска анализа."
            )
            return {}

        all_disciplines_results = {}

        # Определяем список дисциплин для анализа
        if target_disciplines is None:
            disciplines_to_analyze = current_project_data['discipline'].dropna().unique().tolist()
        else:
            disciplines_to_analyze = [d for d in target_disciplines if d in current_project_data['discipline'].unique()]

        if not disciplines_to_analyze:
            print("Предупреждение: Нет дисциплин для анализа в предоставленных данных.")
            return {}

        print(f"Начинаем анализ по дисциплинам: {'. '.join(disciplines_to_analyze)}")

        for discipline in disciplines_to_analyze:
            print(f"\n==== Анализ по дисциплине: {discipline} ====")

            # фильтруем данные для текущей дисциплины
            discipline_data = current_project_data[current_project_data['discipline'] == discipline]

            if discipline_data.empty:
                print(f" Нет данных для дисциплины {discipline}, пропускаем")
                continue

            # Выполняем основной логический поиск для данной дисциплины
            raw_alternatives_by_product = self._find_alternative_suppliers_logical(
                discipline_data, discipline #  передаем отфильтрованные данные и текущую дисциплину
            )

            # Форматируем результаты для данной дисциплины
            formatted_for_discipline = {}
            for product, info in raw_alternatives_by_product.items():
                formatted_for_discipline[product] = {
                    "current_suppliers": info["current_suppliers"],
                    "alternatives_found": len(info["alternative_suppliers"]),
                    "recommendation": generate_product_recommendation_summary(info), # Используем уже существующую функцию
                    "alternative_suppliers": info["alternative_suppliers"]
                }
            all_disciplines_results[discipline] = formatted_for_discipline
        return all_disciplines_results


    def _find_alternative_suppliers_logical(
        self, current_project_data: pd.DataFrame, discipline: str, target_supplier: str = None
    ) -> Dict[str, Any]:
        """
        Логический поиск альтернативных поставщиков на основе всех контрактов организации
        (Теперь это приватный метод класса, использующий self.all_contracts_data).
        """
        # print(f"🔍 ПОИСК АЛЬТЕРНАТИВ для дисциплины: {discipline}")
        # print("=" * 60)

        alternatives_by_product = {}

        current_products = current_project_data["product_name"].unique()

        for product in current_products:
            # Используем 'counterparty_name'
            current_suppliers = {target_supplier} if target_supplier else set(
                current_project_data[current_project_data["product_name"] == product][
                    "counterparty_name"
                ].unique()
            )
            # print(f"  Текущий поставщик для анализа: {current_suppliers}")

            all_product_suppliers = self._find_all_suppliers_for_product(
                product, discipline
            )
            # Ищем альтернативы среди всех поставщиков, исключая target_supplier
            alternative_suppliers = set(all_product_suppliers.keys()) - current_suppliers

            # print(
            #     f"   Найдено альтернативных поставщиков: {len(alternative_suppliers)}"
            # )

            if alternative_suppliers:
                alternatives_by_product[product] = {
                    "current_suppliers": list(current_suppliers),
                    "alternative_suppliers": self._analyze_alternative_suppliers(
                        alternative_suppliers,
                        all_product_suppliers,
                        current_project_data,
                        product,
                    ),
                    "market_analysis": analyze_market_position(
                        current_suppliers, alternative_suppliers, all_product_suppliers
                    ),
                }

                top_alternatives = sorted(
                    alternatives_by_product[product]["alternative_suppliers"],
                    key=lambda x: x["recommendation_score"],
                    reverse=True,
                )[:3]

                # print(f"   🏆 Топ-3 альтернативы:")
                for i, alt in enumerate(top_alternatives, 1):
                    print(
                        f"      {i}. {alt['supplier_name']} (рейтинг: {alt['recommendation_score']:.2f})"
                    )
                    print(
                        f"         Опыт: {alt['contracts_count']} контрактов, средняя цена: {alt['avg_price']:.2f})"
                    )
            else:
                # print(f"   ⚠️  Альтернативные поставщики не найдены")
                alternatives_by_product[product] = {
                    "current_suppliers": list(current_suppliers),
                    "alternative_suppliers": [],
                    "market_analysis": {
                        "message": "Монопольная ситуация или недостаток данных"
                    },
                }

        return alternatives_by_product

    def _find_all_suppliers_for_product(self, product_name: str, discipline: str):
        """
        Находит всех поставщиков конкретного товара во всех контрактах
        (Теперь это приватный метод класса, использующий self.all_contracts_data)
        """
        
        product_contracts = self.all_contracts_data[
            (self.all_contracts_data["product_name"] == product_name)
        ]
        
        # конвертируем цены за единицу и стоимость контракта в EUR
        converter = CurrencyConverter()
        columns_info = [
            ("total_contract_amount", "contract_currency", "total_contract_amount_eur"),
            ("unit_price", "contract_currency", "unit_price_eur"),
        ]
        product_contracts = converter.convert_multiple_columns(product_contracts, columns_info).copy()
        # может есть смысл убрать следующее условие (искатьальтернативу и в других дисциплинах??)
        if "discipline" in product_contracts.columns:
            product_contracts = product_contracts[
                product_contracts["discipline"] == discipline
            ]

        suppliers_info = {}
        # Используем 'counterparty_name'
        for supplier in product_contracts["counterparty_name"].unique():
            supplier_contracts = product_contracts[
                product_contracts["counterparty_name"] == supplier
            ]
            suppliers_info[supplier] = {
                "contracts_count": len(supplier_contracts),
                "total_value": (
                    supplier_contracts["total_contract_amount_eur"].sum()
                    if "total_contract_amount_eur" in supplier_contracts.columns
                    else 0
                ),
                "avg_price": (
                    supplier_contracts["unit_price_eur"].mean()
                    if "unit_price_eur" in supplier_contracts.columns
                    else 0
                ),
                "price_std": (
                    supplier_contracts["unit_price_eur"].std()
                    if "unit_price_eur" in supplier_contracts.columns
                    else 0
                ),
                "first_contract": (
                    supplier_contracts["contract_signing_date"].min()
                    if "contract_signing_date" in supplier_contracts.columns
                    else None
                ),
                "last_contract": (
                    supplier_contracts["contract_signing_date"].max()
                    if "contract_signing_date" in supplier_contracts.columns
                    else None
                ),
                "years_experience": calculate_years_experience(supplier_contracts),
                "projects_count": (
                    supplier_contracts["project"].nunique()
                    if "project" in supplier_contracts.columns
                    else 1
                ),
            }
        return suppliers_info

    def _analyze_alternative_suppliers(
        self,
        alternative_suppliers: set,
        all_suppliers_info: Dict[str, Any],
        current_data: pd.DataFrame,
        product: str,
    ):
        """
        Анализирует альтернативных поставщиков и ранжирует их по привлекательности.
        """
        current_product_data = current_data[current_data["product_name"] == product]
        current_avg_price = (
            current_product_data["unit_price_eur"].mean() # Изменено с 'price' на 'unit_price_eur'
            if "unit_price_eur" in current_product_data.columns
            else 0
        )

        analyzed_alternatives = []
        for supplier in alternative_suppliers:
            supplier_info = all_suppliers_info[supplier]
            recommendation_score = calculate_recommendation_score(
                supplier_info, current_avg_price
            )
            advantages = identify_supplier_advantages(supplier_info, current_avg_price)
            risks = assess_supplier_risks(supplier_info)
            analyzed_alternatives.append(
                {
                    "supplier_name": supplier,
                    "recommendation_score": recommendation_score,
                    "contracts_count": supplier_info["contracts_count"],
                    "avg_price": supplier_info["avg_price"],
                    "price_vs_current": (
                        (
                            (supplier_info["avg_price"] - current_avg_price)
                            / current_avg_price
                            * 100
                        )
                        if current_avg_price > 0
                        else 0
                    ),
                    "years_experience": supplier_info["years_experience"],
                    "projects_count": supplier_info["projects_count"],
                    "advantages": advantages,
                    "risks": risks,
                    "recommendation": generate_supplier_recommendation(
                        supplier_info, current_avg_price
                    ),
                }
            )
        return analyzed_alternatives


def export_alternative_suppliers_to_excel(results: Dict[str, Any], file_path: str):
    """
    Экспортирует результаты анализы альтернативных поставщиков в Excel-файл.
    Каждая дисциплина на отдельном листе.
    """
    print(f"🔄 Начало экспорта в файл: {file_path}")
    print(f"📊 Количество дисциплин для экспорта: {len(results)}")

    try:
        print("🔧 Создание Excel writer...")
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for discipline, products_data in results.items():
                print(
                    f"📋 Обработка дисциплины: {discipline} ({len(products_data)} продуктов)"
                )

                # Создаем DataFrame для текущей дисциплины
                discipline_export_data = []

                for product, info in products_data.items():
                    print(f"  📦 Обработка продукта: {product}")

                    # Текущие поставщики
                    current_suppliers_str = ", ".join(info["current_suppliers"])

                    # Альтернативные поставщики (детализированные)
                    if info["alternative_suppliers"]:
                        print(
                            f"    ✅ Найдено {len(info['alternative_suppliers'])} альтернатив"
                        )
                        for alt in info["alternative_suppliers"]:
                            discipline_export_data.append(
                                {
                                    "Дисциплина": discipline,
                                    "Продукт": product,
                                    "Текущие поставщики": current_suppliers_str,
                                    "Кол-во альтернатив": info["alternatives_found"],
                                    "Рекомендация по продукту": info["recommendation"],
                                    "Альтернативный поставщик": alt["supplier_name"],
                                    "Рейтинг рекомендации": f"{alt['recommendation_score']:.2f}",
                                    "Кол-во контрактов (альт.)": alt["contracts_count"],
                                    "Ср. цена (альт., EUR)": f"{alt['avg_price']:.2f}",
                                    "Цена от текущей (%)": f"{alt['price_vs_current']:.2f}",
                                    "Опыт (лет, альт.)": f"{alt['years_experience']:.1f}",
                                    "Кол-во проектов (альт.)": alt["projects_count"],
                                    "Преимущества (альт.)": "; ".join(
                                        alt["advantages"]
                                    ),
                                    "Риски (альт.)": "; ".join(alt["risks"]),
                                    "Рекомендация по поставщику": alt["recommendation"],
                                }
                            )
                    else:
                        print(f"    ❌ Альтернативы не найдены")
                        # Если нет альтернатив, все равно добавим строку для текущих
                        discipline_export_data.append(
                            {
                                "Дисциплина": discipline,
                                "Продукт": product,
                                "Текущие поставщики": current_suppliers_str,
                                "Кол-во альтернатив": info["alternatives_found"],
                                "Рекомендация по продукту": info["recommendation"],
                                "Альтернативный поставщик": "Нет",
                                "Рейтинг рекомендации": "-",
                                "Кол-во контрактов (альт.)": "-",
                                "Ср. цена (альт., EUR)": "-",
                                "Цена от текущей (%)": "-",
                                "Опыт (лет, альт.)": "-",
                                "Кол-во проектов (альт.)": "-",
                                "Преимущества (альт.)": "-",
                                "Риски (альт.)": "-",
                                "Рекомендация по поставщику": "-",
                            }
                        )

                if discipline_export_data:
                    print(
                        f"  💾 Создание DataFrame для {discipline} ({len(discipline_export_data)} строк)"
                    )
                    df_discipline = pd.DataFrame(discipline_export_data)

                    # Имя листа должно быть коротким и без запрещенных символов
                    sheet_name = discipline[:31]  # Макс. 31 символ
                    print(f"  📄 Запись в лист: {sheet_name}")

                    df_discipline.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✅ Лист {sheet_name} успешно создан")
                else:
                    print(f"  ⚠️ Нет данных для экспорта по дисциплине: {discipline}")

        print("💾 Сохранение файла...")
        print(
            f"✅ Результаты анализа альтернативных поставщиков успешно экспортированы в: {file_path}"
        )
        return True

    except Exception as e:
        print(f"❌ Ошибка при экспорте результатов анализа в Excel: {e}")
        return False