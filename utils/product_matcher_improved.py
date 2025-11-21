"""
Модуль для сопоставления и сравнения товаров с улучшенным извлечением характеристик.
Поддерживает широкий спектр товарных категорий.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher


class ProductMatcher:
    """Класс для сопоставления и анализа товаров"""

    # Категории товаров с ключевыми словами
    CATEGORIES = {
        'металлопрокат': [
            'труба', 'швеллер', 'двутавр', 'арматура', 'уголок', 'балка',
            'профиль', 'лист', 'полоса', 'круг', 'проволока', 'катанка',
            'тавр', 'прокат', 'металлоконструкция', 'сталь'
        ],
        'трубы_фитинги': [
            'труба', 'отвод', 'тройник', 'переход', 'фланец', 'муфта',
            'заглушка', 'колено', 'патрубок', 'штуцер', 'ниппель', 'футорка'
        ],
        'крепеж': [
            'болт', 'гайка', 'шайба', 'винт', 'саморез', 'шуруп', 'гвоздь',
            'анкер', 'дюбель', 'шпилька', 'заклепка', 'скоба', 'хомут',
            'клипса', 'стяжка'
        ],
        'электрика': [
            'кабель', 'провод', 'розетка', 'выключатель', 'лампа', 'светильник',
            'щит', 'автомат', 'контактор', 'пускатель', 'реле', 'трансформатор',
            'счетчик', 'датчик', 'извещатель', 'прожектор', 'патрон'
        ],
        'сантехника': [
            'смеситель', 'унитаз', 'раковина', 'ванна', 'душ', 'сифон',
            'умывальник', 'писсуар', 'кран', 'вентиль', 'задвижка',
            'трап', 'решетка', 'мойка'
        ],
        'клапаны_арматура': [
            'клапан', 'вентиль', 'задвижка', 'кран', 'затвор', 'редуктор',
            'регулятор', 'манометр', 'термометр', 'счетчик', 'обратный',
            'предохранительный', 'запорный'
        ],
        'строительные_материалы': [
            'цемент', 'бетон', 'раствор', 'смесь', 'песок', 'щебень', 'гравий',
            'кирпич', 'блок', 'плита', 'панель', 'гипсокартон', 'утеплитель',
            'минвата', 'пенопласт', 'пеноплекс', 'сэндвич-панель', 'сэндвич',
            'сендвич-панель', 'сендвич'
        ],
        'лкм': [
            'краска', 'эмаль', 'грунтовка', 'лак', 'шпатлевка', 'герметик',
            'растворитель', 'разбавитель', 'олифа', 'мастика', 'праймер',
            'битум', 'смола'
        ],
        'инструмент': [
            'дрель', 'перфоратор', 'шуруповерт', 'болгарка', 'ушм', 'пила',
            'лобзик', 'фрезер', 'рубанок', 'молоток', 'ключ', 'отвертка',
            'плоскогубцы', 'кусачки', 'ножницы', 'нож'
        ],
        'измерительные_приборы': [
            'манометр', 'термометр', 'расходомер', 'счетчик', 'датчик',
            'уровень', 'рулетка', 'штангенциркуль', 'нивелир', 'мультиметр',
            'тестер', 'вольтметр', 'амперметр'
        ],
        'спецодежда_сиз': [
            'костюм', 'комбинезон', 'куртка', 'брюки', 'халат', 'роба',
            'перчатки', 'каска', 'очки', 'респиратор', 'маска', 'противогаз',
            'беруши', 'наушники', 'каски', 'шлем'
        ],
        'пожарное_оборудование': [
            'огнетушитель', 'рукав', 'ствол', 'гидрант', 'кран', 'извещатель',
            'оповещатель', 'датчик', 'щит', 'шкаф', 'ящик'
        ],
        'оборудование': [
            'насос', 'компрессор', 'генератор', 'трансформатор', 'двигатель',
            'редуктор', 'мотор', 'вентилятор', 'калорифер', 'котел'
        ]
    }

    # Единицы измерения
    UNITS = {
        'length': ['мм', 'см', 'м', 'км'],
        'weight': ['г', 'кг', 'т', 'тонн'],
        'volume': ['л', 'мл', 'м3', 'м³'],
        'area': ['м2', 'м²', 'см2', 'см²'],
        'power': ['вт', 'квт', 'мвт', 'w', 'kw'],
        'voltage': ['в', 'кв', 'v'],
        'current': ['а', 'ма', 'a'],
        'pressure': ['бар', 'мпа', 'атм', 'bar', 'psi'],
        'temperature': ['°c', '°с', 'c', 'с']
    }

    @staticmethod
    def extract_key_features(product_name: str) -> Dict[str, Any]:
        """
        Извлекает ключевые характеристики из названия товара.

        Args:
            product_name: название товара

        Returns:
            dict: словарь с извлеченными характеристиками
        """
        name_lower = str(product_name).lower()

        features = {
            'type': None,
            'category': None,
            'diameter': None,
            'thickness': None,
            'length': None,
            'width': None,
            'height': None,
            'number': None,
            'material': None,
            'gost': None,
            'model': None,
            'brand': None,
            'specifications': [],
            'measurements': {}
        }

        # Определение категории и типа
        features['category'], features['type'] = ProductMatcher._extract_category_and_type(name_lower)

        # Извлечение размеров и характеристик
        features.update(ProductMatcher._extract_dimensions(name_lower))

        # Извлечение материала
        features['material'] = ProductMatcher._extract_material(name_lower)

        # Извлечение ГОСТ
        features['gost'] = ProductMatcher._extract_gost(name_lower)

        # Извлечение модели/артикула
        features['model'] = ProductMatcher._extract_model(name_lower)

        # Извлечение бренда
        features['brand'] = ProductMatcher._extract_brand(name_lower)

        # Извлечение всех числовых характеристик
        features['measurements'] = ProductMatcher._extract_all_measurements(name_lower)

        # Извлечение спецификаций (стандарты, классы, типы)
        features['specifications'] = ProductMatcher._extract_specifications(name_lower)

        return features

    @staticmethod
    def _extract_category_and_type(name_lower: str) -> Tuple[Optional[str], Optional[str]]:
        """Определяет категорию и тип товара"""
        category = None
        product_type = None

        # Ищем совпадения по категориям
        for cat, keywords in ProductMatcher.CATEGORIES.items():
            for keyword in keywords:
                if keyword in name_lower:
                    if category is None:
                        category = cat
                        product_type = keyword
                    break
            if category:
                break

        # Если не нашли в категориях, пробуем извлечь первое существительное
        if not product_type:
            words = name_lower.split()
            if words:
                product_type = words[0]

        return category, product_type

    @staticmethod
    def _extract_dimensions(name_lower: str) -> Dict[str, Any]:
        """Извлекает размерные характеристики"""
        dimensions = {
            'diameter': None,
            'thickness': None,
            'length': None,
            'width': None,
            'height': None,
            'number': None
        }

        # ПРИОРИТЕТ 1: Толщина с явным указанием (t=, s=, толщ)
        # Это для сэндвич-панелей и подобных товаров
        thickness_explicit_patterns = [
            r't\s*=\s*(\d+\.?\d*)\s*мм',  # t=100мм
            r't\s*=\s*(\d+\.?\d*)',       # t=100
            r's\s*=\s*(\d+\.?\d*)',       # s=6
            r'толщ[:\-]?\s*(\d+\.?\d*)',  # толщина: 100
        ]
        for pattern in thickness_explicit_patterns:
            match = re.search(pattern, name_lower)
            if match:
                try:
                    dimensions['thickness'] = float(match.group(1))
                    break  # Нашли толщину, выходим
                except:
                    pass

        # ПРИОРИТЕТ 2: Диаметр с явным указанием (Ø, ф, d, диаметр)
        diameter_explicit_patterns = [
            r'[øф∅]\s*(\d+\.?\d*)',
            r'диаметр\s*[:\-]?\s*(\d+\.?\d*)',
        ]
        for pattern in diameter_explicit_patterns:
            match = re.search(pattern, name_lower)
            if match:
                try:
                    dimensions['diameter'] = float(match.group(1))
                    break
                except:
                    pass

        # ПРИОРИТЕТ 3: Размеры через "х" (ДxШxВ)
        # Это для труб типа "219x6" или коробок "100x50x30"
        size_pattern = r'(\d+\.?\d*)\s*[xх×]\s*(\d+\.?\d*)(?:\s*[xх×]\s*(\d+\.?\d*))?'
        match = re.search(size_pattern, name_lower)
        if match:
            try:
                val1 = float(match.group(1))
                val2 = float(match.group(2))
                val3 = float(match.group(3)) if match.group(3) else None

                # Определяем что это: труба (диаметр х толщина) или размеры (ДхШхВ)
                if 'труб' in name_lower and val3 is None:
                    # Труба: первое - диаметр, второе - толщина
                    if dimensions['diameter'] is None:
                        dimensions['diameter'] = val1
                    if dimensions['thickness'] is None:
                        dimensions['thickness'] = val2
                else:
                    # Обычные размеры
                    dimensions['length'] = val1
                    dimensions['width'] = val2
                    if val3:
                        dimensions['height'] = val3
            except:
                pass

        # ПРИОРИТЕТ 4: Одиночное число с "мм"
        # Используем только если ничего не нашли выше
        if (dimensions['diameter'] is None and
                dimensions['thickness'] is None and
                dimensions['length'] is None):

            single_mm_pattern = r'(\d+\.?\d*)\s*мм'
            match = re.search(single_mm_pattern, name_lower)
            if match:
                try:
                    # Пробуем определить по контексту
                    val = float(match.group(1))

                    # Если есть слова типа "толщ", "толст" - это толщина
                    if re.search(r'толщ|толст', name_lower):
                        dimensions['thickness'] = val
                    # Если есть "диам" - это диаметр
                    elif re.search(r'диам|ø|ф', name_lower):
                        dimensions['diameter'] = val
                    # Иначе считаем диаметром по умолчанию
                    else:
                        dimensions['diameter'] = val
                except:
                    pass

        # Номер (для профилей типа швеллер, двутавр)
        number_patterns = [
            r'[№#]\s*(\d+\.?\d*)',
            r'(?:швеллер|двутавр|уголок)\s+(\d+\.?\d*)',
            r'\s(\d+\.?\d*)[упаУПА]'
        ]
        for pattern in number_patterns:
            match = re.search(pattern, name_lower)
            if match:
                try:
                    dimensions['number'] = float(match.group(1))
                    break
                except:
                    pass

        return dimensions

    @staticmethod
    def _extract_material(name_lower: str) -> Optional[str]:
        """Извлекает материал"""
        materials = [
            'ст3', 'ст20', 'ст35', 'ст45', 'ст40х',
            'с245', 'с255', 'с275', 'с345',
            '09г2с', '10хснд', '15хснд',
            'оцинк', 'оцинкованный', 'оцинкованная',
            'нержавейка', 'нержавеющий', 'нержавеющая',
            'алюминий', 'алюминиевый', 'алюминиевая',
            'медь', 'медный', 'медная',
            'пвх', 'пнд', 'пп', 'пэ', 'полиэтилен', 'полипропилен',
            'резина', 'резиновый', 'резиновая',
            'бетон', 'железобетон', 'ж/б', 'жби',
            'дерево', 'деревянный', 'деревянная',
            'пластик', 'пластиковый', 'пластиковая'
        ]

        for material in materials:
            if material in name_lower:
                return material

        return None

    @staticmethod
    def _extract_gost(name_lower: str) -> Optional[str]:
        """Извлекает ГОСТ"""
        gost_patterns = [
            r'гост\s+[\d\-\.]+',
            r'ту\s+[\d\-\.]+',
            r'ост\s+[\d\-\.]+',
            r'сто\s+[\d\-\.]+'
        ]

        for pattern in gost_patterns:
            match = re.search(pattern, name_lower)
            if match:
                return match.group(0)

        return None

    @staticmethod
    def _extract_model(name_lower: str) -> Optional[str]:
        """Извлекает модель/артикул"""
        # Паттерны для моделей (буквы и цифры)
        model_patterns = [
            r'[A-Za-z]{2,}\s*[-\s]?\s*\d+[A-Za-z0-9\-\.]*',
            r'\d+[A-Za-z]+\d+',
            r'арт\.?\s*[A-Za-z0-9\-\.]+',
            r'код\s*[A-Za-z0-9\-\.]+'
        ]

        for pattern in model_patterns:
            match = re.search(pattern, name_lower)
            if match:
                return match.group(0)

        return None

    @staticmethod
    def _extract_brand(name_lower: str) -> Optional[str]:
        """Извлекает бренд"""
        # Список известных брендов
        brands = [
            'bosch', 'makita', 'dewalt', 'hitachi', 'metabo', 'milwaukee',
            'legrand', 'schneider', 'abb', 'iek', 'иэк', 'эра',
            'knauf', 'кнауф', 'ceresit', 'церезит',
            'henkel', 'хенкель', 'sika', 'сика',
            'grohe', 'грое', 'hansgrohe', 'ideal standard',
            'zommer', 'зоммер', 'valtec', 'валтек'
        ]

        for brand in brands:
            if brand in name_lower:
                return brand

        return None

    @staticmethod
    def _extract_all_measurements(name_lower: str) -> Dict[str, List[float]]:
        """Извлекает все числовые измерения с единицами"""
        measurements = {}

        # Ищем все числа с единицами измерения
        for unit_type, units in ProductMatcher.UNITS.items():
            for unit in units:
                pattern = rf'(\d+\.?\d*)\s*{re.escape(unit)}'
                matches = re.findall(pattern, name_lower)
                if matches:
                    measurements[unit_type] = [float(m) for m in matches]

        return measurements

    @staticmethod
    def _extract_specifications(name_lower: str) -> List[str]:
        """Извлекает спецификации (классы, типы, стандарты)"""
        specifications = []

        # Паттерны для спецификаций
        spec_patterns = [
            r'класс\s+[A-Za-zА-Яа-я0-9]+',
            r'тип\s+[A-Za-zА-Яа-я0-9]+',
            r'марка\s+[A-Za-zА-Яа-я0-9]+',
            r'исполнение\s+[A-Za-zА-Яа-я0-9]+',
            r'вид\s+[A-Za-zА-Яа-я0-9]+',
            r'сорт\s+[A-Za-zА-Яа-я0-9]+',
            r'ip\d{2}',
            r'dn\d+',
            r'pn\d+',
            r'р\d+',
            r'с\d+',
            r'м\d+'
        ]

        for pattern in spec_patterns:
            matches = re.findall(pattern, name_lower)
            specifications.extend(matches)

        return specifications

    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        """
        Вычисляет коэффициент схожести двух названий товаров.

        Args:
            name1: первое название
            name2: второе название

        Returns:
            float: коэффициент схожести от 0 до 1
        """
        # Базовая схожесть строк
        base_similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

        # Извлекаем характеристики
        features1 = ProductMatcher.extract_key_features(name1)
        features2 = ProductMatcher.extract_key_features(name2)

        # Вес различных факторов
        weights = {
            'type': 0.3,
            'category': 0.2,
            'dimensions': 0.2,
            'material': 0.1,
            'gost': 0.1,
            'model': 0.1
        }

        feature_similarity = 0.0
        total_weight = 0.0

        # Сравниваем тип
        if features1['type'] and features2['type']:
            if features1['type'] == features2['type']:
                feature_similarity += weights['type']
            total_weight += weights['type']

        # Сравниваем категорию
        if features1['category'] and features2['category']:
            if features1['category'] == features2['category']:
                feature_similarity += weights['category']
            total_weight += weights['category']

        # Сравниваем размеры
        dim_match = 0
        dim_count = 0
        for dim in ['diameter', 'thickness', 'length', 'width', 'height', 'number']:
            if features1[dim] is not None and features2[dim] is not None:
                if abs(features1[dim] - features2[dim]) < 0.1:  # допуск 0.1
                    dim_match += 1
                dim_count += 1

        if dim_count > 0:
            feature_similarity += weights['dimensions'] * (dim_match / dim_count)
            total_weight += weights['dimensions']

        # Сравниваем материал
        if features1['material'] and features2['material']:
            if features1['material'] == features2['material']:
                feature_similarity += weights['material']
            total_weight += weights['material']

        # Сравниваем ГОСТ
        if features1['gost'] and features2['gost']:
            if features1['gost'] == features2['gost']:
                feature_similarity += weights['gost']
            total_weight += weights['gost']

        # Сравниваем модель
        if features1['model'] and features2['model']:
            model_sim = SequenceMatcher(None,
                                        features1['model'],
                                        features2['model']).ratio()
            feature_similarity += weights['model'] * model_sim
            total_weight += weights['model']

        # Комбинируем базовую схожесть и схожесть характеристик
        if total_weight > 0:
            final_similarity = 0.3 * base_similarity + 0.7 * (feature_similarity / total_weight)
        else:
            final_similarity = base_similarity

        return final_similarity

    @staticmethod
    def find_matching_products(
            target_product: str,
            product_list: List[str],
            threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Находит схожие товары из списка.

        Args:
            target_product: целевой товар
            product_list: список товаров для поиска
            threshold: минимальный порог схожести

        Returns:
            list: список кортежей (товар, коэффициент схожести)
        """
        matches = []

        for product in product_list:
            similarity = ProductMatcher.calculate_similarity(target_product, product)
            if similarity >= threshold:
                matches.append((product, similarity))

        # Сортируем по убыванию схожести
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    @staticmethod
    def compare_products(product1: str, product2: str) -> Dict[str, Any]:
        """
        Детальное сравнение двух товаров.

        Args:
            product1: первый товар
            product2: второй товар

        Returns:
            dict: результаты сравнения
        """
        features1 = ProductMatcher.extract_key_features(product1)
        features2 = ProductMatcher.extract_key_features(product2)

        similarity = ProductMatcher.calculate_similarity(product1, product2)

        comparison = {
            'similarity': similarity,
            'product1': product1,
            'product2': product2,
            'features1': features1,
            'features2': features2,
            'differences': {},
            'matches': {}
        }

        # Сравниваем каждую характеристику
        for key in features1.keys():
            if key in ['measurements', 'specifications']:
                continue

            val1 = features1[key]
            val2 = features2[key]

            if val1 is not None and val2 is not None:
                if val1 == val2:
                    comparison['matches'][key] = val1
                else:
                    comparison['differences'][key] = {
                        'product1': val1,
                        'product2': val2
                    }

        return comparison


# Функция для демонстрации
def demo():
    """Демонстрация работы модуля"""

    # Примеры товаров
    products = [
        "Труба стальная 219х6 ГОСТ 8732-78",
        "Труба 219x6 ст3 ГОСТ 8732",
        "Швеллер 20У ГОСТ 8240-89",
        "Кабель ВВГнг 3х2.5",
        "Светильник LED 36W IP65",
        "Болт М12х80 ГОСТ 7798",
        "Клапан DN50 PN16",
        "Насос ЦНС 60-198",
        "Огнетушитель ОП-5",
        "Перчатки рабочие х/б"
    ]

    print("=== ДЕМОНСТРАЦИЯ РАБОТЫ PRODUCT MATCHER ===\n")

    # Извлечение характеристик
    print("1. ИЗВЛЕЧЕНИЕ ХАРАКТЕРИСТИК\n")
    for product in products[:5]:
        print(f"Товар: {product}")
        features = ProductMatcher.extract_key_features(product)
        print(f"  Категория: {features['category']}")
        print(f"  Тип: {features['type']}")
        print(f"  Характеристики: {features}")
        print()

    # Поиск схожих товаров
    print("\n2. ПОИСК СХОЖИХ ТОВАРОВ\n")
    target = "Труба 219х6 мм стальная"
    print(f"Ищем схожие с: {target}\n")

    matches = ProductMatcher.find_matching_products(target, products, threshold=0.5)
    for product, similarity in matches:
        print(f"  {product}")
        print(f"  Схожесть: {similarity:.2%}\n")

    # Сравнение товаров
    print("\n3. ДЕТАЛЬНОЕ СРАВНЕНИЕ\n")
    comparison = ProductMatcher.compare_products(products[0], products[1])
    print(f"Товар 1: {comparison['product1']}")
    print(f"Товар 2: {comparison['product2']}")
    print(f"Общая схожесть: {comparison['similarity']:.2%}")
    print(f"\nСовпадения: {comparison['matches']}")
    print(f"Различия: {comparison['differences']}")


if __name__ == "__main__":
    demo()