"""
Модуль для сопоставления и сравнения товаров - ВЕРСИЯ 3.36

"""

import re
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher


class ProductMatcher:
    """Класс для сопоставления и анализа товаров"""

    # РАСШИРЕННЫЕ КАТЕГОРИИ с новыми типами товаров
    CATEGORIES = {
        'металлопрокат': [
            'труба', 'швеллер', 'двутавр', 'арматура', 'уголок', 'балка',
            'профиль', 'лист', 'полоса', 'круг', 'проволока', 'катанка',
            'тавр', 'прокат', 'металлоконструкция', 'сталь'
        ],
        'трубы_фитинги': [
            'труба', 'отвод', 'тройник', 'переход', 'фланец', 'муфта',
            'заглушка', 'колено', 'патрубок', 'штуцер', 'ниппель', 'футорка',
            'адаптер', 'американка', 'ппр', 'пвх', 'фитинги', 'фитинг'
        ],
        'крепеж': [
            'болт', 'гайка', 'шайба', 'винт', 'саморез', 'шуруп', 'гвоздь',
            'анкер', 'дюбель', 'шпилька', 'заклепка', 'скоба', 'хомут',
            'клипса', 'стяжка', 'талреп', 'зажим', 'коуш'
        ],
        'электрика': [
            'кабель', 'провод', 'розетка', 'выключатель', 'лампа', 'светильник',
            'щит', 'автомат', 'контактор', 'пускатель', 'реле', 'трансформатор',
            'счетчик', 'датчик', 'извещатель', 'прожектор', 'патрон',
            'коммутатор', 'адаптер sc', 'наконечник', 'изолятор'
        ],
        'сантехника': [
            'смеситель', 'унитаз', 'раковина', 'ванна', 'душ', 'сифон',
            'умывальник', 'писсуар', 'кран', 'вентиль', 'задвижка',
            'трап', 'решетка', 'мойка', 'гофра', 'ревизия', 'водонагреватель'
        ],
        'клапаны_арматура': [
            'клапан', 'вентиль', 'задвижка', 'кран', 'затвор', 'редуктор',
            'регулятор', 'манометр', 'термометр', 'счетчик', 'обратный',
            'предохранительный', 'запорный', 'головка пожарная'
        ],
        'строительные_материалы': [
            'цемент', 'бетон', 'раствор', 'смесь', 'песок', 'щебень', 'гравий',
            'кирпич', 'блок', 'плита', 'панель', 'гипсокартон', 'утеплитель',
            'минвата', 'пенопласт', 'пеноплекс', 'сэндвич-панель', 'сэндвич',
            'сендвич-панель', 'сендвич', 'профнастил', 'геотекстиль', 'пленка',
            'брезент', 'брус', 'доска', 'кольцо стеновое', 'днище', 'фундамент',
            'опора', 'стойка', 'клей', 'добавка', 'гипер', 'шпаклёвка'
        ],
        'лкм': [
            'краска', 'эмаль', 'грунтовка', 'лак', 'шпатлевка', 'герметик',
            'растворитель', 'разбавитель', 'олифа', 'мастика', 'праймер',
            'битум', 'смола'
        ],
        'инструмент': [
            'дрель', 'перфоратор', 'шуруповерт', 'болгарка', 'ушм', 'пила',
            'лобзик', 'фрезер', 'рубанок', 'молоток', 'ключ', 'отвертка',
            'плоскогубцы', 'кусачки', 'ножницы', 'нож', 'головка торцевая',
            'набор', 'сверло', 'диск', 'машина шлифовальная', 'машинка',
            'гайковерт', 'миксер', 'трубогиб', 'шарошка', 'бита'
        ],
        'грузоподъемное_оборудование': [
            'строп', 'стропа', 'стропы', 'таль', 'лебедка', 'домкрат',
            'серьга', 'захват', 'штабелер', 'стяжн', 'подвес', 'траверса'
        ],
        'измерительные_приборы': [
            'манометр', 'термометр', 'расходомер', 'счетчик', 'датчик',
            'уровень', 'рулетка', 'штангенциркуль', 'нивелир', 'мультиметр',
            'тестер', 'вольтметр', 'амперметр', 'линейка', 'зеркало'
        ],
        'спецодежда_сиз': [
            'костюм', 'комбинезон', 'куртка', 'брюки', 'халат', 'роба',
            'перчатки', 'каска', 'очки', 'респиратор', 'маска', 'противогаз',
            'беруши', 'наушники', 'каски', 'шлем', 'спецодежда', 'спец одежда',
            'спец обувь', 'фильтр'
        ],
        'пожарное_оборудование': [
            'огнетушитель', 'рукав', 'ствол', 'гидрант', 'кран', 'извещатель',
            'оповещатель', 'датчик', 'щит', 'шкаф', 'ящик', 'кошма',
            'полотнище противопожарное'
        ],
        'оборудование': [
            'насос', 'компрессор', 'генератор', 'трансформатор', 'двигатель',
            'редуктор', 'мотор', 'вентилятор', 'калорифер', 'котел',
            'вибратор', 'мотопомпа', 'пылесос'
        ],
        'сварочное_оборудование': [
            'аппарат сварочный', 'инвертор', 'плазморез', 'горелка', 'держак',
            'электрод', 'флюс', 'сопло', 'сварочный', 'сопло для краскопульта'
        ],
        'расходные_материалы': [
            'диск отрезной', 'диск шлифовальный', 'диск алмазный', 'круг',
            'наждачная', 'валик', 'кисть', 'щетка', 'лента', 'скотч',
            'серпянка', 'маркер', 'мел', 'крестики', 'ремкомплект', 'сальник',
            'прокладка', 'втулка', 'подшипник', 'ремень', 'фильтр', 'гильза'
        ],
        'хозяйственные_товары': [
            'веревка', 'канат', 'жилка', 'леска', 'пленка', 'брезент',
            'палатка', 'дверь', 'плинтус', 'соединитель', 'люк', 'колесо',
            'емкость', 'тележка', 'знак', 'знак дорожный', 'сетка'
        ],
        'газовое_оборудование': [
            'баллон', 'шланг', 'горелка', 'газовый', 'пропан', 'кислород',
            'аргон', 'азот', 'углекислота'
        ],
        'покрасочное_оборудование': [
            'краскораспылитель', 'пульверизатор', 'сопло', 'фильтр',
            'шланг высокого давления', 'трубка'
        ],
        'прочее': [
            'прочее', 'разное', 'иное', 'другое'
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
        """Извлекает категорию и тип товара"""
        category = None
        product_type = None

        # ====================================================================
        # ПРИОРИТЕТНАЯ ОБРАБОТКА: Товары, которые раньше попадали в None
        # ====================================================================

        # 1. ЭЛЕКТРОДЫ (141 товар) - КРИТИЧНО!
        if re.search(r'электрод', name_lower):
            return 'сварочное_оборудование', 'электрод'

        # 2. БУР (416 товаров) - КРИТИЧНО!
        if re.search(r'\bбур\b', name_lower):
            return 'инструмент', 'бур'

        # 3. МАНЖЕТА (195 товаров)
        if re.search(r'манжет', name_lower):
            return 'расходные_материалы', 'манжета'

        # 4. КОЛЬЦО (233 товара) - зависит от контекста
        if re.search(r'кольц[оа]', name_lower):
            # Уплотнительное кольцо → расходные
            if re.search(r'уплотнит|резинов', name_lower):
                return 'расходные_материалы', 'кольцо'
            # Стопорное кольцо → крепёж
            elif re.search(r'стопорн|крепёжн', name_lower):
                return 'крепеж', 'кольцо'
            # Кольцо стеновое → строительные
            elif re.search(r'стенов', name_lower):
                return 'строительные_материалы', 'кольцо стеновое'
            # Остальные → расходные
            else:
                return 'расходные_материалы', 'кольцо'

        # 5. БОБЫШКА (166 товаров)
        if re.search(r'бобышк', name_lower):
            return 'трубы_фитинги', 'бобышка'

        # 6. ВСТАВКА (149 товаров)
        if re.search(r'вставк', name_lower):
            # Резьбовая вставка → крепёж
            if re.search(r'резьбов|m\d+', name_lower):
                return 'крепеж', 'вставка'
            # Остальные → расходные
            else:
                return 'расходные_материалы', 'вставка'

        # 7. ДЕТАЛЬ (141 товар)
        if re.search(r'\bдеталь\b', name_lower):
            return 'металлопрокат', 'деталь'

        # 8. ЭЛЕМЕНТ (420 товаров!) - КРИТИЧНО!
        if re.search(r'элемент', name_lower):
            # Закладной элемент → строительные
            if re.search(r'закладн', name_lower):
                return 'строительные_материалы', 'элемент'
            # Нагревательный элемент → электрика
            elif re.search(r'нагреват|тэн', name_lower):
                return 'электрика', 'элемент'
            # Крепёжный элемент → крепёж
            elif re.search(r'крепёж|крепл', name_lower):
                return 'крепеж', 'элемент'
            # Остальные → строительные
            else:
                return 'строительные_материалы', 'элемент'

        # 9. ЦИЛИНДР (136 товаров)
        if re.search(r'цилиндр', name_lower):
            return 'оборудование', 'цилиндр'

        # 10. ПЛАНКА (120 товаров)
        if re.search(r'планк', name_lower):
            # Монтажная планка → крепёж
            if re.search(r'монтажн|крепёжн', name_lower):
                return 'крепеж', 'планка'
            # Остальные → металлопрокат
            else:
                return 'металлопрокат', 'планка'

        # 11. ШПАТЕЛЬ (112 товаров)
        if re.search(r'шпател', name_lower):
            return 'инструмент', 'шпатель'

        # 12. УГОЛЬНИК (111 товаров)
        if re.search(r'угольник', name_lower):
            # Измерительный угольник → измерительные
            if re.search(r'измерит|провер|контрольн', name_lower):
                return 'измерительные_приборы', 'угольник'
            # Соединительный угольник → крепёж
            elif re.search(r'соедин|крепёж', name_lower):
                return 'крепеж', 'угольник'
            # Остальные → металлопрокат
            else:
                return 'металлопрокат', 'угольник'

        # 13. РАДИАТОР (96 товаров)
        if re.search(r'радиатор', name_lower):
            # Отопительный радиатор → сантехника
            if re.search(r'отопит|биметалл|алюмин|секци', name_lower):
                return 'сантехника', 'радиатор'
            # Остальные → оборудование
            else:
                return 'оборудование', 'радиатор'

        # 14. ЛОТОК (96 товаров)
        if re.search(r'лоток', name_lower):
            # Кабельный лоток → электрика
            if re.search(r'кабельн|провод', name_lower):
                return 'электрика', 'лоток'
            # Водоотводный лоток → строительные
            elif re.search(r'водоотвод|дренаж|ливнев', name_lower):
                return 'строительные_материалы', 'лоток'
            # Остальные → строительные
            else:
                return 'строительные_материалы', 'лоток'

        # 15. ОБУВЬ (89 товаров)
        if re.search(r'обувь|ботинк|сапог', name_lower):
            return 'спецодежда_сиз', 'обувь'

        # 16. КОРОНКА (85 товаров)
        if re.search(r'коронк', name_lower):
            return 'инструмент', 'коронка'

        # 17. ОГРАЖДЕНИЕ (78 товаров)
        if re.search(r'огражден', name_lower):
            return 'строительные_материалы', 'ограждение'

        # 18. ЖУРНАЛ (78 товаров)
        if re.search(r'\bжурнал\b', name_lower):
            return 'хозяйственные_товары', 'журнал'

        # 19. КОМПЕНСАТОР (78 товаров)
        if re.search(r'компенсатор', name_lower):
            return 'трубы_фитинги', 'компенсатор'

        # 20. ДОПОЛНИТЕЛЬНЫЕ ПРОБЛЕМНЫЕ ТОВАРЫ

        # Держатель
        if re.search(r'держател', name_lower):
            return 'крепеж', 'держатель'

        # Накладка
        if re.search(r'накладк', name_lower):
            return 'металлопрокат', 'накладка'

        # Пластина
        if re.search(r'пластин', name_lower):
            return 'металлопрокат', 'пластина'

        # Настил
        if re.search(r'настил', name_lower):
            return 'строительные_материалы', 'настил'

        # Фланец (опечатка "флянец")
        if re.search(r'фл[яе]нец', name_lower):
            return 'трубы_фитинги', 'фланец'

        # Трубы (опечатка "tруба")
        if re.search(r'tруба', name_lower):
            return 'трубы_фитинги', 'труба'

        # ====================================================================
        # КОНЕЦ ПРИОРИТЕТНОЙ ОБРАБОТКИ
        # ====================================================================

        # Ищем совпадения по категориям (порядок важен - более специфичные категории первыми)
        category_priority = [
            'сварочное_оборудование',
            'грузоподъемное_оборудование',
            'покрасочное_оборудование',
            'газовое_оборудование',
            'расходные_материалы',
            'трубы_фитинги',
            'металлопрокат',
            'инструмент',
            'крепеж',
            'электрика',
            'сантехника',
            'клапаны_арматура',
            'строительные_материалы',
            'лкм',
            'измерительные_приборы',
            'спецодежда_сиз',
            'пожарное_оборудование',
            'оборудование',
            'хозяйственные_товары',
            'прочее'
        ]

        for cat in category_priority:
            keywords = ProductMatcher.CATEGORIES.get(cat, [])
            for keyword in keywords:
                # ИСПРАВЛЕНИЕ: ищем целое слово, а не подстроку!
                # "таль" не должна находиться в "стальная"
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, name_lower):
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
                    break
                except:
                    pass

        # ПРИОРИТЕТ 2: Диаметр с явным указанием (Ø, ф, d, диаметр)
        diameter_explicit_patterns = [
            r'[øф∅]\s*(\d+\.?\d*)',
            r'диаметр\s*[:\-]?\s*(\d+\.?\d*)',
            r'ду\s*(\d+\.?\d*)',  # DN (ДУ)
            r'dn\s*(\d+\.?\d*)',  # DN
            r'д\.?\s*(\d+\.?\d*)',  # д.219
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
                elif 'швеллер' in name_lower or 'двутавр' in name_lower:
                    # Профиль: игнорируем, используем номер
                    pass
                else:
                    # Обычные размеры
                    dimensions['length'] = val1
                    dimensions['width'] = val2
                    if val3:
                        dimensions['height'] = val3
            except:
                pass

        # ПРИОРИТЕТ 4: Одиночное число с "мм"
        if (dimensions['diameter'] is None and
                dimensions['thickness'] is None and
                dimensions['length'] is None):

            single_mm_pattern = r'(\d+\.?\d*)\s*мм'
            match = re.search(single_mm_pattern, name_lower)
            if match:
                try:
                    val = float(match.group(1))

                    # Если есть слова типа "толщ", "толст" - это толщина
                    if re.search(r'толщ|толст', name_lower):
                        dimensions['thickness'] = val
                    # Если есть "диам" - это диаметр
                    elif re.search(r'диам|ø|ф|ду|dn', name_lower):
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
            'пвх', 'пнд', 'пп', 'пэ', 'полиэтилен', 'полипропилен', 'ппр',
            'резина', 'резиновый', 'резиновая',
            'бетон', 'железобетон', 'ж/б', 'жби',
            'дерево', 'деревянный', 'деревянная',
            'пластик', 'пластиковый', 'пластиковая',
            'текстильн', 'канат', 'веревк'
        ]

        for material in materials:
            if material in name_lower:
                return material

        return None

    @staticmethod
    def _extract_gost(name_lower: str) -> Optional[str]:
        """Извлекает ГОСТ/ТУ/ОСТ"""
        patterns = [
            r'гост\s*[\s\-]?\s*[\d\-\.]+',
            r'ту\s*[\s\-]?\s*[\d\-\.]+',
            r'ост\s*[\s\-]?\s*[\d\-\.]+',
            r'сто\s*[\s\-]?\s*[\d\-\.]+',
            r'din\s*[\s\-]?\s*[\d\-\.]+',
            r'iso\s*[\s\-]?\s*[\d\-\.]+',
        ]

        for pattern in patterns:
            match = re.search(pattern, name_lower)
            if match:
                return match.group(0).strip()

        return None

    @staticmethod
    def _extract_model(name_lower: str) -> Optional[str]:
        """Извлекает модель/артикул"""
        patterns = [
            r'ral\s*\d+',
            r'серия\s*[\d\-\.]+',
            r'тип\s*[\w\-]+',
            r'модель\s*[\w\-]+',
            r'арт\.?\s*[\w\-]+',
            r'[а-яa-z]+\-\d+',  # типа СТП-10
        ]

        for pattern in patterns:
            match = re.search(pattern, name_lower)
            if match:
                return match.group(0).strip()

        return None

    @staticmethod
    def _extract_brand(name_lower: str) -> Optional[str]:
        """Извлекает бренд"""
        brands = [
            'bosch', 'makita', 'dewalt', 'hitachi', 'metabo', 'milwaukee',
            'legrand', 'schneider', 'abb', 'iek',
            'knauf', 'ceresit', 'henkel', 'sika',
            'grohe', 'hansgrohe', 'ideal standard', 'valtec',
            'ресанта', 'сварог', 'jasic', 'esab', 'патон',
            'graco', 'wagner', 'kronen', 'yato',
            'зубр', 'matrix', 'sturm', 'force'
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

        # IP класс
        ip_match = re.search(r'ip\s*\d+', name_lower)
        if ip_match:
            specifications.append(ip_match.group(0))

        # PN (давление)
        pn_match = re.search(r'pn\s*\d+', name_lower)
        if pn_match:
            specifications.append(pn_match.group(0))

        # DN (диаметр номинальный)
        dn_match = re.search(r'dn\s*\d+', name_lower)
        if dn_match:
            specifications.append(dn_match.group(0))

        # Класс точности/прочности
        class_match = re.search(r'кл\.?[\s\-]?\w+', name_lower)
        if class_match:
            specifications.append(class_match.group(0))

        return specifications

    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        """
        Рассчитывает схожесть между двумя товарами

        ВЕРСИЯ 3.0 - УЛУЧШЕННЫЙ АЛГОРИТМ С ПРОВЕРКОЙ КРИТИЧНЫХ ПАРАМЕТРОВ:
        - Учитывает категорию и тип
        - Сравнивает размеры с допуском
        - НОВОЕ: Проверяет критичные параметры (грузоподъемность, длину, диаметр, сечение)
        - Проверяет материал и ГОСТ
        - Использует взвешенную сумму факторов

        Args:
            name1: название первого товара
            name2: название второго товара

        Returns:
            float: коэффициент схожести (0-1)
        """
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        # КРИТИЧНАЯ ПРОВЕРКА: Если есть явные различия в ключевых параметрах - сразу низкая схожесть
        critical_penalty = ProductMatcher._check_critical_parameters(name1_lower, name2_lower)
        if critical_penalty < 0.5:
            # Есть критичные различия - возвращаем низкую схожесть
            return critical_penalty

        # Извлекаем характеристики
        features1 = ProductMatcher.extract_key_features(name1)
        features2 = ProductMatcher.extract_key_features(name2)

        # Базовая схожесть строк (с нормализацией)
        normalized1 = ProductMatcher._normalize_string(name1)
        normalized2 = ProductMatcher._normalize_string(name2)
        string_similarity = SequenceMatcher(None, normalized1, normalized2).ratio()

        # Факторы схожести
        factors = {}

        # 1. Категория (вес 25%)
        if features1['category'] == features2['category'] and features1['category'] is not None:
            factors['category'] = 1.0
        else:
            factors['category'] = 0.0

        # 2. Тип товара (вес 25%)
        if features1['type'] == features2['type'] and features1['type'] is not None:
            factors['type'] = 1.0
        elif features1['type'] and features2['type']:
            # Частичное совпадение типа
            type_sim = SequenceMatcher(None, features1['type'], features2['type']).ratio()
            factors['type'] = type_sim
        else:
            factors['type'] = 0.0

        # 3. Размеры (вес 20%)
        size_sim = ProductMatcher._compare_dimensions(features1, features2)
        factors['size'] = size_sim

        # 4. Материал (вес 10%)
        if features1['material'] and features2['material']:
            if features1['material'] == features2['material']:
                factors['material'] = 1.0
            else:
                factors['material'] = 0.5  # Разные материалы, но оба указаны
        else:
            factors['material'] = 0.7  # Материал не указан - нейтрально

        # 5. ГОСТ (вес 10%)
        if features1['gost'] and features2['gost']:
            if features1['gost'] == features2['gost']:
                factors['gost'] = 1.0
            else:
                factors['gost'] = 0.3  # Разные ГОСТы - низкая схожесть
        else:
            factors['gost'] = 0.7  # ГОСТ не указан - нейтрально

        # 6. Базовая схожесть строк (вес 10%)
        factors['string'] = string_similarity

        # Взвешенная сумма
        weights = {
            'category': 0.25,
            'type': 0.25,
            'size': 0.20,
            'material': 0.10,
            'gost': 0.10,
            'string': 0.10
        }

        total_similarity = sum(factors[key] * weights[key] for key in weights.keys())

        # Корректировка: если категории разные - сильно снижаем схожесть
        if factors['category'] == 0.0 and features1['category'] and features2['category']:
            total_similarity *= 0.6  # Снижаем на 40%

        # Корректировка: если типы сильно разные - тоже снижаем
        if factors['type'] < 0.5:
            total_similarity *= 0.8  # Снижаем на 20%

        # Применяем штраф за критичные параметры
        total_similarity *= critical_penalty

        return min(1.0, total_similarity)

    @staticmethod
    def _check_critical_parameters(name1: str, name2: str) -> float:
        """
        НОВАЯ ФУНКЦИЯ: Проверяет критичные параметры товаров

        Возвращает коэффициент (0-1):
        - 1.0 = параметры совпадают или не указаны
        - 0.3-0.7 = параметры различаются незначительно
        - 0.0-0.3 = параметры сильно различаются (это РАЗНЫЕ товары!)

        Returns:
            float: коэффициент схожести по критичным параметрам
        """
        import re

        penalty = 1.0  # Начинаем с 1.0 (нет штрафа)

        # 1. РУЛЕТКИ - проверяем длину
        if 'рулетка' in name1 and 'рулетка' in name2:
            nums1 = re.findall(r'(\d+)\s*м(?![а-яё])', name1)
            nums2 = re.findall(r'(\d+)\s*м(?![а-яё])', name2)

            if nums1 and nums2:
                len1 = int(nums1[0])
                len2 = int(nums2[0])

                if len1 != len2:
                    # Разная длина - это РАЗНЫЕ товары!
                    ratio = min(len1, len2) / max(len1, len2)
                    penalty = min(penalty, 0.3 + ratio * 0.2)  # Максимум 0.5

        # 2. СТРОПЫ - проверяем грузоподъемность
        if 'строп' in name1 and 'строп' in name2:
            # Ищем грузоподъемность: 2т, 2,0т, 2.5т, СТП 2,0, или просто "16тн", "2тн"
            nums1 = re.findall(r'(?:стп|скп|усk|сцк|сск)\s*(\d+(?:[.,]\d+)?)', name1)
            if not nums1:
                # Ищем "16тн", "2тн", "10т" и т.д. с учетом "тн" и "т"
                nums1 = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:тн|т)(?![а-я])', name1, re.IGNORECASE)

            nums2 = re.findall(r'(?:стп|скп|усk|сцк|сск)\s*(\d+(?:[.,]\d+)?)', name2)
            if not nums2:
                nums2 = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:тн|т)(?![а-я])', name2, re.IGNORECASE)

            if nums1 and nums2:
                t1 = float(nums1[0].replace(',', '.'))
                t2 = float(nums2[0].replace(',', '.'))

                diff = abs(t1 - t2)

                if diff > 0.5:  # Разница больше 0.5т
                    ratio = min(t1, t2) / max(t1, t2)
                    penalty = min(penalty, 0.35 + ratio * 0.25)  # Максимум 0.6

        # 3. ТАЛИ - проверяем грузоподъемность И длину
        if 'таль' in name1 and 'таль' in name2:
            # Грузоподъемность: 5Т, 5т, 5ТН, 5тн, 5,0 тн
            nums1 = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:тн|т[хx]?)(?![а-я])', name1, re.IGNORECASE)
            nums2 = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:тн|т[хx]?)(?![а-я])', name2, re.IGNORECASE)

            if nums1 and nums2:
                t1 = float(nums1[0].replace(',', '.'))
                t2 = float(nums2[0].replace(',', '.'))

                diff = abs(t1 - t2)

                if diff > 0.5:
                    ratio = min(t1, t2) / max(t1, t2)
                    penalty = min(penalty, 0.35 + ratio * 0.25)  # Максимум 0.6

            # Длина: 3М, 12м
            len1 = re.findall(r'(\d+)[мМ](?![а-яА-Я])', name1)
            len2 = re.findall(r'(\d+)[мМ](?![а-яА-Я])', name2)

            if len1 and len2:
                l1 = int(len1[0])
                l2 = int(len2[0])

                if abs(l1 - l2) > 1:  # Разница больше 1 метра
                    ratio = min(l1, l2) / max(l1, l2)
                    penalty = min(penalty, 0.5 + ratio * 0.3)  # Максимум 0.8

        # 4. ДОМКРАТЫ - проверяем ТИП, грузоподъемность и МОДЕЛЬ
        if 'домкрат' in name1 and 'домкрат' in name2:
            # Определяем тип домкрата
            type1 = None
            type2 = None

            # Приоритет проверки (от более специфичного к менее):
            # 1. Пневмогидравлический
            if 'пневмогидравл' in name1 or 'пневмо-гидравл' in name1:
                type1 = 'пневмогидравлический'
            # 2. Гидравлический
            elif 'гидравл' in name1:
                type1 = 'гидравлический'
            # 3. Реечный
            elif 'реечн' in name1 or re.search(r'др[-\s]*\d', name1):
                type1 = 'реечный'
            # 4. Подкатной
            elif 'подкатн' in name1:
                type1 = 'подкатной'
            # 5. Винтовой
            elif 'винтов' in name1:
                type1 = 'винтовой'
            # 6. Кабельный
            elif 'кабельн' in name1 or re.search(r'дк[бг]?[-\s]*\d', name1):
                type1 = 'кабельный'
            # 7. Резьбовой
            elif 'резьбов' in name1:
                type1 = 'резьбовой'
            # 8. Тепловозный
            elif 'тепловоз' in name1 or re.search(r'дт[-\s]*\d', name1):
                type1 = 'тепловозный'

            # То же для name2
            if 'пневмогидравл' in name2 or 'пневмо-гидравл' in name2:
                type2 = 'пневмогидравлический'
            elif 'гидравл' in name2:
                type2 = 'гидравлический'
            elif 'реечн' in name2 or re.search(r'др[-\s]*\d', name2):
                type2 = 'реечный'
            elif 'подкатн' in name2:
                type2 = 'подкатной'
            elif 'винтов' in name2:
                type2 = 'винтовой'
            elif 'кабельн' in name2 or re.search(r'дк[бг]?[-\s]*\d', name2):
                type2 = 'кабельный'
            elif 'резьбов' in name2:
                type2 = 'резьбовой'
            elif 'тепловоз' in name2 or re.search(r'дт[-\s]*\d', name2):
                type2 = 'тепловозный'

            # Если типы определены и РАЗНЫЕ - это РАЗНЫЕ товары!
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.30)  # Реечный vs гидравлический = РАЗНЫЕ

            # Проверяем грузоподъемность (только если типы одинаковые или не определены)
            if not type1 or not type2 or type1 == type2:
                nums1 = re.findall(r'(\d+)\s*(?:тн|т)(?![а-я])', name1, re.IGNORECASE)
                nums2 = re.findall(r'(\d+)\s*(?:тн|т)(?![а-я])', name2, re.IGNORECASE)

                if nums1 and nums2:
                    t1 = int(nums1[0])
                    t2 = int(nums2[0])
                    diff = abs(t1 - t2)

                    # Для домкратов - строгое сравнение
                    if diff >= 10:  # Разница 10т и более - очень разные
                        penalty = min(penalty, 0.2)
                    elif diff >= 5:  # Разница 5т и более - разные
                        ratio = min(t1, t2) / max(t1, t2)
                        penalty = min(penalty, 0.3 + ratio * 0.2)  # Максимум 0.5
                    elif diff > 0:  # Небольшая разница - средняя схожесть
                        ratio = min(t1, t2) / max(t1, t2)
                        penalty = min(penalty, 0.6 + ratio * 0.2)  # Максимум 0.8

            # НОВОЕ: Проверяем МОДЕЛЬ/АРТИКУЛ (если тип и грузоподъемность одинаковые)
            # Извлекаем модели: ДР-5000, SJ5, HHYG-50B, ДР-10000, и т.д.
            model1 = None
            model2 = None

            # Паттерны моделей:
            # ДР-5000, ДР-10000, ДКБ-10, ДК-5, ДТ-40
            m1 = re.search(r'д[ртк][бг]?[-\s]*\d+', name1, re.IGNORECASE)
            if m1:
                model1 = m1.group(0).upper().replace(' ', '-')

            # SJ5, HHYG-50B, HHYG-100B, и другие буквенно-цифровые
            if not model1:
                m1 = re.search(r'\b([A-Z]{2,}[-]?\d+[A-Z]?)\b', name1, re.IGNORECASE)
                if m1:
                    model1 = m1.group(1).upper()

            # TOR (бренд/модель)
            if not model1 and 'tor' in name1.lower():
                model1 = 'TOR'

            # То же для name2
            m2 = re.search(r'д[ртк][бг]?[-\s]*\d+', name2, re.IGNORECASE)
            if m2:
                model2 = m2.group(0).upper().replace(' ', '-')

            if not model2:
                m2 = re.search(r'\b([A-Z]{2,}[-]?\d+[A-Z]?)\b', name2, re.IGNORECASE)
                if m2:
                    model2 = m2.group(1).upper()

            if not model2 and 'tor' in name2.lower():
                model2 = 'TOR'

            # Если модели определены и РАЗНЫЕ - снижаем схожесть
            if model1 and model2 and model1 != model2:
                # Разные модели при одинаковом типе и грузоподъемности
                # Это могут быть разные товары с разными характеристиками
                penalty = min(penalty, 0.70)  # Снижаем до 70% (не дубликаты, но похожи)

        # 5. ЛЕБЕДКИ - проверяем грузоподъемность
        if 'лебедка' in name1 and 'лебедка' in name2:
            nums1 = re.findall(r'(\d+(?:[.,]\d+)?)\s*[тТ]', name1)
            nums2 = re.findall(r'(\d+(?:[.,]\d+)?)\s*[тТ]', name2)

            if nums1 and nums2:
                t1 = float(nums1[0].replace(',', '.'))
                t2 = float(nums2[0].replace(',', '.'))

                if abs(t1 - t2) > 0.5:
                    ratio = min(t1, t2) / max(t1, t2)
                    penalty = min(penalty, 0.4 + ratio * 0.3)

        # 6. КАБЕЛИ - проверяем сечение
        if 'кабель' in name1 and 'кабель' in name2:
            # Ищем сечение типа 3х2.5
            nums1 = re.findall(r'(\d+)\s*х\s*(\d+[.,]?\d*)', name1)
            nums2 = re.findall(r'(\d+)\s*х\s*(\d+[.,]?\d*)', name2)

            if nums1 and nums2:
                # Сравниваем и количество жил и сечение
                if nums1[0] != nums2[0]:
                    penalty = min(penalty, 0.4)  # Разное сечение - очень низкая схожесть

        # 7. ДИСКИ - проверяем диаметр
        if 'диск' in name1 and 'диск' in name2:
            # Ищем стандартные диаметры 125, 180, 230 и другие
            # Проверяем формат: 125х..., d-180, Ø230, "180 мм"
            nums1 = re.findall(r'(?:^|[^\d])(\d{3})(?:х|мм|$|[^\d])', name1)
            if not nums1:
                nums1 = re.findall(r'd[-\s]*(\d{3})', name1)
            if not nums1:
                nums1 = re.findall(r'[øØ]\s*(\d{3})', name1)

            nums2 = re.findall(r'(?:^|[^\d])(\d{3})(?:х|мм|$|[^\d])', name2)
            if not nums2:
                nums2 = re.findall(r'd[-\s]*(\d{3})', name2)
            if not nums2:
                nums2 = re.findall(r'[øØ]\s*(\d{3})', name2)

            if nums1 and nums2 and nums1[0] != nums2[0]:
                d1 = int(nums1[0])
                d2 = int(nums2[0])
                # Для дисков разный диаметр - это ОЧЕНЬ РАЗНЫЕ товары
                ratio = min(d1, d2) / max(d1, d2)
                penalty = min(penalty, 0.2 + ratio * 0.2)  # Максимум 0.4

        # 8. ЭЛЕКТРОДЫ - проверяем диаметр
        if 'электрод' in name1 and 'электрод' in name2:
            # Ищем диаметр: д-3, 3мм, д.3
            nums1 = re.findall(r'д[.-]?\s*(\d+)', name1)
            nums2 = re.findall(r'д[.-]?\s*(\d+)', name2)

            if not nums1:
                nums1 = re.findall(r'(\d+)\s*мм', name1)
            if not nums2:
                nums2 = re.findall(r'(\d+)\s*мм', name2)

            if nums1 and nums2 and nums1[0] != nums2[0]:
                # Разный диаметр электрода - это РАЗНЫЕ товары
                penalty = min(penalty, 0.3)

        # 9. ШЛАНГИ - проверяем диаметр
        if 'шланг' in name1 and 'шланг' in name2:
            # Ищем диаметр
            nums1 = re.findall(r'd\s*(\d+)', name1)
            nums2 = re.findall(r'd\s*(\d+)', name2)

            if not nums1:
                nums1 = re.findall(r'(\d+)\s*мм', name1)
            if not nums2:
                nums2 = re.findall(r'(\d+)\s*мм', name2)

            if nums1 and nums2 and nums1[0] != nums2[0]:
                ratio = min(int(nums1[0]), int(nums2[0])) / max(int(nums1[0]), int(nums2[0]))
                penalty = min(penalty, 0.4 + ratio * 0.3)

        # 10. КОЛЬЦА СТЕНОВЫЕ - проверяем размеры (диаметр и высоту)
        if 'кольцо стеновое' in name1 and 'кольцо стеновое' in name2:
            # Формат: КЦ 7-3, КС-15-9, где первое число - диаметр, второе - высота
            nums1 = re.findall(r'к[цс][\s\-]*(\d+)[\s\-\.]+(\d+)', name1, re.IGNORECASE)
            nums2 = re.findall(r'к[цс][\s\-]*(\d+)[\s\-\.]+(\d+)', name2, re.IGNORECASE)

            if nums1 and nums2:
                d1, h1 = int(nums1[0][0]), int(nums1[0][1])
                d2, h2 = int(nums2[0][0]), int(nums2[0][1])

                # Если размеры разные - это РАЗНЫЕ товары!
                if d1 != d2 or h1 != h2:
                    # Очень низкая схожесть для колец разных размеров
                    penalty = min(penalty, 0.25)

        # 11. СТРОПЫ - дополнительная проверка ДЛИНЫ при одинаковой грузоподъемности
        if 'строп' in name1 and 'строп' in name2:
            # Ищем длину: /3000, /6000, 6М, 10м и т.д.
            l1_match = re.findall(r'/(\d+)', name1)
            if not l1_match:
                l1_match = re.findall(r'(\d+)[мМ](?![а-яА-Я])', name1)

            l2_match = re.findall(r'/(\d+)', name2)
            if not l2_match:
                l2_match = re.findall(r'(\d+)[мМ](?![а-яА-Я])', name2)

            if l1_match and l2_match:
                l1 = int(l1_match[0])
                l2 = int(l2_match[0])

                # Нормализуем длину (6000мм = 6м)
                if l1 >= 1000:
                    l1 = l1 / 1000
                if l2 >= 1000:
                    l2 = l2 / 1000

                # Если длина сильно отличается - снижаем схожесть
                if abs(l1 - l2) > 1:  # Разница больше 1 метра
                    ratio = min(l1, l2) / max(l1, l2)
                    penalty = min(penalty, 0.5 + ratio * 0.3)  # Максимум 0.8

        # 12. СТРОПЫ - проверка ТИПА (текстильный vs канатный vs цепной)
        if 'строп' in name1 and 'строп' in name2:
            # Определяем тип стропа
            type1 = None
            type2 = None

            # Текстильный: ТОЛЬКО СТП или явное упоминание "текстильн"
            if 'текстильн' in name1 or re.search(r'стп(?![а-я])', name1):
                type1 = 'текстильный'
            # Канатный: СКП (Строп Канатный Петлевой), СКК, или 4СК, 2СК и т.д., или явное упоминание "канатн"
            elif 'канатн' in name1 or re.search(r'с[кк]п(?![а-я])', name1) or re.search(r'с[кк][кк](?![а-я])', name1) or re.search(r'\d+с[кк]', name1):
                type1 = 'канатный'
            # Цепной: СЦК, или явное упоминание "цепн"
            elif 'цепн' in name1 or re.search(r'сцк(?![а-я])', name1):
                type1 = 'цепной'

            if 'текстильн' in name2 or re.search(r'стп(?![а-я])', name2):
                type2 = 'текстильный'
            elif 'канатн' in name2 or re.search(r'с[кк]п(?![а-я])', name2) or re.search(r'с[кк][кк](?![а-я])', name2) or re.search(r'\d+с[кк]', name2):
                type2 = 'канатный'
            elif 'цепн' in name2 or re.search(r'сцк(?![а-я])', name2):
                type2 = 'цепной'

            # Если типы определены и разные - это РАЗНЫЕ товары!
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.30)  # Очень низкая схожесть

        # 13. СВЕРЛА - проверяем диаметр
        if 'сверло' in name1 and 'сверло' in name2:
            nums1 = re.findall(r'(\d+(?:\.\d+)?)\s*мм', name1)
            nums2 = re.findall(r'(\d+(?:\.\d+)?)\s*мм', name2)

            if not nums1:
                nums1 = re.findall(r'd\s*(\d+(?:\.\d+)?)', name1)
            if not nums2:
                nums2 = re.findall(r'd\s*(\d+(?:\.\d+)?)', name2)

            if nums1 and nums2:
                d1 = float(nums1[0])
                d2 = float(nums2[0])

                if abs(d1 - d2) > 0.5:
                    ratio = min(d1, d2) / max(d1, d2)
                    penalty = min(penalty, 0.4 + ratio * 0.3)

        # 14. ШВЕЛЛЕР - проверяем номер
        if 'швеллер' in name1.lower() and 'швеллер' in name2.lower():
            # Ищем номер: "Швеллер 10", "Швеллер 20У"
            num1 = re.search(r'швеллер\s*(\d+\.?\d*)', name1.lower())
            num2 = re.search(r'швеллер\s*(\d+\.?\d*)', name2.lower())

            if num1 and num2:
                n1 = float(num1.group(1))
                n2 = float(num2.group(1))

                # Если номера разные - это РАЗНЫЕ товары!
                if n1 != n2:
                    penalty = min(penalty, 0.30)  # Швеллер 10 vs 20 = РАЗНЫЕ

        # 15. ДВУТАВР/БАЛКА - проверяем номер
        if ('двутавр' in name1.lower() or 'балка' in name1.lower()) and \
                ('двутавр' in name2.lower() or 'балка' in name2.lower()):
            # Ищем номер: "Двутавр 10", "Двутавр 20Б1", "Балка двутавровая 16"
            num1 = re.search(r'(?:двутавр|балка)\s*(?:двутавровая)?\s*(\d+\.?\d*)', name1.lower())
            num2 = re.search(r'(?:двутавр|балка)\s*(?:двутавровая)?\s*(\d+\.?\d*)', name2.lower())

            if num1 and num2:
                n1 = float(num1.group(1))
                n2 = float(num2.group(1))

                # Если номера разные - это РАЗНЫЕ товары!
                if n1 != n2:
                    penalty = min(penalty, 0.30)  # Двутавр 10 vs 20 = РАЗНЫЕ, Балка 16 vs 30 = РАЗНЫЕ

        # 16. АРМАТУРА - проверяем диаметр И класс
        if 'арматура' in name1.lower() and 'арматура' in name2.lower():
            # Диаметр: "Арматура 12", "ф12", "диам. 12 мм"
            d1 = re.search(r'(?:арматура|ф|диам\.?)\s*(\d+\.?\d*)', name1.lower())
            d2 = re.search(r'(?:арматура|ф|диам\.?)\s*(\d+\.?\d*)', name2.lower())

            if d1 and d2:
                diameter1 = float(d1.group(1))
                diameter2 = float(d2.group(1))

                # Если диаметр разный - это РАЗНЫЕ товары!
                if abs(diameter1 - diameter2) >= 2:  # Разница 2мм и более
                    penalty = min(penalty, 0.30)  # Арматура 10мм vs 16мм = РАЗНЫЕ

            # Класс: "А500", "А3", "А-III", "А-I"
            k1 = re.search(r'а-?([ivxIVX\d]+)', name1.lower())
            k2 = re.search(r'а-?([ivxIVX\d]+)', name2.lower())

            if k1 and k2:
                # Нормализуем: А-III -> А3, А-I -> А1
                class1 = k1.group(1).upper().replace('III', '3').replace('II', '2').replace('I', '1').replace('V', '5')
                class2 = k2.group(1).upper().replace('III', '3').replace('II', '2').replace('I', '1').replace('V', '5')

                # Если класс разный - это РАЗНЫЕ товары!
                if class1 != class2:
                    penalty = min(penalty, 0.30)  # А1 vs А3 = РАЗНЫЕ

        # 17. ЛИСТ - проверяем толщину и марку стали
        if 'лист' in name1.lower() and 'лист' in name2.lower():
            # Ищем толщину: "Лист 5", "Лист 12 ГОСТ", "Лист 16 С345"
            # Паттерны: "Лист 12", "Лист рифленый 5х1500х6000", "Лист 6х1500х6000"

            # Сначала пробуем найти толщину в формате "лист число"
            thick1 = re.search(r'лист\s+(?:рифленый\s+)?(\d+(?:[.,]\d+)?)', name1.lower())
            thick2 = re.search(r'лист\s+(?:рифленый\s+)?(\d+(?:[.,]\d+)?)', name2.lower())

            if thick1 and thick2:
                t1 = float(thick1.group(1).replace(',', '.'))
                t2 = float(thick2.group(1).replace(',', '.'))

                # Проверка толщины (КРИТИЧНО для листов!)
                # Даже 1мм разница критична!
                if abs(t1 - t2) >= 1:  # Лист 5мм vs 6мм = РАЗНЫЕ
                    penalty = min(penalty, 0.30)

            # Проверка марки стали: Ст3, С245, С345 и т.д.
            # Ст3 vs С245 = разные, С245 vs С345 = разные
            grade1 = None
            grade2 = None

            # Ищем: Ст3, Ст3сп5, С245, С345 и т.д.
            g1_match = re.search(r'(?:ст|с)(\d+)', name1.lower())
            g2_match = re.search(r'(?:ст|с)(\d+)', name2.lower())

            if g1_match:
                grade1 = int(g1_match.group(1))
            if g2_match:
                grade2 = int(g2_match.group(1))

            # Если марки найдены и различаются значительно
            if grade1 and grade2:
                # Ст3 (3) vs С345 (345) = РАЗНЫЕ
                # С245 (245) vs С345 (345) = могут быть допустимыми, но лучше считать разными
                if abs(grade1 - grade2) > 50:  # Большая разница в марке
                    penalty = min(penalty, 0.40)

        # 18. ТРУБА - проверяем материал, назначение, диаметр, толщину и SDR
        if 'труба' in name1.lower() and 'труба' in name2.lower():
            # СНАЧАЛА проверяем трубы с одиночным числом: "труба 159,4", "труба 219"
            # Эти трубы могут не иметь указания толщины, только диаметр
            single_dim1 = re.search(r'труба\s+(\d+(?:[.,]\d+)?)', name1.lower())
            single_dim2 = re.search(r'труба\s+(\d+(?:[.,]\d+)?)', name2.lower())

            # Проверка диаметра и толщины: 100х5, 100*5, 100×5, 159х8
            # Поддерживаем: х (латинская), * (звездочка), × (знак умножения)
            dims1 = re.search(r'(\d+)\s*[х*×x]\s*(\d+(?:[.,]\d+)?)', name1.lower())
            dims2 = re.search(r'(\d+)\s*[х*×x]\s*(\d+(?:[.,]\d+)?)', name2.lower())

            # Если есть формат диаметр×толщина
            if dims1 and dims2:
                d1 = float(dims1.group(1))
                t1 = float(dims1.group(2).replace(',', '.'))
                d2 = float(dims2.group(1))
                t2 = float(dims2.group(2).replace(',', '.'))

                # Проверка диаметра (КРИТИЧНО!)
                # Даже 5мм разница критична для труб!
                if abs(d1 - d2) >= 5:  # Труба 108 vs 114 (разница 6мм) = РАЗНЫЕ
                    penalty = min(penalty, 0.30)

                # Проверка толщины стенки
                # Для стальных труб - строгая проверка (даже 1мм критична!)
                is_steel1 = 'сталь' in name1.lower() or 'гост' in name1.lower() or 'ст2' in name1.lower() or 'ст3' in name1.lower()
                is_steel2 = 'сталь' in name2.lower() or 'гост' in name2.lower() or 'ст2' in name2.lower() or 'ст3' in name2.lower()

                if is_steel1 and is_steel2:
                    # Для стальных труб - толщина должна совпадать точно или различаться не более чем на 1мм
                    if abs(t1 - t2) >= 1:  # 377×6 vs 377×5 = РАЗНЫЕ
                        penalty = min(penalty, 0.30)
                else:
                    # Для остальных труб (ПВХ, ПНД и т.д.) - разница >2мм
                    if abs(t1 - t2) > 2:  # Разница больше 2мм
                        penalty = min(penalty, 0.40)  # Труба х5 vs х8 = разные

            # Если формата диаметр×толщина НЕТ, но есть одиночное число (только диаметр)
            elif single_dim1 and single_dim2:
                d1 = float(single_dim1.group(1).replace(',', '.'))
                d2 = float(single_dim2.group(1).replace(',', '.'))

                # Проверяем только диаметр (толщина неизвестна)
                if abs(d1 - d2) >= 5:  # Труба 159,4 vs 219 = РАЗНЫЕ
                    penalty = min(penalty, 0.30)

            # Или одна труба с форматом ×, другая без
            elif (dims1 and single_dim2 and not dims2) or (dims2 and single_dim1 and not dims1):
                # Одна труба: "108х4", другая: "114" (без толщины)
                # Извлекаем диаметры
                d1 = float(dims1.group(1)) if dims1 else float(single_dim1.group(1).replace(',', '.'))
                d2 = float(dims2.group(1)) if dims2 else float(single_dim2.group(1).replace(',', '.'))

                # Сравниваем диаметры
                if abs(d1 - d2) >= 5:
                    penalty = min(penalty, 0.30)

            # Проверка материала
            material1 = None
            material2 = None

            # ПНД/ПЭ/HDPE/PE80/PE100 (полиэтилен)
            if 'пнд' in name1.lower() or 'пэ' in name1.lower() or 'полиэтилен' in name1.lower() or 'hdpe' in name1.lower() or re.search(r'pe\s*\d+', name1.lower()):
                material1 = 'полиэтилен'
            # ППР/PPR/PP-R (полипропилен)
            elif 'ппр' in name1.lower() or 'полипропилен' in name1.lower() or 'pp-r' in name1.lower() or 'ppr' in name1.lower():
                material1 = 'полипропилен'
            # ПВХ/PVC
            elif 'пвх' in name1.lower() or 'поливинилхлорид' in name1.lower() or 'pvc' in name1.lower():
                material1 = 'пвх'
            # PEX (сшитый полиэтилен)
            elif 'pex' in name1.lower() or 'сшит' in name1.lower():
                material1 = 'pex'
            # Медная
            elif 'медн' in name1.lower():
                material1 = 'медь'
            # Стальная (по умолчанию, если есть ГОСТ или "стальная")
            elif 'сталь' in name1.lower() or 'гост' in name1.lower() or re.search(r'\d+х\d+', name1):
                material1 = 'сталь'

            # То же для name2
            if 'пнд' in name2.lower() or 'пэ' in name2.lower() or 'полиэтилен' in name2.lower() or 'hdpe' in name2.lower() or re.search(r'pe\s*\d+', name2.lower()):
                material2 = 'полиэтилен'
            elif 'ппр' in name2.lower() or 'полипропилен' in name2.lower() or 'pp-r' in name2.lower() or 'ppr' in name2.lower():
                material2 = 'полипропилен'
            elif 'пвх' in name2.lower() or 'поливинилхлорид' in name2.lower() or 'pvc' in name2.lower():
                material2 = 'пвх'
            elif 'pex' in name2.lower() or 'сшит' in name2.lower():
                material2 = 'pex'
            elif 'медн' in name2.lower():
                material2 = 'медь'
            elif 'сталь' in name2.lower() or 'гост' in name2.lower() or re.search(r'\d+х\d+', name2):
                material2 = 'сталь'

            # Если материалы определены и разные - это РАЗНЫЕ товары!
            if material1 and material2 and material1 != material2:
                penalty = min(penalty, 0.30)  # Труба стальная vs ПЭ = РАЗНЫЕ

            # НОВОЕ: Проверка НАЗНАЧЕНИЯ (ХВС vs ГВС vs Канализация)
            purpose1 = None
            purpose2 = None

            # Приоритет: канализация должна проверяться первой
            # Канализация
            if 'канализ' in name1.lower():
                purpose1 = 'канализация'
            # Холодная вода
            elif 'хвс' in name1.lower() or 'холодн' in name1.lower() or 'для воды' in name1.lower():
                purpose1 = 'хвс'
            # Горячая вода
            elif 'гвс' in name1.lower() or 'горяч' in name1.lower():
                purpose1 = 'гвс'
            # Отопление
            elif 'отоплен' in name1.lower():
                purpose1 = 'отопление'
            # Газ
            elif 'газ' in name1.lower():
                purpose1 = 'газ'

            # То же для name2
            # Канализация
            if 'канализ' in name2.lower():
                purpose2 = 'канализация'
            # Холодная вода
            elif 'хвс' in name2.lower() or 'холодн' in name2.lower() or 'для воды' in name2.lower():
                purpose2 = 'хвс'
            # Горячая вода
            elif 'гвс' in name2.lower() or 'горяч' in name2.lower():
                purpose2 = 'гвс'
            # Отопление
            elif 'отоплен' in name2.lower():
                purpose2 = 'отопление'
            # Газ
            elif 'газ' in name2.lower():
                purpose2 = 'газ'

            # Если назначения определены и РАЗНЫЕ - это РАЗНЫЕ товары!
            if purpose1 and purpose2 and purpose1 != purpose2:
                penalty = min(penalty, 0.30)  # ХВС vs ГВС = РАЗНЫЕ

            # НОВОЕ: Проверка SDR (отношение диаметра к толщине стенки)
            sdr1 = re.search(r'sdr\s*(\d+)', name1.lower())
            sdr2 = re.search(r'sdr\s*(\d+)', name2.lower())

            if sdr1 and sdr2:
                s1 = int(sdr1.group(1))
                s2 = int(sdr2.group(1))

                # Если SDR разный - разная толщина стенки
                if s1 != s2:
                    penalty = min(penalty, 0.30)  # SDR11 vs SDR17 = РАЗНЫЕ

        # 19. ФЛАНЕЦ - проверяем Ду (диаметр) и Ру (давление)
        if 'фланец' in name1.lower() and 'фланец' in name2.lower():
            # Маркировка фланца: 1-150-16 или 1200-25-11-В
            # Формат: [тип]-[Ду]-[Ру]-[исполнение]
            # Где: Ду - условный диаметр (критично!)
            #      Ру - давление (критично!)

            # Ищем паттерн: цифры-цифры-цифры (с поддержкой дробных Ру: 0.3, 1.6)
            pattern1 = re.search(r'(\d+)[-](\d+)[-](\d+(?:[.,]\d+)?)', name1.lower())
            pattern2 = re.search(r'(\d+)[-](\d+)[-](\d+(?:[.,]\d+)?)', name2.lower())

            if pattern1 and pattern2:
                # Извлекаем параметры
                # ВНИМАНИЕ: Два разных формата!
                # Формат 1: [тип]-[Ду]-[Ру] (1-800-0.3, 2-800-0.6)
                # Формат 2: [Ду]-[Ру]-[тип] (150-16-01, 350-16-01)
                # Различаем по первому числу: малое (<20) → тип, большое (>=20) → Ду

                first1 = int(pattern1.group(1))
                if first1 < 20:  # Формат: тип-Ду-Ру
                    type1 = first1
                    du1 = int(pattern1.group(2))
                    ru1 = float(pattern1.group(3).replace(',', '.'))
                else:  # Формат: Ду-Ру-тип
                    du1 = first1
                    ru1 = float(pattern1.group(2).replace(',', '.'))
                    type1 = int(pattern1.group(3).replace(',', '.'))

                first2 = int(pattern2.group(1))
                if first2 < 20:  # Формат: тип-Ду-Ру
                    type2 = first2
                    du2 = int(pattern2.group(2))
                    ru2 = float(pattern2.group(3).replace(',', '.'))
                else:  # Формат: Ду-Ру-тип
                    du2 = first2
                    ru2 = float(pattern2.group(2).replace(',', '.'))
                    type2 = int(pattern2.group(3).replace(',', '.'))

                # Проверка типа исполнения
                # Малая разница (1 vs 2, 100 vs 200) - допустима
                # Большая разница (100 vs 1200, 1 vs 1200) - РАЗНЫЕ товары
                type_diff = abs(type1 - type2)
                if type_diff > 100:  # Большая разница в типе
                    penalty = min(penalty, 0.30)  # РАЗНЫЕ фланцы
                elif type1 != type2:  # Малая разница
                    penalty = min(penalty, 0.85)  # Похожи, но не идентичны

                # Проверка Ду (диаметра) - КРИТИЧНО!
                if abs(du1 - du2) > 10:  # Разница больше 10мм
                    penalty = min(penalty, 0.30)  # Фланец 150 vs 65 = РАЗНЫЕ

                # Проверка Ру (давления) - КРИТИЧНО!
                # Адаптивный порог: дробные (<5) vs целые (>=5)
                max_ru = max(ru1, ru2)
                if max_ru < 5:
                    # Дробные значения: 0.3, 1.6, 2.5 и т.д.
                    if abs(ru1 - ru2) >= 1:  # Разница >= 1 МПа
                        penalty = min(penalty, 0.40)  # Ру 0.3 vs 1.6 = РАЗНЫЕ
                else:
                    # Целые значения: 6, 10, 16, 25 и т.д.
                    if abs(ru1 - ru2) > 5:  # Разница > 5 МПа
                        penalty = min(penalty, 0.40)  # Ру 16 vs 6 = РАЗНЫЕ

            # Альтернативный паттерн для фланцев типа "Фланец 65-16"
            if not pattern1:
                pattern1 = re.search(r'(\d+)[-](\d+)', name1.lower())
            if not pattern2:
                pattern2 = re.search(r'(\d+)[-](\d+)', name2.lower())

            if pattern1 and pattern2 and not (pattern1.groups() == 3):
                du1 = int(pattern1.group(1))  # Первое число - Ду
                ru1 = int(pattern1.group(2))  # Второе число - Ру

                du2 = int(pattern2.group(1))
                ru2 = int(pattern2.group(2))

                # Проверка Ду
                if abs(du1 - du2) > 10:
                    penalty = min(penalty, 0.30)

                # Проверка Ру
                if abs(ru1 - ru2) > 5:
                    penalty = min(penalty, 0.40)

            # Европейский формат DN (DIN/EN): "DN25 16" или "DN 25 16"
            # Формат: DN[Ду] [Ру]
            dn1_pattern = re.search(r'dn\s*(\d+)\s+(\d+(?:[.,]\d+)?)', name1.lower())
            dn2_pattern = re.search(r'dn\s*(\d+)\s+(\d+(?:[.,]\d+)?)', name2.lower())

            if dn1_pattern and dn2_pattern:
                du1 = int(dn1_pattern.group(1))  # DN - условный диаметр
                ru1 = float(dn1_pattern.group(2).replace(',', '.'))  # Давление (может быть дробным)

                du2 = int(dn2_pattern.group(1))
                ru2 = float(dn2_pattern.group(2).replace(',', '.'))

                # Проверка Ду
                if abs(du1 - du2) > 10:
                    penalty = min(penalty, 0.30)  # DN25 vs DN800 = РАЗНЫЕ

                # Проверка Ру
                if abs(ru1 - ru2) > 5:
                    penalty = min(penalty, 0.40)

            # Если один формат DN, другой обычный - проверяем Ду
            if dn1_pattern and pattern2:
                du1 = int(dn1_pattern.group(1))  # DN25 → 25

                # Определяем формат pattern2
                first2 = int(pattern2.group(1))
                if first2 < 20:  # Формат: тип-Ду-Ру
                    du2 = int(pattern2.group(2))
                else:  # Формат: Ду-Ру-тип
                    du2 = first2

                if abs(du1 - du2) > 10:
                    penalty = min(penalty, 0.30)  # DN25 vs Ду150 = РАЗНЫЕ

            if dn2_pattern and pattern1:
                du2 = int(dn2_pattern.group(1))  # DN25 → 25

                # Определяем формат pattern1
                first1 = int(pattern1.group(1))
                if first1 < 20:  # Формат: тип-Ду-Ру
                    du1 = int(pattern1.group(2))
                else:  # Формат: Ду-Ру-тип
                    du1 = first1

                if abs(du1 - du2) > 10:
                    penalty = min(penalty, 0.30)  # Ду150 vs DN25 = РАЗНЫЕ

            # Проверка фланцев ASME (американский стандарт)
            # Форматы: "WN 2\" CL150", "SO 4\" Sch 30", "SW 2\"", "глухой BL 4\""

            # Проверка ТИПА фланца (WN, SO, SW, BL)
            type1 = None
            type2 = None

            # Типы фланцев
            if ' wn ' in name1.lower() or name1.lower().startswith('wn '):
                type1 = 'WN'  # Weld Neck
            elif ' so ' in name1.lower() or name1.lower().startswith('so '):
                type1 = 'SO'  # Slip-On
            elif ' sw ' in name1.lower() or name1.lower().startswith('sw '):
                type1 = 'SW'  # Socket Weld
            elif ' bl ' in name1.lower() or 'глухой' in name1.lower():
                type1 = 'BL'  # Blind
            elif 'loose' in name1.lower():
                type1 = 'LOOSE'  # Loose/Lap Joint

            if ' wn ' in name2.lower() or name2.lower().startswith('wn '):
                type2 = 'WN'
            elif ' so ' in name2.lower() or name2.lower().startswith('so '):
                type2 = 'SO'
            elif ' sw ' in name2.lower() or name2.lower().startswith('sw '):
                type2 = 'SW'
            elif ' bl ' in name2.lower() or 'глухой' in name2.lower():
                type2 = 'BL'
            elif 'loose' in name2.lower():
                type2 = 'LOOSE'

            # Если типы разные - РАЗНЫЕ фланцы
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.30)  # WN vs SO = РАЗНЫЕ

            # Проверка РАЗМЕРА в дюймах (2", 4", 12" и т.д.)
            size1 = None
            size2 = None

            # Паттерн: число с дюймами "2\"", "4\"", "12\""
            s1 = re.search(r'(\d+(?:[.,]\d+)?)\s*["\']', name1.lower())
            s2 = re.search(r'(\d+(?:[.,]\d+)?)\s*["\']', name2.lower())

            if s1:
                size1 = float(s1.group(1).replace(',', '.'))
            if s2:
                size2 = float(s2.group(1).replace(',', '.'))

            # Если размеры разные - РАЗНЫЕ фланцы
            if size1 and size2:
                if abs(size1 - size2) >= 1:  # Разница >= 1 дюйм
                    penalty = min(penalty, 0.30)  # 2" vs 4" = РАЗНЫЕ

            # Проверка Schedule (толщины) - Sch 30, Sch 40, Sch60, Sch80S и т.д.
            sch1 = None
            sch2 = None

            # Паттерн: "Sch 30", "Sch30", "Sch60", "Sch80S"
            sc1 = re.search(r'sch\s*(\d+[a-z]*)', name1.lower())
            sc2 = re.search(r'sch\s*(\d+[a-z]*)', name2.lower())

            if sc1:
                sch1 = sc1.group(1).upper()  # "30", "40", "60", "80S"
            if sc2:
                sch2 = sc2.group(1).upper()

            # Если Schedule разные - РАЗНЫЕ фланцы (влияет на толщину!)
            if sch1 and sch2 and sch1 != sch2:
                # Извлекаем числа для сравнения
                num1 = int(re.search(r'\d+', sch1).group())
                num2 = int(re.search(r'\d+', sch2).group())
                if abs(num1 - num2) >= 10:  # Разница >= 10
                    penalty = min(penalty, 0.30)  # Sch 30 vs Sch 40 = РАЗНЫЕ

            # Проверка МАРКИ СТАЛИ для фланцев (КРИТИЧНО для цены!)
            # Форматы:
            # ГОСТ: "50-25-1-В-03Х18Н11", "300-16-1-В-Ст20", "100-16-1-В-09Г2С"
            # ASME: марка может быть в конце или в середине
            steel1 = None
            steel2 = None

            # Ищем марки стали
            # "03Х18Н11", "12Х18Н10Т" (нержавейки)
            st1 = re.search(r'(\d{2}х\d+н\d+[а-яА-Я]*)', name1.lower())
            if st1:
                steel1 = st1.group(1).upper()

            # "09Г2С", "09Г2", "10Г2С1" (низколегированные)
            if not steel1:
                st1 = re.search(r'(\d{2}г\d[а-яА-Я\d]*)', name1.lower())
                if st1:
                    steel1 = st1.group(1).upper()

            # "Ст20", "Ст.20", "Ст 20" (углеродистые)
            if not steel1:
                st1 = re.search(r'ст[.\s]*(\d+)', name1.lower())
                if st1:
                    steel1 = f"СТ{st1.group(1)}"

            # То же для name2
            st2 = re.search(r'(\d{2}х\d+н\d+[а-яА-Я]*)', name2.lower())
            if st2:
                steel2 = st2.group(1).upper()

            if not steel2:
                st2 = re.search(r'(\d{2}г\d[а-яА-Я\d]*)', name2.lower())
                if st2:
                    steel2 = st2.group(1).upper()

            if not steel2:
                st2 = re.search(r'ст[.\s]*(\d+)', name2.lower())
                if st2:
                    steel2 = f"СТ{st2.group(1)}"

            # Если марки стали разные - РАЗНЫЕ фланцы (КРИТИЧНО для цены!)
            if steel1 and steel2 and steel1 != steel2:
                penalty = min(penalty, 0.30)  # 03Х18Н11 vs 09Г2С = РАЗНЫЕ

        # 19a. ЗАГЛУШКА - проверяем Ду (условный диаметр), тип, размеры
        if 'заглушка' in name1.lower() and 'заглушка' in name2.lower():
            # Форматы:
            # 1. "Заглушка 1-1200-25" (формат фланца: тип-Ду-Ру)
            # 2. "Заглушка Ду 15", "Заглушка Ду25"
            # 3. "Заглушка 159х4", "Заглушка 100х4" (диаметр×толщина)
            # 4. "Заглушка эллиптическая 159х4" (тип + размеры)

            # ПРИОРИТЕТ 1: Формат фланца "1-1200-25" (тип-Ду-Ру)
            pattern1 = re.search(r'(\d+)[-](\d+)[-](\d+(?:[.,]\d+)?)', name1.lower())
            pattern2 = re.search(r'(\d+)[-](\d+)[-](\d+(?:[.,]\d+)?)', name2.lower())

            diameter1 = None
            diameter2 = None
            pressure1 = None
            pressure2 = None

            if pattern1 and pattern2:
                # Формат: тип-Ду-Ру
                # Нам важны Ду (второе число) И Ру (третье число)
                diameter1 = int(pattern1.group(2))  # Ду
                diameter2 = int(pattern2.group(2))  # Ду

                # Извлекаем Ру (давление)
                pressure1 = float(pattern1.group(3).replace(',', '.'))  # Ру (может быть 0.6)
                pressure2 = float(pattern2.group(3).replace(',', '.'))  # Ру

            # ПРИОРИТЕТ 2: Формат "диаметр×толщина" (как труба)
            if not diameter1:
                dim1 = re.search(r'(\d+)\s*[х*×x]\s*(\d+)', name1.lower())
                if dim1:
                    diameter1 = int(dim1.group(1))  # Диаметр

            if not diameter2:
                dim2 = re.search(r'(\d+)\s*[х*×x]\s*(\d+)', name2.lower())
                if dim2:
                    diameter2 = int(dim2.group(1))  # Диаметр

            # ПРИОРИТЕТ 3: Формат "Ду XX"
            if not diameter1:
                du1 = re.search(r'ду\s*(\d+)', name1.lower())
                if du1:
                    diameter1 = int(du1.group(1))

            if not diameter2:
                du2 = re.search(r'ду\s*(\d+)', name2.lower())
                if du2:
                    diameter2 = int(du2.group(1))

            # ПРИОРИТЕТ 4: Простое число после "заглушка"
            if not diameter1:
                simple1 = re.search(r'заглушка\s+(\d+)', name1.lower())
                if simple1:
                    diameter1 = int(simple1.group(1))

            if not diameter2:
                simple2 = re.search(r'заглушка\s+(\d+)', name2.lower())
                if simple2:
                    diameter2 = int(simple2.group(1))

            # Проверка диаметра/Ду - КРИТИЧНО!
            if diameter1 and diameter2:
                if diameter1 != diameter2:  # Любая разница = РАЗНЫЕ
                    penalty = min(penalty, 0.30)  # Заглушка 1200 vs 63 = РАЗНЫЕ

            # Проверка Ру (давления) - КРИТИЧНО для формата фланца!
            if pressure1 is not None and pressure2 is not None:
                if abs(pressure1 - pressure2) > 5:  # Разница больше 5 (как у фланцев)
                    penalty = min(penalty, 0.40)  # Заглушка Ру25 vs Ру16 = РАЗНЫЕ

            # Проверка ТИПА заглушки
            type1 = None
            type2 = None

            if 'эллиптическ' in name1.lower():
                type1 = 'эллиптическая'
            if 'эллиптическ' in name2.lower():
                type2 = 'эллиптическая'

            # Если типы разные - РАЗНЫЕ заглушки
            if type1 != type2 and (type1 or type2):
                penalty = min(penalty, 0.40)  # Эллиптическая vs обычная = РАЗНЫЕ

        # 19b. ОТВОД - проверяем угол, диаметр, толщину, тип
        if 'отвод' in name1.lower() and 'отвод' in name2.lower():
            # Форматы:
            # 1. "Отвод 90 76×5" (угол + диаметр×толщина)
            # 2. "Отвод 90-1-60.3×4" (угол-тип-диаметр×толщина)
            # 3. "Отвод крутоизогнутый Дн 114×5.0" (тип + размеры)
            # 4. "Отвод П 90 114×5-20" (тип П + угол + размеры)
            # 5. "Отвод 90° 57×4" (угол с символом градуса)

            # Извлекаем УГОЛ
            angle1 = re.search(r'(?:отвод|п)\s*(\d+)°?', name1.lower())
            angle2 = re.search(r'(?:отвод|п)\s*(\d+)°?', name2.lower())

            if angle1 and angle2:
                a1 = int(angle1.group(1))
                a2 = int(angle2.group(1))

                # Проверка угла (90° vs 45° = РАЗНЫЕ)
                if a1 != a2:
                    penalty = min(penalty, 0.30)  # Отвод 90° vs 45° = РАЗНЫЕ

            # Извлекаем ДИАМЕТР и ТОЛЩИНУ
            # Паттерн: "76×5", "60.3×4", "114×5.0"
            dims1 = re.search(r'(\d+(?:[.,]\d+)?)\s*[х*×x]\s*(\d+(?:[.,]\d+)?)', name1.lower())
            dims2 = re.search(r'(\d+(?:[.,]\d+)?)\s*[х*×x]\s*(\d+(?:[.,]\d+)?)', name2.lower())

            if dims1 and dims2:
                d1 = float(dims1.group(1).replace(',', '.'))
                t1 = float(dims1.group(2).replace(',', '.'))

                d2 = float(dims2.group(1).replace(',', '.'))
                t2 = float(dims2.group(2).replace(',', '.'))

                # Проверка диаметра (КРИТИЧНО!)
                # Чем меньше диаметр, тем строже проверка:
                # - Очень малые (<50мм): >=1.5мм критично
                # - Малые (50-100мм): >=3мм критично
                # - Большие (>=100мм): >=5мм критично
                max_d = max(d1, d2)

                if max_d < 50:
                    # Очень малые отводы: 21.3 vs 25, 26.9 vs 25
                    if abs(d1 - d2) >= 1.5:  # Даже 1.9мм = РАЗНЫЕ
                        penalty = min(penalty, 0.30)
                elif max_d < 100:
                    # Малые отводы: 57 vs 60.3, 76 vs 80
                    if abs(d1 - d2) >= 3:  # Разница 3мм и более = РАЗНЫЕ
                        penalty = min(penalty, 0.30)
                else:
                    # Большие отводы: 159 vs 219 и т.д.
                    if abs(d1 - d2) >= 5:  # Разница 5мм и более = РАЗНЫЕ
                        penalty = min(penalty, 0.30)

                # Проверка толщины стенки
                if abs(t1 - t2) >= 1:  # Толщина 5 vs 4 = РАЗНЫЕ
                    penalty = min(penalty, 0.30)

            # Проверка ТИПА отвода
            type1 = None
            type2 = None

            if 'крутоизогнут' in name1.lower():
                type1 = 'крутоизогнутый'
            elif 'п 90' in name1.lower() or 'п90' in name1.lower():
                type1 = 'поворотный'

            if 'крутоизогнут' in name2.lower():
                type2 = 'крутоизогнутый'
            elif 'п 90' in name2.lower() or 'п90' in name2.lower():
                type2 = 'поворотный'

            # Если типы разные - РАЗНЫЕ отводы
            if type1 != type2 and (type1 or type2):
                penalty = min(penalty, 0.40)  # Крутоизогнутый vs поворотный = РАЗНЫЕ

        # 20. КРУГ - проверяем диаметр
        if 'круг' in name1.lower() and 'круг' in name2.lower():
            # Ищем диаметр: "Круг 18", "Круг 10 ГОСТ", "Круг 18х5000ММ"
            d1 = re.search(r'круг\s+(\d+)', name1.lower())
            d2 = re.search(r'круг\s+(\d+)', name2.lower())

            if d1 and d2:
                diameter1 = int(d1.group(1))
                diameter2 = int(d2.group(1))

                # Если диаметр отличается на 2мм и больше - РАЗНЫЕ
                if abs(diameter1 - diameter2) >= 2:  # Круг 18 vs 16 = РАЗНЫЕ
                    penalty = min(penalty, 0.30)

        # 21. ПРОКЛАДКА - проверяем тип, размер, давление (Ру)
        if 'прокладка' in name1.lower() and 'прокладка' in name2.lower():
            # Форматы:
            # 1. "А-800-16" (тип-размер-Ру)
            # 2. "1-800-0.3" (тип-размер-Ру, тип=цифра)
            # 3. "СНП-Г-1-1-80-16" (СНП-тип-x-x-размер-Ру)
            # 4. "Б-100-16-А" (тип-размер-Ру-модификация)
            # 5. "65-10 ПОН" (размер-Ру + материал)

            size1 = None
            size2 = None
            pressure1 = None
            pressure2 = None
            type1 = None
            type2 = None

            # ФОРМАТ СНП: "СНП-Г-1-1-80-16" или "СНП-Д-1-1-50-16"
            # Структура: СНП-[тип]-x-x-[размер]-[Ру]
            snp1 = re.search(r'снп[-]([а-яА-Я])[-]\d+[-]\d+[-](\d+)[-](\d+)', name1.lower())
            snp2 = re.search(r'снп[-]([а-яА-Я])[-]\d+[-]\d+[-](\d+)[-](\d+)', name2.lower())

            if snp1:
                type1 = snp1.group(1).upper()  # Г, Д и т.д.
                size1 = int(snp1.group(2))     # Размер (80, 50 и т.д.)
                pressure1 = int(snp1.group(3)) # Ру (16 и т.д.)

            if snp2:
                type2 = snp2.group(1).upper()
                size2 = int(snp2.group(2))
                pressure2 = int(snp2.group(3))

            # ФОРМАТ СТАНДАРТНЫЙ: "А-800-16", "Б-100-16-А", "1-800-0.3"
            # Если НЕ СНП, ищем стандартный формат
            if not size1:
                # Паттерн: [буква/цифра]-[размер]-[Ру]
                pattern1 = re.search(r'([а-яА-Я\d]+)[-](\d+)[-](\d+(?:[.,]\d+)?)', name1.lower())
                if pattern1:
                    type1 = pattern1.group(1).upper()  # А, Б, 1 и т.д.
                    size1 = int(pattern1.group(2))      # Размер (800, 100 и т.д.)
                    pressure1 = float(pattern1.group(3).replace(',', '.'))  # Ру (16, 0.3 и т.д.)

            if not size2:
                pattern2 = re.search(r'([а-яА-Я\d]+)[-](\d+)[-](\d+(?:[.,]\d+)?)', name2.lower())
                if pattern2:
                    type2 = pattern2.group(1).upper()
                    size2 = int(pattern2.group(2))
                    pressure2 = float(pattern2.group(3).replace(',', '.'))

            # ФОРМАТ ПРОСТОЙ: "65-10 ПОН" (размер-Ру)
            # Если все еще не нашли, ищем просто два числа
            if not size1:
                simple1 = re.search(r'(\d+)[-](\d+)', name1.lower())
                if simple1:
                    size1 = int(simple1.group(1))      # Размер
                    pressure1 = int(simple1.group(2))  # Ру

            if not size2:
                simple2 = re.search(r'(\d+)[-](\d+)', name2.lower())
                if simple2:
                    size2 = int(simple2.group(1))
                    pressure2 = int(simple2.group(2))

            # Проверка ТИПА (А, Б, Г, Д и т.д.)
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.40)  # Прокладка Б vs А = РАЗНЫЕ

            # Проверка РАЗМЕРА (КРИТИЧНО!)
            # Адаптивные пороги в зависимости от размера:
            # - Малые (<=100мм): >=10мм критично (15 vs 25, 50 vs 25, 100 vs 80)
            # - Большие (>100мм): >=30мм критично (800 vs 100)
            if size1 and size2:
                max_size = max(size1, size2)

                if max_size <= 100:
                    # Малые прокладки: СНП (15, 25, 50, 80, 100)
                    if abs(size1 - size2) >= 10:  # Даже 10мм = РАЗНЫЕ
                        penalty = min(penalty, 0.30)
                else:
                    # Большие прокладки: А-800, Б-200 и т.д.
                    if abs(size1 - size2) >= 30:  # 30мм и более = РАЗНЫЕ
                        penalty = min(penalty, 0.30)

            # Проверка ДАВЛЕНИЯ Ру (КРИТИЧНО!)
            if pressure1 is not None and pressure2 is not None:
                if abs(pressure1 - pressure2) > 5:  # Разница > 5 (16 vs 0.3)
                    penalty = min(penalty, 0.40)  # Ру 16 vs Ру 0.3 = РАЗНЫЕ

        # 21a. ОПОРА - проверяем тип, размер, модель
        if 'опора' in name1.lower() and 'опора' in name2.lower():
            # Форматы:
            # 1. "Опора хомутовая UG500 114-ХБ-А"
            # 2. "Опора трубная UG-2 57ММ"
            # 3. "Опора скользящая ф57×4,0-09Г2с"
            # 4. "Опора неподвижная хомутовая"
            # 5. "Опора 45-ХБ-А" (без типа)

            # Проверка ТИПА опоры
            type1 = None
            type2 = None

            if 'хомутов' in name1.lower():
                type1 = 'хомутовая'
            elif 'трубн' in name1.lower():
                type1 = 'трубная'
            elif 'скользя' in name1.lower():
                type1 = 'скользящая'
            elif 'неподвижн' in name1.lower():
                type1 = 'неподвижная'

            if 'хомутов' in name2.lower():
                type2 = 'хомутовая'
            elif 'трубн' in name2.lower():
                type2 = 'трубная'
            elif 'скользя' in name2.lower():
                type2 = 'скользящая'
            elif 'неподвижн' in name2.lower():
                type2 = 'неподвижная'

            # Если типы разные - РАЗНЫЕ опоры
            if type1 != type2 and (type1 or type2):
                penalty = min(penalty, 0.40)  # Хомутовая vs скользящая = РАЗНЫЕ

            # Проверка РАЗМЕРА (диаметра)
            # Паттерны: "114-ХБ-А", "57ММ", "ф57×4,0"
            size1 = None
            size2 = None

            # Формат: "114-ХБ-А", "89-ХБ-А"
            s1 = re.search(r'(\d+)[-]хб', name1.lower())
            s2 = re.search(r'(\d+)[-]хб', name2.lower())

            if s1:
                size1 = int(s1.group(1))
            if s2:
                size2 = int(s2.group(1))

            # Формат: "57ММ", "89ММ", "108ММ"
            if not size1:
                s1 = re.search(r'(\d+)мм', name1.lower())
                if s1:
                    size1 = int(s1.group(1))

            if not size2:
                s2 = re.search(r'(\d+)мм', name2.lower())
                if s2:
                    size2 = int(s2.group(1))

            # Формат: "ф57×4,0", "ф89×4,5"
            if not size1:
                s1 = re.search(r'ф(\d+)[×x]', name1.lower())
                if s1:
                    size1 = int(s1.group(1))

            if not size2:
                s2 = re.search(r'ф(\d+)[×x]', name2.lower())
                if s2:
                    size2 = int(s2.group(1))

            # Проверка размера
            if size1 and size2:
                if abs(size1 - size2) >= 10:  # Разница >= 10мм
                    penalty = min(penalty, 0.30)  # Опора 114 vs 45 = РАЗНЫЕ

            # Проверка МОДЕЛИ (UG500, UG-2, UG-1 и т.д.)
            model1 = None
            model2 = None

            m1 = re.search(r'ug[-]?(\d+)', name1.lower())
            m2 = re.search(r'ug[-]?(\d+)', name2.lower())

            if m1:
                model1 = m1.group(1)  # 500, 2 и т.д.
            if m2:
                model2 = m2.group(1)

            # Если модели разные - могут быть РАЗНЫЕ опоры
            if model1 and model2 and model1 != model2:
                penalty = min(penalty, 0.70)  # UG500 vs UG-2 = похожи, но разные модели

        # 21b. ТРОЙНИК - проверяем основной диаметр, ответвление, тип
        if 'тройник' in name1.lower() and 'тройник' in name2.lower():
            # Форматы:
            # 1. "57×3.5-45×3.5" (основной×толщина - ответвление×толщина)
            # 2. "150×7-25×3.5" (основной×толщина - ответвление×толщина)
            # 3. "57×4" (только основной, без ответвления)
            # 4. "оц.76" (оцинкованный, диаметр)
            # 5. "П 89×6-57×4-20" (переходной)
            # 6. "равнопроходной сталь 20" (тип)
            # 7. "1-48.3×3.6" (формат с кодом)

            # Проверка ТИПА тройника
            type1 = None
            type2 = None

            if 'равнопроходн' in name1.lower():
                type1 = 'равнопроходной'
            elif 'переходн' in name1.lower() or ' п ' in name1.lower():
                type1 = 'переходной'
            elif 'оц' in name1.lower():
                type1 = 'оцинкованный'

            if 'равнопроходн' in name2.lower():
                type2 = 'равнопроходной'
            elif 'переходн' in name2.lower() or ' п ' in name2.lower():
                type2 = 'переходной'
            elif 'оц' in name2.lower():
                type2 = 'оцинкованный'

            # Если типы разные - РАЗНЫЕ тройники
            if type1 != type2 and (type1 or type2):
                penalty = min(penalty, 0.40)  # Равнопроходной vs переходной = РАЗНЫЕ

            # Проверка РАЗМЕРОВ (основной диаметр и ответвление)
            # Паттерн: "57×3.5-45×3.5" или "150×7-25×3.5"
            # Формат: основной×толщина-ответвление×толщина

            main1 = None
            branch1 = None
            main_thick1 = None      # Толщина основной трубы
            branch_thick1 = None    # Толщина ответвления
            main2 = None
            branch2 = None
            main_thick2 = None
            branch_thick2 = None

            # Формат с ответвлением: "57×4-32×4"
            pattern1 = re.search(r'(\d+(?:[.,]\d+)?)[×x](\d+(?:[.,]\d+)?)[-](\d+(?:[.,]\d+)?)[×x](\d+(?:[.,]\d+)?)', name1.lower())
            pattern2 = re.search(r'(\d+(?:[.,]\d+)?)[×x](\d+(?:[.,]\d+)?)[-](\d+(?:[.,]\d+)?)[×x](\d+(?:[.,]\d+)?)', name2.lower())

            if pattern1:
                main1 = float(pattern1.group(1).replace(',', '.'))           # Основной диаметр
                main_thick1 = float(pattern1.group(2).replace(',', '.'))     # Толщина основной
                branch1 = float(pattern1.group(3).replace(',', '.'))         # Диаметр ответвления
                branch_thick1 = float(pattern1.group(4).replace(',', '.'))   # Толщина ответвления

            if pattern2:
                main2 = float(pattern2.group(1).replace(',', '.'))
                main_thick2 = float(pattern2.group(2).replace(',', '.'))
                branch2 = float(pattern2.group(3).replace(',', '.'))
                branch_thick2 = float(pattern2.group(4).replace(',', '.'))

            # Формат без ответвления: "57×4" или "оц.76"
            if not main1:
                # "57×4"
                simple1 = re.search(r'(\d+(?:[.,]\d+)?)[×x](\d+(?:[.,]\d+)?)', name1.lower())
                if simple1:
                    main1 = float(simple1.group(1).replace(',', '.'))
                # "оц.76" или просто "76"
                elif not main1:
                    simple1 = re.search(r'оц[.,]?\s*(\d+)', name1.lower())
                    if simple1:
                        main1 = float(simple1.group(1))
                # "Тройник 80" (одиночное число после слова "тройник")
                elif not main1:
                    alone1 = re.search(r'тройник\s+(\d+)(?:\s|$)', name1.lower())
                    if alone1:
                        main1 = float(alone1.group(1))

            if not main2:
                simple2 = re.search(r'(\d+(?:[.,]\d+)?)[×x](\d+(?:[.,]\d+)?)', name2.lower())
                if simple2:
                    main2 = float(simple2.group(1).replace(',', '.'))
                elif not main2:
                    simple2 = re.search(r'оц[.,]?\s*(\d+)', name2.lower())
                    if simple2:
                        main2 = float(simple2.group(1))
                # "Тройник 80" (одиночное число)
                elif not main2:
                    alone2 = re.search(r'тройник\s+(\d+)(?:\s|$)', name2.lower())
                    if alone2:
                        main2 = float(alone2.group(1))

            # Проверка основного диаметра
            # Если у одного есть размер, а у другого нет - КРИТИЧНО РАЗНЫЕ!
            if (main1 is not None) != (main2 is not None):
                penalty = 0.10  # Тройник 80 vs Тройник = КРИТИЧНО РАЗНЫЕ (не min!)

            # Если у обоих есть размеры - сравниваем их
            if main1 and main2:
                if abs(main1 - main2) >= 10:  # Разница >= 10мм
                    penalty = min(penalty, 0.30)  # Тройник 76 vs 150 = РАЗНЫЕ

            # Проверка ответвления (если есть)
            # Если у одного есть ответвление, а у другого нет - РАЗНЫЕ
            if (branch1 is not None) != (branch2 is not None):
                penalty = min(penalty, 0.40)  # 57×4 vs 57×4-32×4 = РАЗНЫЕ

            # Если у обоих есть ответвления - сравниваем их
            # Адаптивные пороги для ответвлений:
            # - Малые (<50мм): >=5мм критично
            # - Большие (>=50мм): >=10мм критично
            if branch1 is not None and branch2 is not None:
                max_branch = max(branch1, branch2)

                if max_branch < 50:
                    # Малые ответвления: 20 vs 25 и т.д.
                    if abs(branch1 - branch2) >= 5:  # Даже 5мм = РАЗНЫЕ
                        penalty = min(penalty, 0.30)
                else:
                    # Большие ответвления: 45 vs 32 и т.д.
                    if abs(branch1 - branch2) >= 10:  # 10мм и более = РАЗНЫЕ
                        penalty = min(penalty, 0.30)

            # Проверка ТОЛЩИНЫ СТЕНОК (КРИТИЧНО для цены!)
            # Проверяем толщину основной трубы
            if main_thick1 is not None and main_thick2 is not None:
                if abs(main_thick1 - main_thick2) >= 0.5:  # Разница >= 0.5мм
                    penalty = min(penalty, 0.30)  # 3.5 vs 4 = РАЗНЫЕ

            # Проверяем толщину ответвления
            if branch_thick1 is not None and branch_thick2 is not None:
                if abs(branch_thick1 - branch_thick2) >= 0.5:  # Разница >= 0.5мм
                    penalty = min(penalty, 0.30)  # 3.5 vs 4 = РАЗНЫЕ

            # Проверка МАРКИ СТАЛИ (КРИТИЧНО для цены!)
            # Паттерны: "09Г2С", "Ст20", "Ст.20", "20", "09Г2"
            steel1 = None
            steel2 = None

            # Ищем марки стали
            # "09Г2С", "09Г2", "10Г2С1" и т.д.
            s1 = re.search(r'(\d{2}г\d[^\s]*)', name1.lower())
            if s1:
                steel1 = s1.group(1).upper()

            # "Ст20", "Ст.20", "Ст 20"
            if not steel1:
                s1 = re.search(r'ст[.\s]*(\d+)', name1.lower())
                if s1:
                    steel1 = f"СТ{s1.group(1)}"

            s2 = re.search(r'(\d{2}г\d[^\s]*)', name2.lower())
            if s2:
                steel2 = s2.group(1).upper()

            if not steel2:
                s2 = re.search(r'ст[.\s]*(\d+)', name2.lower())
                if s2:
                    steel2 = f"СТ{s2.group(1)}"

            # Если марки стали разные - РАЗНЫЕ тройники (влияет на цену!)
            if steel1 and steel2 and steel1 != steel2:
                penalty = min(penalty, 0.30)  # 09Г2С vs Ст20 = РАЗНЫЕ

        # 22. ТРУБА - дополнительная проверка диаметра для труб БЕЗ размеров вида "Труба ПВХ 25"
        if 'труба' in name1.lower() and 'труба' in name2.lower():
            # Уже есть проверка для труб с х (100х5), но нужна проверка для "Труба ПВХ 25"

            # Ищем одиночный диаметр без х: "Труба ПВХ 25", "Труба ПВХ 100"
            d1_alone = re.search(r'труба\s+\S+\s+(\d+)', name1.lower())
            d2_alone = re.search(r'труба\s+\S+\s+(\d+)', name2.lower())

            # Альтернативный паттерн: просто число после ПВХ
            if not d1_alone:
                d1_alone = re.search(r'пвх\s+(\d+)', name1.lower())
            if not d2_alone:
                d2_alone = re.search(r'пвх\s+(\d+)', name2.lower())

            if d1_alone and d2_alone:
                dia1 = int(d1_alone.group(1))
                dia2 = int(d2_alone.group(1))

                # Если диаметр отличается больше чем на 10мм - РАЗНЫЕ
                if abs(dia1 - dia2) > 10:
                    penalty = min(penalty, 0.30)

        # 23. КРАН - проверяем диаметр
        if 'кран' in name1.lower() and 'кран' in name2.lower():
            # Ищем диаметр: Ø15, Ø20, Д-20, d20, д-20 (русская д!)
            d1 = re.search(r'[øØdдД][-\s]*(\d+)', name1.lower())
            d2 = re.search(r'[øØdдД][-\s]*(\d+)', name2.lower())

            if d1 and d2:
                diameter1 = int(d1.group(1))
                diameter2 = int(d2.group(1))

                # Если диаметр отличается - РАЗНЫЕ
                if abs(diameter1 - diameter2) >= 5:  # Разница >= 5мм
                    penalty = min(penalty, 0.30)

        # 24. УГОЛОК - проверяем размеры и тип
        if 'уголок' in name1.lower() and 'уголок' in name2.lower():
            # Проверка типа: наружный vs внутренний
            if ('наружн' in name1.lower() and 'внутр' in name2.lower()) or \
                    ('внутр' in name1.lower() and 'наружн' in name2.lower()):
                penalty = min(penalty, 0.30)  # Наружный vs внутренний = РАЗНЫЕ

            # Проверка размеров: 63х63х5, 63*6, 50*5
            # Ищем паттерн: числа с разделителями
            dims1 = re.findall(r'\d+', name1.lower())
            dims2 = re.findall(r'\d+', name2.lower())

            if len(dims1) >= 1 and len(dims2) >= 1:
                # Берем первые числа как основные размеры
                # Для уголка важны: ширина и толщина

                # Если есть хотя бы 2 числа в одном из названий, проверяем
                if len(dims1) >= 2 or len(dims2) >= 2:
                    # Последнее число обычно толщина
                    thick1 = int(dims1[-1]) if len(dims1) >= 1 else 0
                    thick2 = int(dims2[-1]) if len(dims2) >= 1 else 0

                    # Первое число - ширина полки
                    width1 = int(dims1[0]) if len(dims1) >= 1 else 0
                    width2 = int(dims2[0]) if len(dims2) >= 1 else 0

                    # Проверка толщины (КРИТИЧНО - даже 1мм разница важна!)
                    if thick1 != thick2 and thick1 > 0 and thick2 > 0:
                        penalty = min(penalty, 0.30)  # Уголок 5мм vs 6мм = РАЗНЫЕ

                    # Проверка ширины
                    if abs(width1 - width2) > 10:  # Разница больше 10мм
                        penalty = min(penalty, 0.30)

        # 25. САЛЬНИК - проверяем размер
        if 'сальник' in name1.lower() and 'сальник' in name2.lower():
            # Ищем число: "Сальник 100", "Сальник 50"
            d1 = re.search(r'сальник\s+(\d+)', name1.lower())
            d2 = re.search(r'сальник\s+(\d+)', name2.lower())

            if d1 and d2:
                size1 = int(d1.group(1))
                size2 = int(d2.group(1))

                # Если размер отличается больше чем на 10 - РАЗНЫЕ
                if abs(size1 - size2) > 10:
                    penalty = min(penalty, 0.30)

        # 26. ПРОФИЛЬ - проверяем размеры (квадратный/прямоугольный)
        if 'профиль' in name1.lower() and 'профиль' in name2.lower():
            # Ищем размеры: 50х50х5, 100х100х4, 200х200х6
            dims1 = re.search(r'(\d+)\s*[хx*×]\s*(\d+)\s*[хx*×]\s*(\d+)', name1.lower())
            dims2 = re.search(r'(\d+)\s*[хx*×]\s*(\d+)\s*[хx*×]\s*(\d+)', name2.lower())

            if dims1 and dims2:
                # Для квадратного профиля: ширина х высота х толщина
                w1 = int(dims1.group(1))
                h1 = int(dims1.group(2))
                t1 = int(dims1.group(3))

                w2 = int(dims2.group(1))
                h2 = int(dims2.group(2))
                t2 = int(dims2.group(3))

                # Проверка размеров
                if abs(w1 - w2) > 10 or abs(h1 - h2) > 10:
                    penalty = min(penalty, 0.30)  # Профиль 50х50 vs 200х200 = РАЗНЫЕ

                # Проверка толщины (КРИТИЧНО - должна совпадать!)
                if t1 != t2:  # Даже 1мм разница критична!
                    penalty = min(penalty, 0.30)  # Профиль 100х100х5 vs 100х100х4 = РАЗНЫЕ

        # 27. РАЗНЫЕ КАТЕГОРИИ - сетка vs шток, лист vs труба и т.д.
        # Если один товар - сетка, а другой - шток, это РАЗНЫЕ товары!
        categories_mismatch = [
            ('сетка', 'шток'),
            ('сетка', 'труба'),
            ('шток', 'труба'),
            ('лист', 'труба'),
            ('полоса', 'труба'),
            ('круг', 'труба'),
        ]

        for cat1, cat2 in categories_mismatch:
            if (cat1 in name1.lower() and cat2 in name2.lower()) or \
                    (cat2 in name1.lower() and cat1 in name2.lower()):
                penalty = min(penalty, 0.30)  # Сетка vs Шток = РАЗНЫЕ

        # 28. СВАРОЧНЫЙ АППАРАТ - проверяем модель и максимальный ток
        if ('аппарат' in name1.lower() and 'аппарат' in name2.lower()) or \
                ('сварочн' in name1.lower() and 'сварочн' in name2.lower()):

            # Извлекаем модели: TIG 300, W229, САИ-315, ARC 250 и т.д.
            # Ищем буквенно-цифровые коды
            models1 = re.findall(r'[A-ZА-Я]{2,}\s*\d+|[A-ZА-Я]\d+', name1.upper())
            models2 = re.findall(r'[A-ZА-Я]{2,}\s*\d+|[A-ZА-Я]\d+', name2.upper())

            # Проверяем совпадение моделей
            if models1 and models2:
                # Если ни одна модель не совпадает - РАЗНЫЕ аппараты
                common_models = set(models1) & set(models2)
                if not common_models:
                    penalty = min(penalty, 0.40)  # TIG 300 vs ARC 250 = РАЗНЫЕ

            # Проверяем максимальный ток: "250А", "200А", "315А"
            current1 = re.search(r'(\d+)а', name1.lower())
            current2 = re.search(r'(\d+)а', name2.lower())

            if current1 and current2:
                # Берем последнее значение тока (обычно это максимум)
                all_currents1 = re.findall(r'(\d+)а', name1.lower())
                all_currents2 = re.findall(r'(\d+)а', name2.lower())

                if all_currents1 and all_currents2:
                    max_current1 = max(int(c) for c in all_currents1)
                    max_current2 = max(int(c) for c in all_currents2)

                    # Если разница в токе больше 30А - РАЗНЫЕ
                    if abs(max_current1 - max_current2) > 30:
                        penalty = min(penalty, 0.35)  # 250А vs 200А может быть допустимо

        # 29. ДНИЩЕ - проверяем диаметр, толщину, высоту
        if 'днище' in name1.lower() and 'днище' in name2.lower():
            # Формат: Днище 1640-10-426, Днище 2400-12-600, Днище 2700-10
            # Где: диаметр-толщина-высота

            # Ищем три числа подряд: диаметр-толщина-высота
            dims1 = re.search(r'(\d+)[-](\d+)[-](\d+)', name1.lower())
            dims2 = re.search(r'(\d+)[-](\d+)[-](\d+)', name2.lower())

            # Или два числа: диаметр-толщина
            if not dims1:
                dims1 = re.search(r'(\d+)[-](\d+)', name1.lower())
            if not dims2:
                dims2 = re.search(r'(\d+)[-](\d+)', name2.lower())

            if dims1 and dims2:
                diameter1 = int(dims1.group(1))
                diameter2 = int(dims2.group(1))

                # Проверка диаметра (КРИТИЧНО!)
                if abs(diameter1 - diameter2) > 200:  # Разница больше 200мм
                    penalty = min(penalty, 0.30)  # Днище 1640 vs 2400 = РАЗНЫЕ

        # 30. МУФТА - проверяем тип и диаметр
        if 'муфта' in name1.lower() and 'муфта' in name2.lower():
            # Определяем тип муфты
            type1 = None
            type2 = None

            # Кабельная
            if 'кабельн' in name1.lower():
                type1 = 'кабельная'
            # Соединительная
            elif 'соединительн' in name1.lower():
                type1 = 'соединительная'
            # Концевая
            elif 'концев' in name1.lower():
                type1 = 'концевая'
            # Переходная
            elif 'перех' in name1.lower():
                type1 = 'переходная'

            if 'кабельн' in name2.lower():
                type2 = 'кабельная'
            elif 'соединительн' in name2.lower():
                type2 = 'соединительная'
            elif 'концев' in name2.lower():
                type2 = 'концевая'
            elif 'перех' in name2.lower():
                type2 = 'переходная'

            # Если типы определены и разные - это РАЗНЫЕ товары!
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.35)  # Кабельная vs соединительная = РАЗНЫЕ

            # Проверяем диаметр: Ø50, DN25, ø32, d50
            d1 = re.search(r'[øØd]\s*(\d+)', name1.lower())
            d2 = re.search(r'[øØd]\s*(\d+)', name2.lower())

            if not d1:
                d1 = re.search(r'dn\s*(\d+)', name1.lower())
            if not d2:
                d2 = re.search(r'dn\s*(\d+)', name2.lower())

            if d1 and d2:
                diameter1 = int(d1.group(1))
                diameter2 = int(d2.group(1))

                # Если диаметр сильно отличается - разные товары
                if abs(diameter1 - diameter2) > 10:  # Разница >10мм
                    ratio = min(diameter1, diameter2) / max(diameter1, diameter2)
                    penalty = min(penalty, 0.4 + ratio * 0.2)  # Муфта Ø50 vs Ø32 = разные

        # 31. КЛАПАНЫ И КРАНЫ - проверяем тип, DN (диаметр) и CL (класс давления)
        # ВАЖНО: Исключаем грузоподъемные краны (опорные, подвесные, козловые, башенные)
        is_valve_or_valve_crane1 = any(word in name1.lower() for word in ['клапан', 'задвижк']) or \
                                   ('кран' in name1.lower() and not any(word in name1.lower() for word in ['опорн', 'подвесн', 'козлов', 'башенн', 'мостов', 'портал']))
        is_valve_or_valve_crane2 = any(word in name2.lower() for word in ['клапан', 'задвижк']) or \
                                   ('кран' in name2.lower() and not any(word in name2.lower() for word in ['опорн', 'подвесн', 'козлов', 'башенн', 'мостов', 'портал']))

        if is_valve_or_valve_crane1 and is_valve_or_valve_crane2:

            # 1. Тип клапана - КРИТИЧНО!
            valve_types = {
                'запорн': 'gate',
                'обратн': 'check',
                'предохран': 'safety',
                'регулир': 'control',
                'дроссель': 'control',  # Дроссель-клапан = регулирующий (та же группа!)
                'шаров': 'ball',
                'задвижк': 'gate_valve',
                'дисков': 'butterfly'  # Дисковый затвор (butterfly valve)
            }

            type1 = None
            type2 = None

            for rus, eng in valve_types.items():
                if rus in name1.lower():
                    type1 = eng
                if rus in name2.lower():
                    type2 = eng

            # Если типы определены и РАЗНЫЕ - это РАЗНЫЕ клапаны!
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.30)  # Запорный vs Обратный = РАЗНЫЕ

            # 2. DN (диаметр) - КРИТИЧНО!
            # Поддержка форматов: DN4", DN 4", DN4, Ду100, DN3/4" (дроби), 15мм, Ø32
            dn1_match = re.search(r'(?:DN|Ду)\s*(\d+(?:/\d+)?|\d+(?:\.\d+)?)\s*["\']?', name1, re.IGNORECASE)
            dn2_match = re.search(r'(?:DN|Ду)\s*(\d+(?:/\d+)?|\d+(?:\.\d+)?)\s*["\']?', name2, re.IGNORECASE)

            # Если не нашли DN/Ду, ищем просто "15мм", "20мм" и т.д.
            if not dn1_match:
                dn1_match = re.search(r'(\d+(?:\.\d+)?)\s*мм', name1, re.IGNORECASE)
            if not dn2_match:
                dn2_match = re.search(r'(\d+(?:\.\d+)?)\s*мм', name2, re.IGNORECASE)

            # Если всё ещё не нашли, ищем формат Ø32 (миллиметры)
            if not dn1_match:
                dn1_match = re.search(r'Ø\s*(\d+(?:\.\d+)?)', name1, re.IGNORECASE)
            if not dn2_match:
                dn2_match = re.search(r'Ø\s*(\d+(?:\.\d+)?)', name2, re.IGNORECASE)

            if dn1_match and dn2_match:
                # Конвертируем дроби в десятичные числа
                dn1_str = dn1_match.group(1)
                dn2_str = dn2_match.group(1)

                # Обработка дробей (3/4 → 0.75, 1/2 → 0.5)
                if '/' in dn1_str:
                    parts = dn1_str.split('/')
                    dn1_value = float(parts[0]) / float(parts[1])
                else:
                    dn1_value = float(dn1_str)

                if '/' in dn2_str:
                    parts = dn2_str.split('/')
                    dn2_value = float(parts[0]) / float(parts[1])
                else:
                    dn2_value = float(dn2_str)

                # Определяем единицы измерения
                # DN с кавычками (" или ") = дюймы
                # DN без кавычек = миллиметры (как Ду)
                # Ду или "мм" = миллиметры
                dn1_has_quotes = ('"' in name1[max(0, dn1_match.start()-5):dn1_match.end()+5] or
                                  "'" in name1[max(0, dn1_match.start()-5):dn1_match.end()+5])
                dn2_has_quotes = ('"' in name2[max(0, dn2_match.start()-5):dn2_match.end()+5] or
                                  "'" in name2[max(0, dn2_match.start()-5):dn2_match.end()+5])

                if 'DN' in dn1_match.group(0).upper():
                    dn1_unit = 'дюйм' if dn1_has_quotes else 'мм'
                else:
                    dn1_unit = 'мм'  # Ду или мм всегда миллиметры

                if 'DN' in dn2_match.group(0).upper():
                    dn2_unit = 'дюйм' if dn2_has_quotes else 'мм'
                else:
                    dn2_unit = 'мм'  # Ду или мм всегда миллиметры

                # Если разные единицы - считаем РАЗНЫМИ (пока без конверсии)
                if dn1_unit != dn2_unit:
                    penalty = min(penalty, 0.30)  # DN4" vs Ду100 = РАЗНЫЕ
                # Если одинаковые единицы - сравниваем значения
                elif abs(dn1_value - dn2_value) > 0.1:  # Допуск 0.1 для округлений
                    penalty = min(penalty, 0.30)  # DN3/4" vs DN3" = РАЗНЫЕ (0.75 vs 3)

            # 3. CL (класс давления) - КРИТИЧНО при большой разнице!
            cl1_match = re.search(r'CL\s*(\d+)', name1, re.IGNORECASE)
            cl2_match = re.search(r'CL\s*(\d+)', name2, re.IGNORECASE)

            if cl1_match and cl2_match:
                cl1 = int(cl1_match.group(1))
                cl2 = int(cl2_match.group(1))

                # Большая разница в классе давления = РАЗНЫЕ клапаны
                cl_diff = abs(cl1 - cl2)
                if cl_diff >= 150:  # Порог для значимой разницы (включая 150)
                    penalty = min(penalty, 0.30)  # CL300 vs CL150 = РАЗНЫЕ

            # 3.5. PN (номинальное давление) - КРИТИЧНО при большой разнице!
            # PN используется для кранов (PN4, PN16, PN63, Ру10)
            pn1_match = re.search(r'(?:PN|Ру)\s*(\d+)', name1, re.IGNORECASE)
            pn2_match = re.search(r'(?:PN|Ру)\s*(\d+)', name2, re.IGNORECASE)

            if pn1_match and pn2_match:
                pn1 = int(pn1_match.group(1))
                pn2 = int(pn2_match.group(1))

                # Большая разница в номинальном давлении = РАЗНЫЕ краны
                pn_diff = abs(pn1 - pn2)
                if pn_diff >= 10:  # Порог для значимой разницы
                    penalty = min(penalty, 0.30)  # PN4 vs PN16 = РАЗНЫЕ (разница 12)

            # 3.6. Конструкция крана - КРИТИЧНО для шаровых кранов!
            # Полнопроходной, Неполнопроходной, Водоразборный
            crane_design_types = {
                'полнопроход': 'full_bore',
                'неполнопроход': 'reduced_bore',
                'водоразбор': 'water_tap'  # Водоразборный со штуцером
            }

            design1 = None
            design2 = None

            for pattern, design_type in crane_design_types.items():
                if pattern in name1.lower():
                    design1 = design_type
                if pattern in name2.lower():
                    design2 = design_type

            # Если оба имеют указание конструкции И они разные
            if design1 and design2 and design1 != design2:
                penalty = min(penalty, 0.30)  # Полнопроходной vs Водоразборный = РАЗНЫЕ

            # 4. Материал ASTM - средняя важность
            # A216-WCB, A351-CF3A, A494-CW6MC, A217-WC9
            # Нормализуем: "ASTM A216-WCB" -> "A216-WCB", "A216-WCB" -> "A216-WCB"
            material1_match = re.search(r'(?:ASTM\s+)?(A\d+(?:-[A-Z0-9]+)?)', name1, re.IGNORECASE)
            material2_match = re.search(r'(?:ASTM\s+)?(A\d+(?:-[A-Z0-9]+)?)', name2, re.IGNORECASE)

            if material1_match and material2_match:
                material1 = material1_match.group(1).upper()
                material2 = material2_match.group(1).upper()

                # Разный материал - средняя важность (не критично, но влияет)
                if material1 != material2:
                    penalty = min(penalty, 0.70)  # A216-WCB vs A351-CF3A = средняя схожесть

        # 32. ТРУБЫ - проверяем DN, материал, толщину стенки, PWHT
        if 'труб' in name1.lower() and 'труб' in name2.lower():

            # 1. DN (диаметр) - КРИТИЧНО!
            # Используем ту же логику, что и для клапанов
            dn1_match = re.search(r'(?:DN|Ду)\s*(\d+(?:/\d+)?|\d+(?:\.\d+)?)\s*["\']?', name1, re.IGNORECASE)
            dn2_match = re.search(r'(?:DN|Ду)\s*(\d+(?:/\d+)?|\d+(?:\.\d+)?)\s*["\']?', name2, re.IGNORECASE)

            if dn1_match and dn2_match:
                # Парсим значения
                dn1_str = dn1_match.group(1)
                dn2_str = dn2_match.group(1)

                # Конвертируем дроби
                if '/' in dn1_str:
                    parts = dn1_str.split('/')
                    dn1_value = float(parts[0]) / float(parts[1])
                else:
                    dn1_value = float(dn1_str)

                if '/' in dn2_str:
                    parts = dn2_str.split('/')
                    dn2_value = float(parts[0]) / float(parts[1])
                else:
                    dn2_value = float(dn2_str)

                # Определяем единицы (DN с кавычками = дюймы, без = мм)
                dn1_has_quotes = ('"' in name1[max(0, dn1_match.start()-5):dn1_match.end()+5] or
                                  "'" in name1[max(0, dn1_match.start()-5):dn1_match.end()+5])
                dn2_has_quotes = ('"' in name2[max(0, dn2_match.start()-5):dn2_match.end()+5] or
                                  "'" in name2[max(0, dn2_match.start()-5):dn2_match.end()+5])

                if 'DN' in dn1_match.group(0).upper():
                    dn1_unit = 'дюйм' if dn1_has_quotes else 'мм'
                else:
                    dn1_unit = 'мм'

                if 'DN' in dn2_match.group(0).upper():
                    dn2_unit = 'дюйм' if dn2_has_quotes else 'мм'
                else:
                    dn2_unit = 'мм'

                # Разные единицы или значения = РАЗНЫЕ
                if dn1_unit != dn2_unit or abs(dn1_value - dn2_value) > 0.1:
                    penalty = min(penalty, 0.30)  # DN4" vs DN24" = РАЗНЫЕ

            # 2. Материал ASTM - КРИТИЧНО для труб!
            # A106 (углеродистая) vs A312 (нержавеющая) = РАЗНЫЕ
            material1_match = re.search(r'ASTM\s+(A\d+)', name1, re.IGNORECASE)
            material2_match = re.search(r'ASTM\s+(A\d+)', name2, re.IGNORECASE)

            if material1_match and material2_match:
                material1 = material1_match.group(1).upper()
                material2 = material2_match.group(1).upper()

                if material1 != material2:
                    penalty = min(penalty, 0.30)  # A106 vs A312 = РАЗНЫЕ

            # 3. Толщина стенки - КРИТИЧНО!
            # Формат: (114.3x6.02) или (609.6x6.35)
            thickness1_match = re.search(r'\([\d.]+x([\d.]+)\)', name1)
            thickness2_match = re.search(r'\([\d.]+x([\d.]+)\)', name2)

            if thickness1_match and thickness2_match:
                thickness1 = float(thickness1_match.group(1))
                thickness2 = float(thickness2_match.group(1))

                # Допуск 0.1 мм
                if abs(thickness1 - thickness2) > 0.1:
                    penalty = min(penalty, 0.30)  # 6.02 vs 6.35 = РАЗНЫЕ

            # 4. PWHT (термообработка) - КРИТИЧНО!
            pwht1 = 'pwht' in name1.lower()
            pwht2 = 'pwht' in name2.lower()

            if pwht1 != pwht2:
                penalty = min(penalty, 0.30)  # С PWHT vs без PWHT = РАЗНЫЕ

        # 33. ТРАВЕРСЫ - проверяем модель (ТМ-1, ТН-4, ТВ-255 и т.д.)
        if 'траверс' in name1.lower() and 'траверс' in name2.lower():

            # Извлекаем модель траверсы
            # Форматы: ТМ-1, ТН-4, ТВ-255, ТМ6, ТП2, ТМ-20, ТМ-19 и т.д.
            model1_match = re.search(r'Т[МНВПР][\-\s]?\d+(?:\-\d+)?', name1, re.IGNORECASE)
            model2_match = re.search(r'Т[МНВПР][\-\s]?\d+(?:\-\d+)?', name2, re.IGNORECASE)

            if model1_match and model2_match:
                # Нормализуем модель: убираем пробелы, приводим к верхнему регистру
                model1 = model1_match.group(0).upper().replace(' ', '')
                model2 = model2_match.group(0).upper().replace(' ', '')

                # Проверяем "с контуром"
                has_contour1 = 'контур' in name1.lower()
                has_contour2 = 'контур' in name2.lower()

                # Если модели ОДИНАКОВЫЕ и контур одинаковый - не применяем штрафы
                # Это позволит базовой схожести работать нормально
                if model1 == model2 and has_contour1 == has_contour2:
                    # Не применяем штраф - пусть базовый алгоритм работает
                    pass
                else:
                    # Разные модели или разный контур = РАЗНЫЕ траверсы
                    penalty = min(penalty, 0.30)
            else:
                # Если модель не найдена - проверяем только контур
                has_contour1 = 'контур' in name1.lower()
                has_contour2 = 'контур' in name2.lower()

                if has_contour1 != has_contour2:
                    penalty = min(penalty, 0.30)

        # 34. РОЗЕТКИ - проверяем ток, тип установки, IP, количество мест
        if 'розетк' in name1 and 'розетк' in name2:

            # 1. Ток (16А vs 32А) - КРИТИЧНО!
            # 16А = бытовые, 32А = промышленные
            current1_match = re.search(r'(\d+)а', name1)  # name1 уже lower
            current2_match = re.search(r'(\d+)а', name2)  # name2 уже lower

            if current1_match and current2_match:
                current1 = int(current1_match.group(1))
                current2 = int(current2_match.group(1))

                # Разный ток = РАЗНЫЕ розетки
                if current1 != current2:
                    penalty = min(penalty, 0.25)  # 16А vs 32А = РАЗНЫЕ (усиленный штраф)

            # 2. Тип установки - КРИТИЧНО!
            # Переносная, накладная, скрытая, кабельная, открытая, наружная
            # ВАЖНО: проверяем в порядке приоритета!
            type_patterns = [
                ('переносн', 'portable'),
                ('кабельн', 'portable'),     # Кабельная = переносная
                ('скрыт', 'recessed'),       # Скрытая - ПРИОРИТЕТ!
                ('встра', 'recessed'),       # Встраиваемая
                ('оп ', 'surface'),          # ОП = открытая пиллинг = накладная
                ('накладн', 'surface'),
                ('наружн', 'surface'),       # Наружная = накладная
                ('открыт', 'surface'),       # Открытая = накладная
                ('штепсельн', 'surface'),    # Штепсельная открытой = накладная
            ]

            type1 = None
            type2 = None

            # Проверяем в порядке приоритета (первое совпадение)
            for pattern, install_type in type_patterns:
                if not type1 and pattern in name1:  # name1 уже lower
                    type1 = install_type
                if not type2 and pattern in name2:  # name2 уже lower
                    type2 = install_type

            # ВАЖНО: Если типы определены и разные
            if type1 and type2 and type1 != type2:
                penalty = min(penalty, 0.25)  # Разные типы = РАЗНЫЕ (усиленный штраф)

            # КРИТИЧНО: Если у одной розетки тип определен, а у другой НЕТ
            # Значит они вероятно РАЗНЫЕ
            elif type1 and not type2:
                penalty = min(penalty, 0.40)  # Тип определен vs неизвестный = вероятно РАЗНЫЕ
            elif type2 and not type1:
                penalty = min(penalty, 0.40)  # Тип определен vs неизвестный = вероятно РАЗНЫЕ

            # 3. IP (степень защиты) - КРИТИЧНО!
            # IP20 vs IP44/IP54
            ip1_match = re.search(r'IP\s*(\d+)', name1, re.IGNORECASE)
            ip2_match = re.search(r'IP\s*(\d+)', name2, re.IGNORECASE)

            if ip1_match and ip2_match:
                ip1 = int(ip1_match.group(1))
                ip2 = int(ip2_match.group(1))

                # IP20 vs IP44/IP54 = РАЗНЫЕ (сухие vs влагозащита)
                # IP44 vs IP54 = похожие (оба влагозащита)

                # Группы IP:
                # IP20-23: сухие помещения
                # IP44-54: влагозащита
                # IP65+: полная защита

                ip1_group = 'dry' if ip1 < 40 else ('wet' if ip1 < 60 else 'full')
                ip2_group = 'dry' if ip2 < 40 else ('wet' if ip2 < 60 else 'full')

                if ip1_group != ip2_group:
                    penalty = min(penalty, 0.30)  # IP20 vs IP44 = РАЗНЫЕ
                # Если в одной группе (IP44 vs IP54) - не применяем штраф

            # 4. Количество мест/розеток - КРИТИЧНО!
            # Парсим: "2 места", "двойная", "2-х местная", "тройная", "1 место", просто цифра в конце

            # Извлекаем количество мест для name1
            places1 = None

            # Варианты: "2 места", "2 мест", "2места"
            match1 = re.search(r'(\d+)[\s\-]?мест', name1)
            if match1:
                places1 = int(match1.group(1))
            # Варианты: "двойная", "тройная", "одинарная"
            elif 'двойн' in name1:
                places1 = 2
            elif 'тройн' in name1:
                places1 = 3
            elif 'одинарн' in name1 or 'одноместн' in name1:
                places1 = 1
            # Варианты: "2-х местная", "3-х местная"
            elif re.search(r'(\d+)[\s\-]?х\s*местн', name1):
                match1 = re.search(r'(\d+)[\s\-]?х\s*местн', name1)
                places1 = int(match1.group(1))
            # Варианты: просто цифра в конце (но НЕ напряжение, НЕ ток)
            # "Розетка 220В 16А 2" → 2 места
            # "Розетка 16А 220В 1" → 1 место (любой порядок!)
            elif re.search(r'(?:(?:220|230|250|380)в|(?:16|32)а).*?(?:(?:220|230|250|380)в|(?:16|32)а).*?(\d+)\s*$', name1):
                match1 = re.search(r'(?:(?:220|230|250|380)в|(?:16|32)а).*?(?:(?:220|230|250|380)в|(?:16|32)а).*?(\d+)\s*$', name1)
                places1 = int(match1.group(1))

            # Извлекаем количество мест для name2
            places2 = None

            match2 = re.search(r'(\d+)[\s\-]?мест', name2)
            if match2:
                places2 = int(match2.group(1))
            elif 'двойн' in name2:
                places2 = 2
            elif 'тройн' in name2:
                places2 = 3
            elif 'одинарн' in name2 or 'одноместн' in name2:
                places2 = 1
            elif re.search(r'(\d+)[\s\-]?х\s*местн', name2):
                match2 = re.search(r'(\d+)[\s\-]?х\s*местн', name2)
                places2 = int(match2.group(1))
            elif re.search(r'(?:(?:220|230|250|380)в|(?:16|32)а).*?(?:(?:220|230|250|380)в|(?:16|32)а).*?(\d+)\s*$', name2):
                match2 = re.search(r'(?:(?:220|230|250|380)в|(?:16|32)а).*?(?:(?:220|230|250|380)в|(?:16|32)а).*?(\d+)\s*$', name2)
                places2 = int(match2.group(1))

            # Сравниваем количество мест
            if places1 and places2:
                if places1 != places2:
                    # Разное количество = КРИТИЧНО
                    penalty = min(penalty, 0.30)  # 1 место vs 2 места = РАЗНЫЕ
            elif places1 and not places2:
                # У одной указано, у другой нет = вероятно разные
                penalty = min(penalty, 0.40)
            elif places2 and not places1:
                # У одной указано, у другой нет = вероятно разные
                penalty = min(penalty, 0.40)

        return penalty

    @staticmethod
    def _normalize_string(s: str) -> str:
        """Нормализует строку для сравнения"""
        s = s.lower()
        # Убираем лишние пробелы
        s = ' '.join(s.split())
        # Убираем точки и запятые
        s = s.replace('.', '').replace(',', '.')
        return s

    @staticmethod
    def _compare_dimensions(features1: Dict, features2: Dict) -> float:
        """
        Сравнивает размеры с допуском

        Returns:
            float: схожесть размеров (0-1)
        """
        dimensions = ['diameter', 'thickness', 'length', 'width', 'height', 'number']

        matches = 0
        total = 0

        for dim in dimensions:
            val1 = features1.get(dim)
            val2 = features2.get(dim)

            if val1 is not None and val2 is not None:
                total += 1
                # Проверяем с допуском 5%
                tolerance = 0.05
                diff = abs(val1 - val2)
                avg = (val1 + val2) / 2

                if avg > 0 and diff / avg <= tolerance:
                    matches += 1
                elif diff <= 1:  # Абсолютный допуск 1мм
                    matches += 1

        if total == 0:
            return 0.7  # Нейтрально, если размеры не указаны

        return matches / total

    @staticmethod
    def find_matching_products(
            target_product: str,
            product_list: List[str],
            threshold: float = 0.85
    ) -> List[Tuple[str, float]]:
        """
        Находит схожие товары в списке

        Args:
            target_product: целевой товар
            product_list: список товаров для поиска
            threshold: минимальный порог схожести

        Returns:
            list: список кортежей (товар, схожесть), отсортированный по убыванию схожести
        """
        matches = []

        for product in product_list:
            if product == target_product:
                continue

            similarity = ProductMatcher.calculate_similarity(target_product, product)

            if similarity >= threshold:
                matches.append((product, similarity))

        # Сортируем по убыванию схожести
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    @staticmethod
    def compare_products(product1: str, product2: str) -> Dict[str, Any]:
        """
        Детальное сравнение двух товаров

        Args:
            product1: первый товар
            product2: второй товар

        Returns:
            dict: детальная информация о сравнении
        """
        features1 = ProductMatcher.extract_key_features(product1)
        features2 = ProductMatcher.extract_key_features(product2)

        similarity = ProductMatcher.calculate_similarity(product1, product2)

        # Находим совпадения
        matches = {}
        differences = {}

        for key in ['type', 'category', 'diameter', 'thickness', 'length',
                    'width', 'height', 'material', 'gost', 'model', 'brand']:
            val1 = features1.get(key)
            val2 = features2.get(key)

            if val1 is not None and val2 is not None:
                if val1 == val2:
                    matches[key] = val1
                else:
                    differences[key] = {
                        'product1': val1,
                        'product2': val2
                    }

        return {
            'similarity': similarity,
            'product1': product1,
            'product2': product2,
            'features1': features1,
            'features2': features2,
            'matches': matches,
            'differences': differences
        }


# Демонстрация улучшений
if __name__ == "__main__":
    # Примеры проблемных товаров из вашего файла
    test_cases = [
        ("Муфта ППР 75", "Муфта американка ППР 32ММ ВР"),
        ("Круг зачистной 180х6.0х22", "Круг 16 Ст20 ГОСТ 1050-2013"),
        ("Кран ППР 63", "Кран шаровой Ду80"),
        ("Шайба 16 ГОСТ 11371-78", "Шайба М12 ISO 7090"),
        ("Строп текстильный СТП 10,0(6 000)", "Строп канатный 4СК 10,0 / 4000"),
    ]

    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОГО АЛГОРИТМА:")
    print("=" * 80)

    for prod1, prod2 in test_cases:
        similarity = ProductMatcher.calculate_similarity(prod1, prod2)
        f1 = ProductMatcher.extract_key_features(prod1)
        f2 = ProductMatcher.extract_key_features(prod2)

        print(f"\n1: {prod1}")
        print(f"   Категория: {f1['category']}, Тип: {f1['type']}")
        print(f"2: {prod2}")
        print(f"   Категория: {f2['category']}, Тип: {f2['type']}")
        print(f"Схожесть: {similarity:.1%}")
        print("-" * 80)