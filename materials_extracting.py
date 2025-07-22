# Отбор Труб

import pandas as pd
import re

def extract_dimensions(text):
    text = str(text).lower() # Приводим к нижнему регистру для регистронезависимого поиска

    # Паттерн для поиска размеров вида ЧИСЛОхЧИСЛО
    match_diam_thick = re.search(r'(\d+(?:,\d+)?)[хx](\d+(?:,\d+)?)', text)
    if match_diam_thick:
        return f"диаметр {match_diam_thick.group(1)}, толщина {match_diam_thick.group(2)}"

    # Паттерн для поиска размеров вида ЧИСЛОхЧИСЛОхЧИСЛОмм
    match_multi_mm = re.search(r'(\d+)[хx](\d+)[хx](\d+)мм', text)
    if match_multi_mm:
        return f"{match_multi_mm.group(1)}x{match_multi_mm.group(2)}x{match_multi_mm.group(3)} мм"

    # Паттерн для поиска диаметра вида ∅ЧИСЛО или d=ЧИСЛО
    match_diameter_phi = re.search(r'[∅d]=(\d+(?:,\d+)?)', text)
    if match_diameter_phi:
        return f"диаметр {match_diameter_phi.group(1)}"

    # Паттерн для поиска диаметра с "мм"
    match_diameter_mm = re.search(r'диаметром\s*(\d+(?:,\d+)?)', text)
    if match_diameter_mm:
        return f"диаметр {match_diameter_mm.group(1)} мм"

    return None

def extract_dimensions_inches(text):
    text = str(text).lower()

    # Паттерн для поиска размеров в дюймах (целые и дробные)
    match_inches = re.search(r'(\d+(?:\s*\d+/\d+)?)["“”]', text)
    if match_inches:
        return f"диаметр {match_inches.group(1)} дюйм"
    return None

# Предположим, ваш DataFrame с основными трубами называется main_pipes_df
main_pipes_df['dimensions'] = main_pipes_df['all_products'].apply(extract_dimensions)
main_pipes_df['dimensions_inches'] = main_pipes_df['all_products'].apply(extract_dimensions_inches)

print(main_pipes_df[['all_products', 'dimensions', 'dimensions_inches']].head(50))

# Для Листов

import pandas as pd
import re

def extract_sheet_thickness(text):
    text = str(text).lower()

    # Паттерны для толщины
    match_t_eq = re.search(r'[tт]=(\d+(?:,\d+)?)мм', text)
    if match_t_eq:
        return f"{match_t_eq.group(1)} мм"

    match_t_prefix = re.search(r'[tт]=(\d+(?:,\d+))', text)
    if match_t_prefix:
        # Проверяем, не является ли это частью размеров листа
        if not re.search(r'\d+[хx]\d+[хx]\d+', text):
            return f"{match_t_prefix.group(1)} мм" # Предполагаем, что если нет формата лист, то это толщина

    match_thickness_mm = re.search(r'(\d+(?:,\d+)?)мм', text)
    if match_thickness_mm:
        # Проверяем, не является ли это частью размеров листа
        if not re.search(r'\d+[хx]\d+[хx]\d+', text):
            return f"{match_thickness_mm.group(1)} мм" # Предполагаем, что если нет формата лист, то это толщина

    match_thickness_only = re.search(r'лист\s*(\d+(?:,\d+)?)мм', text)
    if match_thickness_only:
        return f"{match_thickness_only.group(1)} мм"

    match_thickness_num_only = re.search(r'(\d+(?:,\d+)?)мм$', text) # Толщина в конце строки
    if match_thickness_num_only:
        return f"{match_thickness_num_only.group(1)} мм"

    match_thickness_num_prefix = re.search(r'(\d+(?:,\d+))х', text) # Число перед "х", если нет "мм" рядом
    if match_thickness_num_prefix:
        # Дополнительная проверка, чтобы не захватить часть размеров
        if not re.search(r'\d+[хx]\d+[хx]\d+', text):
            return f"{match_thickness_num_prefix.group(1)} мм"

    match_thickness_word = re.search(r't=(\d+(?:,\d+)?)', text)
    if match_thickness_word:
        return f"{match_thickness_word.group(1)} мм"

    match_thickness_simple_num = re.search(r'лист\s*(\d+)(?:мм)?$', text) # "Лист 20" или "Лист 20мм"
    if match_thickness_simple_num:
        return f"{match_thickness_simple_num.group(1)} мм"

    return None

def extract_sheet_dimensions(text):
    text = str(text).lower()

    # Паттерн для размеров вида ЧИСЛОхЧИСЛОхЧИСЛОмм
    match_3d_mm = re.search(r'(\d+(?:,\d+)?)[хx](\d+(?:,\d+)?)[хx](\d+(?:,\d+)?)мм', text)
    if match_3d_mm:
        return f"{match_3d_mm.group(1)}x{match_3d_mm.group(2)}x{match_3d_mm.group(3)} мм"

    # Паттерн для размеров вида ЧИСЛОхЧИСЛОмм
    match_2d_mm = re.search(r'(\d+(?:,\d+)?)[хx](\d+(?:,\d+)?)мм', text)
    if match_2d_mm:
        return f"{match_2d_mm.group(1)}x{match_2d_mm.group(2)} мм"

    # Паттерн для размеров вида ЧИСЛОхЧИСЛО
    match_2d = re.search(r'(\d+(?:,\d+)?)[хx](\d+(?:,\d+)?)', text)
    if match_2d:
        return f"{match_2d.group(1)}x{match_2d.group(2)} мм" # Предполагаем, что если нет "мм", то в мм

    return None

# Предположим, ваш DataFrame с листами называется sheets_df
sheets_df['thickness'] = sheets_df['all_products'].apply(extract_sheet_thickness)
sheets_df['dimensions'] = sheets_df['all_products'].apply(extract_sheet_dimensions)

print(sheets_df[['all_products', 'thickness', 'dimensions']].head(50))


""" Для Арматуры"""

import pandas as pd
import re

def extract_armature_characteristics(text):
    text = str(text).lower()
    characteristics = {}

    # Извлечение диаметра
    diameter_match = re.search(r'[фød=]\s*(\d+(?:,\d+)?)мм?', text)
    if diameter_match:
        characteristics['diameter'] = diameter_match.group(1) + ' мм'
    else:
        diameter_match_num_only = re.search(r'арматура\s*(\d+)', text)
        if diameter_match_num_only:
            characteristics['diameter'] = diameter_match_num_only.group(1) + ' мм'

    # Извлечение класса арматуры (буква А и цифра/римская цифра)
    class_match = re.search(r'а[-]?([iвиx]+|\d{3})', text, re.IGNORECASE)
    if class_match:
        characteristics['class'] = 'А' + class_match.group(1).upper()

    # Извлечение марки стали
    steel_grade_match = re.search(r'([0-9]{2}г[с2]?с|35гс|09г2с|вст3сп[2-5]?)', text, re.IGNORECASE)
    if steel_grade_match:
        characteristics['steel_grade'] = steel_grade_match.group(1).upper()

    # Извлечение ГОСТ
    gost_match = re.search(r'гост\s*(\d{4}[-\d]*)', text, re.IGNORECASE)
    if gost_match:
        characteristics['gost'] = 'ГОСТ ' + gost_match.group(1)

    return characteristics

# Предположим, ваш DataFrame с арматурой называется armature_df
armature_df['armature_info'] = armature_df['all_products'].apply(extract_armature_characteristics)

# Для удобства можно развернуть словарь в отдельные столбцы
armature_df = pd.concat([armature_df.drop(['armature_info'], axis=1), armature_df['armature_info'].apply(pd.Series)], axis=1)

print(armature_df[['all_products', 'diameter', 'class', 'steel_grade', 'gost']].head(50))

""" Для Уголков"""

import pandas as pd
import re

def extract_angle_characteristics(text):
    text = str(text).lower()
    characteristics = {}

    # Извлечение размеров (LШxШxТ или ШxШxТ)
    size_match_l = re.search(r'l(\d+)[хx](\d+)[хx](\d+)', text)
    if size_match_l:
        characteristics['size'] = f"{size_match_l.group(1)}x{size_match_l.group(2)}x{size_match_l.group(3)}"
    else:
        size_match = re.search(r'(\d+)[хx](\d+)[хx](\d+)(?:мм)?', text)
        if size_match:
            characteristics['size'] = f"{size_match.group(1)}x{size_match.group(2)}x{size_match.group(3)}"
        else:
            size_match_2d = re.search(r'(\d+)[хx](\d+)(?:мм)?', text)
            if size_match_2d:
                characteristics['size'] = f"{size_match_2d.group(1)}x{size_match_2d.group(2)}"

    # Извлечение толщины (если не было в размере)
    thickness_match = re.search(r'(\d+)мм', text)
    if thickness_match and 'size' not in characteristics:
        characteristics['thickness'] = f"{thickness_match.group(1)} мм"
    elif 'size' in characteristics and len(characteristics['size'].split('x')) < 3:
        thickness_implied = re.search(r'[хx](\d+)(?:мм)?$', characteristics['size'])
        if thickness_implied:
            characteristics['thickness'] = f"{thickness_implied.group(1)} мм"

    # Извлечение марки стали
    steel_grade_match = re.search(r'(ст\s*\d+|с\s*245|ст3сп[2-5]?|09г2с|вст\s*\.\s*3)', text, re.IGNORECASE)
    if steel_grade_match:
        characteristics['steel_grade'] = steel_grade_match.group(1).upper().replace(' ', '')

    # Извлечение покрытия
    if 'оцинкованный' in text:
        characteristics['coating'] = 'оцинкованный'
    elif 'горячеоцинкованный' in text:
        characteristics['coating'] = 'горячеоцинкованный'

    # Извлечение ГОСТ
    gost_match = re.search(r'гост\s*(\d{4}[-\d]*)', text, re.IGNORECASE)
    if gost_match:
        characteristics['gost'] = 'ГОСТ ' + gost_match.group(1)

    # Извлечение типа уголка
    if 'штукатурный' in text:
        characteristics['type'] = 'штукатурный'
        if 'плоский тип' in text:
            characteristics['subtype'] = 'плоский'
        elif 'угловой тип внешний' in text:
            characteristics['subtype'] = 'угловой внешний'
    elif 'периметральный' in text:
        characteristics['type'] = 'периметральный'
    elif 'равнополочный' in text:
        characteristics['type'] = 'равнополочный'
    elif 'наружный для плинтуса' in text:
        characteristics['type'] = 'для плинтуса наружный'
    elif 'внутренний для плинтуса' in text:
        characteristics['type'] = 'для плинтуса внутренний'
    elif 'наружный для плитки' in text:
        characteristics['type'] = 'для плитки наружный'
        if 'нерж' in text:
            characteristics['material'] = 'нержавеющая сталь'
    elif 'перфорированный' in text:
        characteristics['type'] = 'перфорированный'
    elif 'для плинтуса' in text:
        characteristics['type'] = 'для плинтуса'
        color_match = re.search(r'коричневый', text)
        if color_match:
            characteristics['color'] = 'коричневый'
    elif 'потребителя' in text:
        characteristics['type'] = 'потребителя'
    elif re.search(r'угф-\d', text):
        characteristics['type'] = re.search(r'(угф-\d)', text).group(1).upper()

    return characteristics

# Предположим, ваш DataFrame с уголками называется angles_df
angles_df['angle_info'] = angles_df['all_products'].apply(extract_angle_characteristics)

# Разворачиваем словарь в отдельные столбцы
angles_df = pd.concat([angles_df.drop(['angle_info'], axis=1), angles_df['angle_info'].apply(pd.Series)], axis=1)

print(angles_df[['all_products', 'size', 'thickness', 'steel_grade',
                 'coating', 'gost', 'type', 'subtype', 'material', 'color']].head(50))


""" Для Полоса"""

import pandas as pd
import re

def extract_strip_characteristics(text):
    text = str(text).lower()
    characteristics = {}

    # Извлечение размеров (ШхТ или ТхШ)
    size_match_sht = re.search(r'(\d+(?:\,\d+)?)[х\*](\d+(?:\,\d+)?)мм', text)
    if size_match_sht:
        characteristics['width'] = size_match_sht.group(1) + ' мм'
        characteristics['thickness'] = size_match_sht.group(2) + ' мм'
    else:
        size_match_thw = re.search(r'(\d+(?:\,\d+)?)[х\*](\d+(?:\,\d+)?)', text)
        if size_match_thw:
            # Порядок может быть разным, попробуем определить по контексту или оставим оба варианта
            characteristics['size'] = f"{size_match_thw.group(1)}x{size_match_thw.group(2)}"

    # Извлечение длины
    length_match_mm = re.search(r'[хl]\s*(\d+)мм', text)
    if length_match_mm:
        characteristics['length'] = length_match_mm.group(1) + ' мм'
    length_match_m = re.search(r'[хl]=\s*(\d+)м', text)
    if length_match_m:
        characteristics['length'] = f"{float(length_match_m.group(1)) * 1000} мм" # Переводим в мм

    # Извлечение марки стали
    steel_grade_match = re.search(r'(ст\s*\d+[а-я0-9]*|с\s*245)', text, re.IGNORECASE)
    if steel_grade_match:
        characteristics['steel_grade'] = steel_grade_match.group(1).upper().replace(' ', '')

    # Извлечение покрытия
    if 'горячеоцинкованной' in text or 'горячеоцинкованная' in text:
        characteristics['coating'] = 'горячеоцинкованная'
    elif 'оцинкованной' in text or 'оцинкованная' in text:
        characteristics['coating'] = 'оцинкованная'

    # Извлечение ГОСТ
    gost_match = re.search(r'гост\s*(\d{3,}[-\d]*)', text, re.IGNORECASE)
    if gost_match:
        characteristics['gost'] = 'ГОСТ ' + gost_match.group(1)

    # Извлечение типа полосы
    if 'перфорированная' in text:
        characteristics['type'] = 'перфорированная'
        if re.search(r'к\d+', text.upper()):
            characteristics['subtype'] = re.search(r'(К\d+)', text.upper()).group(1)
    elif 'для верхней части двери' in text:
        characteristics['type'] = 'для верхней части двери'
    elif 'для торцевой части двери' in text:
        characteristics['type'] = 'для торцевой части двери'
    elif 'для верхней части окон' in text:
        characteristics['type'] = 'для верхней части окон'
    elif 'для торцевой части окон' in text:
        characteristics['type'] = 'для торцевой части окон'
    elif 'для фиксатора крышки' in text:
        characteristics['type'] = 'для фиксатора крышки'
    elif 'заземления' in text:
        characteristics['type'] = 'заземления'
    elif 'износостойкая' in text:
        characteristics['type'] = 'износостойкая'
        subtype_match = re.search(r'(gieba\s*w\d+[/w\d+-в\s*кожух\s*\d+-\d+|w\d+-b[/w\d+-b]\s*\d+-\d+)', text, re.IGNORECASE)
        if subtype_match:
            characteristics['subtype'] = subtype_match.group(1).upper()
    elif 'стальная' in text:
        characteristics['type'] = 'стальная'

    return characteristics

# Предположим, ваш DataFrame с полосами называется strips_df
strips_df['strip_info'] = strips_df['all_products'].apply(extract_strip_characteristics)

# Разворачиваем словарь в отдельные столбцы
strips_df = pd.concat([strips_df.drop(['strip_info'], axis=1), strips_df['strip_info'].apply(pd.Series)], axis=1)

print(strips_df[['all_products', 'width', 'thickness', 'size', 'length', 'steel_grade', 'coating', 'gost', 'type', 'subtype']].head(50))


""" Для Швеллеров"""

import pandas as pd
import re

def extract_channel_characteristics(text):
    text = str(text).lower()
    characteristics = {}

    # Извлечение типоразмера (номер и тип полки)
    channel_match = re.search(r'швеллер\s*(\d+[а-я]*)', text)
    if channel_match:
        characteristics['type_size'] = channel_match.group(1).upper()
    else:
        channel_match_num_only = re.search(r'швеллер\s*(\d+)(?:\s*[а-я]*)', text)
        if channel_match_num_only:
            characteristics['type_size'] = channel_match_num_only.group(1)
        else:
            channel_match_simple = re.search(r'^(\d+[а-я]?)\s+', text)
            if channel_match_simple:
                characteristics['type_size'] = channel_match_simple.group(1).upper()
            else:
                channel_match_num_only_start = re.search(r'^(\d+)\s+', text)
                if channel_match_num_only_start:
                    characteristics['type_size'] = channel_match_num_only_start.group(1)

    # Извлечение размеров (высота х ширина полки х толщина стенки)
    size_match = re.search(r'(\d+)[х\*](\d+)[х\*](\d+)(?:t)?', text)
    if size_match:
        characteristics['dimensions'] = f"{size_match.group(1)}x{size_match.group(2)}x{size_match.group(3)}"
    else:
        size_match_c = re.search(r'с(\d+)[х\*](\d+)[х\*](\d+)(?:t)?', text)
        if size_match_c:
            characteristics['dimensions'] = f"{size_match_c.group(1)}x{size_match_c.group(2)}x{size_match_c.group(3)}"
        else:
            size_match_2d_c = re.search(r'с(\d+)[х\*](\d+)[х\*]([\d\.,]+)t', text)
            if size_match_2d_c:
                characteristics['dimensions'] = f"{size_match_2d_c.group(1)}x{size_match_2d_c.group(2)}x{size_match_2d_c.group(3)}"
            else:
                size_match_2d = re.search(r'(\d+)[х\*](\d+)[х\*]([\d\.,]+)t', text)
                if size_match_2d:
                    characteristics['dimensions'] = f"{size_match_2d.group(1)}x{size_match_2d.group(2)}x{size_match_2d.group(3)}"
            else:
                size_match_2d_simple = re.search(r'(\d+)[х\*](\d+)[х\*](\d+)', text)
                if size_match_2d_simple:
                    characteristics['dimensions'] = f"{size_match_2d_simple.group(1)}x{size_match_2d_simple.group(2)}x{size_match_2d_simple.group(3)}"
                else:
                    size_match_1d_p = re.search(r'(\d+)п\s+(\d+)[х\*](\d+)[х\*](\d+)', text)
                    if size_match_1d_p:
                        characteristics['dimensions'] = f"{size_match_1d_p.group(2)}x{size_match_1d_p.group(3)}x{size_match_1d_p.group(4)}"

    # Извлечение длины
    length_match = re.search(r'(\d+)мм', text)
    if length_match:
        characteristics['length'] = length_match.group(1) + ' мм'

    # Извлечение марки стали
    steel_grade_match = re.search(r'(с\s*\d+|ст\s*\d+[а-я0-9]*)', text, re.IGNORECASE)
    if steel_grade_match:
        characteristics['steel_grade'] = steel_grade_match.group(1).upper().replace(' ', '')

    # Извлечение покрытия
    if 'оцинкованный' in text:
        characteristics['coating'] = 'оцинкованный'

    # Извлечение ГОСТ
    gost_match = re.search(r'гост\s*(\d{4}[-\d]*)', text, re.IGNORECASE)
    if gost_match:
        characteristics['gost'] = 'ГОСТ ' + gost_match.group(1)

    # Определение типа швеллера по букве
    if 'type_size' in characteristics:
        if characteristics['type_size'][-1].isalpha():
            type_letter = characteristics['type_size'][-1].upper()
            if type_letter == 'П':
                characteristics['type'] = 'с параллельными полками'
            elif type_letter == 'У':
                characteristics['type'] = 'с уклоном полок'
            elif type_letter == 'А':
                characteristics['type'] = 'облегченный'
            elif type_letter == 'С':
                characteristics['type'] = 'специальный'
            characteristics['type_number'] = characteristics['type_size'][:-1]
        else:
            characteristics['type_number'] = characteristics['type_size']

    # Дополнительные ключевые слова
    if 'гнутый' in text:
        characteristics['additional'] = 'гнутый'
    elif 'горячекатаный' in text:
        characteristics['additional'] = 'горячекатаный'

    return characteristics

# Предположим, ваш DataFrame со швеллерами называется channels_df
channels_df['channel_info'] = channels_df['all_products'].apply(extract_channel_characteristics)

# Разворачиваем словарь в отдельные столбцы
channels_df = pd.concat([channels_df.drop(['channel_info'], axis=1), channels_df['channel_info'].apply(pd.Series)], axis=1)

print(channels_df[['all_products', 'type_number', 'type', 'dimensions', 'length', 'steel_grade',
                   'coating', 'gost', 'additional']].head(50))

""" Для Двутавров"""

import pandas as pd
import re

def extract_i_beam_characteristics(text):
    text = str(text).lower()
    characteristics = {}

    # Извлечение типоразмера (номер и серия)
    i_beam_match = re.search(r'двутавр\s*([iw]?\d+[а-я\d]*)', text, re.IGNORECASE)
    if i_beam_match:
        characteristics['type_size'] = i_beam_match.group(1).upper()
    else:
        i_beam_match_start = re.search(r'^([iw]?\d+[а-я\d]*)\s+', text, re.IGNORECASE)
        if i_beam_match_start:
            characteristics['type_size'] = i_beam_match_start.group(1).upper()
        else:
            i_beam_match_hw = re.search(r'двутавр\s*(hw\d+)', text, re.IGNORECASE)
            if i_beam_match_hw:
                characteristics['type_size'] = i_beam_match_hw.group(1).upper()
            else:
                i_beam_match_w = re.search(r'двутавр\s*(w\d+x\d+)', text, re.IGNORECASE)
                if i_beam_match_w:
                    characteristics['type_size'] = i_beam_match_w.group(1).upper()
                else:
                    i_beam_match_i = re.search(r'([i]\d+[а-я\d]+)', text, re.IGNORECASE)
                    if i_beam_match_i:
                        characteristics['type_size'] = i_beam_match_i.group(1).upper()

    # Извлечение размеров (высота х ширина полки х толщина полки х толщина стенки)
    size_match_4d = re.search(r'(\d+)[х\*](\d+)[х\*](\d+)[х\*](\d+)', text)
    if size_match_4d:
        characteristics['dimensions'] = f"{size_match_4d.group(1)}x{size_match_4d.group(2)}x{size_match_4d.group(3)}x{size_match_4d.group(4)}"
    else:
        size_match_3d = re.search(r'(\d+)[х\*](\d+)[х\*](\d+)', text)
        if size_match_3d:
            characteristics['dimensions'] = f"{size_match_3d.group(1)}x{size_match_3d.group(2)}x{size_match_3d.group(3)}"
        else:
            size_match_hw_dim = re.search(r'hw(\d+)', characteristics.get('type_size', ''), re.IGNORECASE)
            if size_match_hw_dim:
                characteristics['height_mm'] = size_match_hw_dim.group(1)
            else:
                size_match_w_dim = re.search(r'w(\d+)x(\d+)', characteristics.get('type_size', ''), re.IGNORECASE)
                if size_match_w_dim:
                    characteristics['height_inch'] = size_match_w_dim.group(1)
                    characteristics['weight_per_foot_lbs'] = size_match_w_dim.group(2)

    # Извлечение марки стали
    steel_grade_match = re.search(r'(с\s*\d+[а-я0-9-]*|ст\s*\d+[а-я0-9/]*|[a-z]+\s*\d+)', text, re.IGNORECASE)
    if steel_grade_match:
        characteristics['steel_grade'] = steel_grade_match.group(1).upper().replace(' ', '')

    # Извлечение стандартов
    standard_match_gost = re.search(r'гост\s*([р\s]?\d{4}[-\d]*)', text, re.IGNORECASE)
    if standard_match_gost:
        characteristics['gost'] = 'ГОСТ ' + standard_match_gost.group(1).upper().replace(' ', '')
    standard_match_sto = re.search(r'сто\s*([а-я\d\s-]+)', text, re.IGNORECASE)
    if standard_match_sto:
        characteristics['sto'] = 'СТО ' + standard_match_sto.group(1).upper().replace(' ', '')
    standard_match_astm = re.search(r'astm\s*([a-z\d]+)', text, re.IGNORECASE)
    if standard_match_astm:
        characteristics['astm'] = 'ASTM ' + standard_match_astm.group(1).upper()

    # Извлечение длины
    length_match = re.search(r'(\d+)мм', text)
    if length_match:
        characteristics['length'] = length_match.group(1) + ' мм'

    # Дополнительные характеристики
    if 'широкополочный' in text:
        characteristics['type'] = 'широкополочный'
    elif 'горячекатаный' in text:
        characteristics['process'] = 'горячекатаный'

    return characteristics

# Предположим, ваш DataFrame с двутаврами называется i_beams_df
i_beams_df['i_beam_info'] = i_beams_df['all_products'].apply(extract_i_beam_characteristics)

# Разворачиваем словарь в отдельные столбцы
i_beams_df = pd.concat([i_beams_df.drop(['i_beam_info'], axis=1), i_beams_df['i_beam_info'].apply(pd.Series)], axis=1)

print(i_beams_df[['all_products', 'type_size', 'dimensions', 'height_mm', 'height_inch',
                  'weight_per_foot_lbs', 'steel_grade', 'gost', 'sto', 'astm', 'length',
                  'type', 'process']].head(50))

