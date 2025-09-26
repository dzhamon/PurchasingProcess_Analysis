"""
    Эта функция принимает один или несколько DataFrame и записывает
    их в файл Excel
"""
import pandas as pd
from openpyxl import load_workbook
from PyQt5.QtWidgets import QMessageBox, QApplication

def save_multiple_dfs_to_excel(dfs_dict, file_name):
    """
    Сохраняет несколько DataFrame на разные листы в один Excel-файл,
    применяя форматирование.

    Args:
        dfs_dict (dict): Словарь, где ключи - названия листов,
                        а значения - соответствующие pd.DataFrame.
        file_name (str): Имя выходного Excel-файла (например, 'multi_sheet_report.xlsx').
    """
    try:
        # Шаг 1: Записываем каждый DataFrame на отдельный лист
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            for sheet_name, df in dfs_dict.items():
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Шаг 2: Открываем файл для форматирования
        book = load_workbook(file_name)
        
        # Шаг 3: Проходим по каждому листу и применяем форматирование
        for sheet_name in dfs_dict.keys():
            sheet = book[sheet_name]
            
            for col in sheet.columns:
                max_length = 0
                column_letter = col[0].column_letter
                
                for cell in col:
                    cell_value_str = str(cell.value) if cell.value is not None else ""
                    if len(cell_value_str) > max_length:
                        max_length = len(cell_value_str)
                    
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '#,##0.00'
                        if isinstance(cell.value, int):
                            cell.number_format = '#,##00'
                
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column_letter].width = adjusted_width
        
        # Шаг 4: Сохраняем итоговый файл
        book.save(file_name)
        print("Все DataFrame успешно сохранены и отформатированы  {file_name}")
    except Exception as e:
        print("Произошла ошибка при сохранении файла")
    
    return