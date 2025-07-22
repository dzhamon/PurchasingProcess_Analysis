import pandas as pd
from typing import Dict, Union, List, Optional
from pathlib import Path


class ExcelExporter:
	"""
	Класс для экспорта нескольких DataFrame в один файл Excel с разными листами.

	Параметры:
	----------
	file_path : Union[str, Path]
		Путь к файлу Excel для сохранения
	mode : str, optional
		Режим записи: 'w' - перезаписать, 'a' - добавить (по умолчанию 'w')
	engine : str, optional
		Движок для записи Excel (по умолчанию 'openpyxl')
	"""
	
	def __init__(
			self,
			file_path: Union[str, Path],
			mode: str = 'w',
			engine: str = 'openpyxl'
	):
		self.file_path = Path(file_path)
		self.mode = mode
		self.engine = engine
		self.writer = None
	
	def __enter__(self):
		"""Контекстный менеджер для безопасной работы с файлом"""
		self.writer = pd.ExcelWriter(
			self.file_path,
			engine=self.engine,
			mode=self.mode
		)
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Автоматическое закрытие writer при выходе из контекста"""
		if self.writer is not None:
			self.writer.close()
	
	def save_dataframe(
			self,
			df: pd.DataFrame,
			sheet_name: str,
			index: bool = False,
			header: bool = True,
			startrow: int = 0,
			startcol: int = 0,
			float_format: Optional[str] = None,
			columns: Optional[List[str]] = None,
			**to_excel_kwargs
	) -> None:
		"""
		Сохраняет DataFrame на указанный лист в Excel файл.

		Параметры:
		----------
		df : pd.DataFrame
			DataFrame для сохранения
		sheet_name : str
			Название листа в Excel
		index : bool, optional
			Включать индекс (по умолчанию False)
		header : bool, optional
			Включать заголовки (по умолчанию True)
		startrow : int, optional
			Строка для начала записи (по умолчанию 0)
		startcol : int, optional
			Столбец для начала записи (по умолчанию 0)
		float_format : str, optional
			Формат чисел с плавающей точкой (например, "%.2f")
		columns : list, optional
			Список колонок для экспорта (по умолчанию все)
		to_excel_kwargs : dict
			Дополнительные аргументы для pd.DataFrame.to_excel()
		"""
		if self.writer is None:
			raise ValueError("Excel writer не инициализирован. Используйте контекстный менеджер (with).")
		
		# Выбираем только указанные колонки, если они заданы
		df_to_export = df[columns] if columns is not None else df
		
		df_to_export.to_excel(
			self.writer,
			sheet_name=sheet_name,
			index=index,
			header=header,
			startrow=startrow,
			startcol=startcol,
			float_format=float_format,
			**to_excel_kwargs
		)
	
	def save_multiple_dataframes(
			self,
			dataframes: Dict[str, pd.DataFrame],
			sheet_names: Optional[Union[List[str], Dict[str, str]]] = None,
			**to_excel_kwargs
	) -> None:
		"""
		Сохраняет несколько DataFrame в один файл на разные листы.

		Параметры:
		----------
		dataframes : dict
			Словарь {ключ: DataFrame} или список DataFrame
		sheet_names : list or dict, optional
			Список названий листов или словарь {ключ: название листа}
		to_excel_kwargs : dict
			Дополнительные аргументы для pd.DataFrame.to_excel()
		"""
		if isinstance(dataframes, dict):
			if sheet_names is None:
				sheet_names = {k: str(k) for k in dataframes.keys()}
			elif isinstance(sheet_names, list):
				if len(sheet_names) != len(dataframes):
					raise ValueError("Количество названий листов не совпадает с количеством DataFrame")
				sheet_names = {k: v for k, v in zip(dataframes.keys(), sheet_names)}
			
			for df_key, df in dataframes.items():
				self.save_dataframe(df, sheet_name=sheet_names[df_key], **to_excel_kwargs)
		elif isinstance(dataframes, (list, tuple)):
			if sheet_names is None:
				sheet_names = [f"Sheet{i + 1}" for i in range(len(dataframes))]
			elif isinstance(sheet_names, dict):
				raise ValueError("Для списка DataFrame sheet_names должен быть списком")
			
			if len(sheet_names) != len(dataframes):
				raise ValueError("Количество названий листов не совпадает с количеством DataFrame")
			
			for df, sheet_name in zip(dataframes, sheet_names):
				self.save_dataframe(df, sheet_name=sheet_name, **to_excel_kwargs)
		else:
			raise TypeError("dataframes должен быть словарем, списком или кортежем")