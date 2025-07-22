# date_range_selector.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDateEdit, QPushButton, QMessageBox
from PyQt5.QtCore import QDate


class DateRangeSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Создаем layout
        layout = QVBoxLayout()

        # Создаем интерфейс выбора дат
        self.start_date_edit, self.end_date_edit, self.confirm_button = self.create_date_range_selector()

        # Подключаем кнопку к функции
        self.confirm_button.clicked.connect(self.on_confirm)

        # Добавляем элементы в layout
        layout.addWidget(self.start_date_edit)
        layout.addWidget(self.end_date_edit)
        layout.addWidget(self.confirm_button)

        # Устанавливаем layout для основного окна
        self.setLayout(layout)

    def create_date_range_selector(self):
        """
        Создает и возвращает виджеты для выбора начальной и конечной даты,
        а также кнопку подтверждения.
        """
        # Выбор начальной даты
        start_date_label = QLabel("Начальная дата:")
        start_date_edit = QDateEdit(self)
        start_date_edit.setDate(QDate.currentDate())  # Устанавливаем текущую дату по умолчанию
        start_date_edit.setCalendarPopup(True)  # Включаем выпадающий календарь

        # Выбор конечной даты
        end_date_label = QLabel("Конечная дата:")
        end_date_edit = QDateEdit(self)
        end_date_edit.setDate(QDate.currentDate())  # Устанавливаем текущую дату по умолчанию
        end_date_edit.setCalendarPopup(True)  # Включаем выпадающий календарь

        # Кнопка подтверждения
        confirm_button = QPushButton("Загрузить данные", self)

        return start_date_edit, end_date_edit, confirm_button

    def on_confirm(self):
        """
        Обрабатывает нажатие кнопки подтверждения.
        Проверяет даты и вызывает функцию загрузки данных.
        """
        # Получаем выбранные даты
        start_date = self.start_date_edit.date()
        end_date = self.end_date_edit.date()

        # Проверка: конечная дата не может быть раньше начальной
        if start_date > end_date:
            QMessageBox.warning(self, "Ошибка", "Конечная дата не может быть раньше начальной!")
            return

        # Проверка: даты не могут быть пустыми (в данном случае это маловероятно, но на будущее)
        if start_date.isNull() or end_date.isNull():
            QMessageBox.warning(self, "Ошибка", "Даты не могут быть пустыми!")
            return

        # Преобразуем даты в строки
        start_date_str = start_date.toString("yyyy-MM-dd")
        end_date_str = end_date.toString("yyyy-MM-dd")

        # Выводим даты (можно заменить на вызов функции загрузки данных)
        print(f"Выбрана начальная дата: {start_date_str}")
        print(f"Выбрана конечная дата: {end_date_str}")

        # Вызываем функцию загрузки данных
        self.load_data_from_db(start_date_str, end_date_str)

    def load_data_from_db(self, start_date, end_date):
        """
        Пример функции загрузки данных из базы данных.
        """
        query = f"""
        SELECT * FROM your_table_name
        WHERE date_column BETWEEN '{start_date}' AND '{end_date}'
        """
        print(f"Выполняется запрос: {query}")
        # Здесь можно добавить код для выполнения SQL-запроса и загрузки данных