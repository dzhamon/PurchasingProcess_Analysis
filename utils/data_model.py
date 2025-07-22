import os
import threading
from PyQt5.QtWidgets import QFileDialog, QProgressBar, QMessageBox, QWidget
from PyQt5.QtCore import pyqtSignal, QObject
import sys
import pandas as pd
import sqlite3

from utils.config import SQL_PATH
from utils.logic import (clean_data_from_xls, upload_to_sql_df,
                         clean_contr_data_from_xls, isfilepresent, addfilename)
from utils.functions import (prepare_main_datas, create_treeview_table, trim_actor_name)



pd.options.display.float_format = '{:,.2f}'.format


class DataModel(QWidget):
    progress_update = pyqtSignal(int)
    
    def __init__(self, window):
        super().__init__()
        self.frame_ontop_2 = None
        self.data = {}
        self.mywindow = window
        self.progress_bar = QProgressBar()
        

    def  open_file_dialog(self):
        try:
            file, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "",
                                                  "All Files (*);;Excel Files (*.xlsx);;Text Files (*.txt)")
            if file:
                print(file)
                # код для обработки выбранного файла
            else:
                reply = QMessageBox.question(self, "Отмена выбора файла",
                                             "Вы не выбрали файл. Хотите попробовать снова?",
                                             QMessageBox.Retry | QMessageBox.Cancel)
                if reply == QMessageBox.Retry:
                    return self.open_file_dialog()
                else:
                    # Пользователь отменил выбор, ничего не делаем и выходим из метода
                    return
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            QMessageBox.warning(self, "Ошибка", "Произошла ошибка при выборе файла!")

        def real_traitement():
            list_files = []
            file_name = os.path.basename(file)
            name, extension = file_name.rsplit(".", 1)
            simb = name.split("_", 1)[0]
            
            if extension == 'xlsx' and (simb == 'KP' or simb == 'Contr'):
                
                if simb == 'KP':
                    
                    list_files = isfilepresent()
                    if file_name not in list_files:
                        # добавить имя в таблицу files_name
                        df = clean_data_from_xls(file)
                        conn = sqlite3.connect(SQL_PATH)
                        cur = conn.cursor()
                        cur.execute("DELETE FROM data_tmp")
                        upload_to_sql_df(df, conn, "data_tmp")
                        
                        cur.executescript(
                            """INSERT INTO data_kp(lot_number, lot_status, discipline, project_name,
                                open_date, close_date, actor_name, good_name,
                                good_count, unit, supplier_qty, supplier_unit,
                                winner_name, unit_price, total_price, currency)
                            SELECT
                            a.lot_number, a.lot_status, a.discipline, a.project_name,
                            a.open_date, a.close_date, a.actor_name, a.good_name,
                            a.good_count, a.unit, a.supplier_qty, a.supplier_unit,
                            a.winner_name, a.unit_price, a.total_price, a.currency
                            FROM data_tmp AS a;"""
                        )
                        conn.commit()
                        addfilename(file_name)
                        print("Данные из файла ", file_name, ' вставлены в Таблицу')
                       
                    else:
                        # если файл содержится в таблице files_name
                        print("Информация из файла ", file_name, " уже собержится в базе данных")
                # если файл начинается с Contr
                else: #если simb == "Contr"
                    list_files = isfilepresent()
                    if file_name not in list_files:
                        df = clean_contr_data_from_xls(file)
                        conn = sqlite3.connect(SQL_PATH)
                        cur = conn.cursor()
                        cur.execute("DELETE FROM data_contr_tmp")
                        upload_to_sql_df(df, conn, "data_contr_tmp")
                        
                        cur.execute(
                            """INSERT INTO data_contract(
                                    lot_number ,
                                    lot_end_date ,
                                    contract_number ,
                                    contract_signing_date ,
                                    contract_name ,
                                    executor_dak ,
                                    counterparty_name ,
                                    product_name ,
                                    supplier_unit ,
                                    quantity ,
                                    unit ,
                                    unit_price ,
                                    product_amount ,
                                    additional_expenses ,
                                    total_contract_amount ,
                                    contract_currency ,
                                    delivery_conditions ,
                                    payment_conditions ,
                                    delivery_time_days ,
                                    discipline )
                              SELECT
                                    a.lot_number ,
                                    a.lot_end_date ,
                                    a.contract_number ,
                                    a.contract_signing_date ,
                                    a.contract_name ,
                                    a.executor_dak ,
                                    a.counterparty_name ,
                                    a.product_name ,
                                    a.supplier_unit ,
                                    a.quantity ,
                                    a.unit ,
                                    a.unit_price ,
                                    a.product_amount ,
                                    a.additional_expenses ,
                                    a.total_contract_amount ,
                                    a.contract_currency ,
                                    a.delivery_conditions ,
                                    a.payment_conditions ,
                                    a.delivery_time_days ,
                                    a.discipline
                              FROM data_contr_tmp AS a;"""
                        )
                        conn.commit()
                        cur.execute('DELETE FROM data_contract WHERE quantity = 0.0')
                        conn.commit()
                        addfilename(file_name)
                        print("Данные файла ", file_name, " успешно вставлены в Таблицу")
                        
                    else:
                        # если файл содержится в базе данных
                        print("Информация из  ", file_name, " уже есть в базе данных" )
            else:
                choice = QMessageBox.retry(self, "Ошибка выбора файла!", "Повторите выбор файла")
                if choice == QMessageBox.Retry:
                    return None
                
        threading.Thread(target=real_traitement).start()
        
    def open_from_db(self):
        conn = sqlite3.connect(SQL_PATH)
        data_df = pd.read_sql("select * from data_kp", conn)
        print('Основной датафрейм формируется из БД здесь')
        self.mywindow.data_df = data_df
        
        # Применение функции к столбцу 'actor_name'
        data_df['actor_name'] = data_df['actor_name'].apply(trim_actor_name)

        str_query_1 = ("""
                        SELECT discipline as 'Дисциплина', currency as 'Валюта контракта',
                        sum(total_price) as 'Всего в валюте контракта',
                        min(total_price) as 'Минимальная сумма', max(total_price) as 'Максимальная сумма'
                        FROM data_kp WHERE currency not null and total_price <> 0
                        GROUP BY discipline, currency;
                        """)
        df_query_1 = prepare_main_datas(str_query_1)
        print(df_query_1)

        str_query_2 = ("""
                        SELECT discipline as 'Дисциплина', currency as 'Валюта контракта',
                        count(DISTINCT(lot_number)) as 'Количество лотов'
                        FROM data_kp WHERE currency not null GROUP BY discipline, currency;
                        """)
        df_query_2 = prepare_main_datas(str_query_2)
        print(df_query_2)

        str_query_3 = ("""
                        SELECT discipline as 'Дисциплина', actor_name as 'Исполнитель', currency as 'Валюта',
                        count(distinct(lot_number)) as 'Кол-во проработ. лотов'
                        FROM data_kp WHERE currency IS NOT NULL AND discipline IS NOT NULL
                        GROUP BY discipline, actor_name, currency;
                        """)
        df_query_3 = prepare_main_datas(str_query_3)
        print(df_query_3)

        param1, param2 = create_treeview_table(df=df_query_3)
        print(param1, param2)

    def prepare_analytic_data(self):
        conn = sqlite3.connect(SQL_PATH)
        cur = conn.cursor()

        cur.execute("""
                    SELECT distinct(data_contract.lot_number), data_contract.close_date,
                        data_contract.contract_maker, data_contract.contract_keeper,
                        data_contract.good_name, data_contract.good_count, data_contract.unit,
                        data_contract.unit_price, data_contract.total_price, data_contract.currency
                    FROM data_contract
                    WHERE data_contract.close_date >= '2023-01-01' AND
                    data_contract.lot_number NOT IN
                    (SELECT distinct(lot_number) FROM data_kp WHERE close_date >= '2023-01-01')
                    ORDER BY data_contract.contract_keeper;
                    """)
        columns = [column[0] for column in cur.description]
        values = cur.fetchall()
        row_dict = {}
        for i, column in enumerate(columns):
            row_dict[column] = [value[i] for value in values]
        df_1 = pd.DataFrame(row_dict)
        print('df_1 = ', df_1)
        df_1.to_excel('contracts_without_KP.xlsx')
