import sqlite3

DB_PATH = "D:\Project_DataFiles\data\sql_krd_new.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("📌 Таблицы в базе данных:", tables)

conn.close()