from PyQt5.QtCore import QAbstractTableModel, Qt

class PandasModel(QAbstractTableModel):
    def __init__(self, df, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df.index)
        # start_idx = self.current_page * self.page_size
        # end_idx = start_idx + self.page_size
        # return min(self.page_size, len(self._df) - start_idx)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
            return None

# Добавьте обработчики для кнопок
def headerData(self, col, orientation, role):
    if orientation == Qt.Horizontal and role == Qt.DisplayRole:
        return self._df.columns[col]
    return None
