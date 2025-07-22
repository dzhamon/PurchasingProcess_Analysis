from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
import pandas as pd

class PandasModel(QAbstractTableModel):
    # def __init__(self, dataframe = pd.DataFrame(), parent=None):
    def __init__(self, dataframe=pd.DataFrame(), parent=None):
        super(PandasModel, self).__init__(parent)
        self._dataframe = dataframe

    def rowCount(self, parent=None):
        return len(self._dataframe)

    def columnCount(self, parent=None):
        return len(self._dataframe.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            row, column = index.row(), index.column()
            value = self._dataframe.iloc[row, column]
            return str(value) if not pd.isna(value) else ''
            #return str(self._dataframe.iloc[row, column])
        return None
   
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(section)
        return None
