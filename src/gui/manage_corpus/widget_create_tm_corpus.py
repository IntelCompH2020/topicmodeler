from PyQt6.QtWidgets import QWidget, QTableWidgetItem
from PyQt6.uic import loadUi

from src.gui.utils import utils
from src.gui.utils.constants import Constants


class WidgetCreateTMCorpus(QWidget):
    def __init__(self, columns_dts):
        """

        """

        super(WidgetCreateTMCorpus, self).__init__()

        #####################################################################################
        # Load UI
        #####################################################################################
        loadUi("src/gui/uis/widget_createTMCorpus.ui", self)

        #####################################################################################
        # ATTRIBUTES
        #####################################################################################
        self.columns_dts = columns_dts

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        # Configure tables
        self.populate_widgets()
        utils.configure_table_header(Constants.CREATE_TM_CORPUS_TABLES, self)

    def populate_widgets(self):
        self.comboBox_trdts_id.addItems(self.columns_dts)
        self.tableWidget_available_fields_raw.setRowCount(len(self.columns_dts))
        self.tableWidget_available_fields_lemmas.setRowCount(len(self.columns_dts))
        self.tableWidget_fields_to_include_raw.setRowCount(len(self.columns_dts))
        self.tableWidget_fields_to_include_lemmas.setRowCount(len(self.columns_dts))

        for i in range(len(self.columns_dts)):
            self.tableWidget_available_fields_raw.setItem(
                i, 0, QTableWidgetItem(str(self.columns_dts[i])))
            self.tableWidget_available_fields_lemmas.setItem(
                i, 0, QTableWidgetItem(str(self.columns_dts[i])))

