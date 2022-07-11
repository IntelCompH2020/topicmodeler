"""
* *IntelComp H2020 project*

Class that defines a customized QWidget that is instantiated and embedded within the 'generate_tm_corpus_window' a number of times equal to the number of local datasets selected for the generation of the training corpus. From this widget, the fields to be used for the raw text and the lemmas can be chosen by the user.
"""

from PyQt6.QtWidgets import QTableWidgetItem, QWidget
from PyQt6.uic import loadUi
from src.gui.utils import utils
from src.gui.utils.constants import Constants


class WidgetCreateTMCorpus(QWidget):
    def __init__(self, columns_dts):
        """
        Initializes the WidgetCreateTMCorpus that is embedded within the 'generate_tm_corpus_window' subwindow.

        Parameters
        ----------
        columns_dts: list
            List of the fields available for the dataset this widget refers
        """

        super(WidgetCreateTMCorpus, self).__init__()

        ########################################################################
        # Load UI
        ########################################################################
        loadUi("src/gui/uis/widget_createTMCorpus.ui", self)

        ########################################################################
        # ATTRIBUTES
        ########################################################################
        self.columns_dts = columns_dts

        ########################################################################
        # Widgets initial configuration
        ########################################################################
        # Configure tables
        self.populate_widgets()
        utils.configure_table_header(Constants.CREATE_TM_CORPUS_TABLES, self)

    def populate_widgets(self):
        """It fills the contents of the widgets contained in the WidgetCreateTMCorpus.
        """        

        self.comboBox_trdts_id.addItems(self.columns_dts)
        self.tableWidget_available_fields_raw.setRowCount(
            len(self.columns_dts)-1)
        self.tableWidget_available_fields_lemmas.setRowCount(
            len(self.columns_dts)-1)
        self.tableWidget_fields_to_include_raw.setRowCount(
            len(self.columns_dts)-1)
        self.tableWidget_fields_to_include_lemmas.setRowCount(
            len(self.columns_dts)-1)

        for i in range(len(self.columns_dts)):
            if str(self.columns_dts[i]) != "id":
                self.tableWidget_available_fields_raw.setItem(
                    i, 0, QTableWidgetItem(str(self.columns_dts[i])))
                self.tableWidget_available_fields_lemmas.setItem(
                    i, 0, QTableWidgetItem(str(self.columns_dts[i])))
        
        return
