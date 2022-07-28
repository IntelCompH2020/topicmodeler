"""
* *IntelComp H2020 project*

Class that defines a customized QWidget that is instantiated and embedded within the 'generate_tm_corpus_window' a number of times equal to the number of local datasets selected for the generation of the training corpus. From this widget, the fields to be used for the raw text and the lemmas can be chosen by the user.
"""

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QTableWidgetItem, QWidget
from PyQt6.uic import loadUi
from src.gui.utils import utils
from src.gui.utils.constants import Constants


class WidgetSimilarTopics(QWidget):
    def __init__(self, pair, df):
        """
        Initializes the WidgetCreateTMCorpus that is embedded within the 'generate_tm_corpus_window' subwindow.

        Parameters
        ----------
        pair: list
            List of the fields available for the dataset this widget refers
        """

        super(WidgetSimilarTopics, self).__init__()

        ########################################################################
        # Load UI
        ########################################################################
        loadUi("src/gui/uis/similarity_widget.ui", self)

        ########################################################################
        # ATTRIBUTES
        ########################################################################
        self.pair = pair
        self.df = df

        ########################################################################
        # Widgets initial configuration
        ########################################################################
        # Configure tables
        self.populate_widgets()
        #utils.configure_table_header(["tableWidget_similar_topics"], self)

    def populate_widgets(self):
        """It fills the contents of the widgets contained in the WidgetCreateTMCorpus.
        """

        corr = 100 * self.pair[2]
        corr_ = f"{corr:.2f}"  + " %"
        self.lineEdit_correlation.setText(corr_)
        self.tableWidget_similar_topics.setRowCount(2)

        df2 = self.df.loc[[self.pair[0], self.pair[1]],['Label', 'Word Description']]
        for tp in range(len(df2)):
            df3 = df2.iloc[[tp]]
            self.tableWidget_similar_topics.setItem(tp, 0, QtWidgets.QTableWidgetItem(str(self.pair[tp])))
            self.tableWidget_similar_topics.setItem(tp, 1, QtWidgets.QTableWidgetItem(df3['Label'].item()))
            self.tableWidget_similar_topics.setItem(tp, 2, QtWidgets.QTableWidgetItem(df3['Word Description'].item()))
        self.tableWidget_similar_topics.resizeColumnsToContents()

        return
