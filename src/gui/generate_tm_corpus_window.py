import json

import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QLabel, QWidget
from PyQt6.uic import loadUi
from functools import partial

from src.gui.utils import utils
from src.gui.utils.constants import Constants
from src.gui.widget_create_tm_corpus import WidgetCreateTMCorpus


class GenerateTMCorpus(QtWidgets.QDialog):
    """
    @ TODO: Describe
    """

    def __init__(self, dts_ids_list, tm):
        """
        @ TODO: Describe

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        """

        super(GenerateTMCorpus, self).__init__()

        # Load UI and configure default geometry of the window
        # #####################################################################
        loadUi("src/gui/uis/createTMCorpus.ui", self)

        #####################################################################################
        # ATTRIBUTES
        #####################################################################################
        self.dts_ids_list = dts_ids_list
        self.tm = tm
        self.allDtsets = json.loads(self.tm.allDtsets)
        self.current_dts = 0
        self.current_stack = None
        self.current_stack_name = None
        self.stackedWidget_trdts_widgets = []
        self.stackedWidget_dts_name_widgets = []
        self.status = 0

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        self.initialize_stackedWidget_trdts()

        #####################################################################################
        # Connect buttons
        #####################################################################################
        self.pushButton_trdts_back.clicked.connect(self.clicked_pushButton_trdts_back)
        self.pushButton_trdts_next.clicked.connect(self.clicked_pushButton_trdts_next)
        self.pushButton_create_tm_corpus.clicked.connect(self.clicked_pushButton_create_tm_corpus)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        # @ TODO: When icons ready
        # self.setWindowIcon(QIcon('UIs/Images/dc_logo.png'))
        # self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def initialize_stackedWidget_trdts(self):
        for dts_id in self.dts_ids_list:
            Dtsets = [el for el in self.allDtsets.keys()]
            Dtset_loc = Dtsets.pop(dts_id)
            Dtset_name = self.allDtsets[Dtset_loc]['name']
            Dtset_columns = self.allDtsets[Dtset_loc]['schema']

            stack = WidgetCreateTMCorpus(Dtset_columns)
            stack.setObjectName('stack_dts_' + str(dts_id))
            self.stackedWidget_trdts.addWidget(stack)
            self.stackedWidget_trdts_widgets.append(stack)

            stack_name = QWidget()
            stack_name.setObjectName('stack_name_dts_' + str(dts_id))
            label = QLabel()
            label.setObjectName("label_name_dts_" + str(dts_id))
            label.setText(Dtset_name)
            label.setStyleSheet(Constants.Q_LABEL_EDIT_STYLESHEET)
            layout = QHBoxLayout(stack_name)
            layout.addWidget(label)
            self.stackedWidget_dts_name.addWidget(stack_name)
            self.stackedWidget_dts_name_widgets.append(stack_name)

            if dts_id == self.dts_ids_list[0]:
                self.current_stack = stack
                self.current_stack_name = stack_name
                self.update_stacks()

    def clicked_pushButton_trdts_back(self):
        if self.current_dts + 1 > 0:
            self.current_dts -= 1
            self.current_stack = self.stackedWidget_trdts_widgets[self.current_dts]
            self.current_stack_name = self.stackedWidget_dts_name_widgets[self.current_dts]
            self.update_stacks()

        return

    def clicked_pushButton_trdts_next(self):

        if self.current_dts + 1 < len(self.dts_ids_list):
            self.current_dts += 1
            self.current_stack = self.stackedWidget_trdts_widgets[self.current_dts]
            self.current_stack_name = self.stackedWidget_dts_name_widgets[self.current_dts]
            self.update_stacks()

        return

    def update_stacks(self):
        self.stackedWidget_trdts.setCurrentWidget(self.current_stack)
        self.stackedWidget_dts_name.setCurrentWidget(self.current_stack_name)

    def clicked_pushButton_create_tm_corpus(self):
        dict_to_tm_corpus = {}
        for i in range(len(self.dts_ids_list)):
            i_dts_id = self.dts_ids_list[i]
            i_widget = self.stackedWidget_trdts_widgets[i]
            identifier_field = i_widget.comboBox_trdts_id.currentText()
            fields_for_lemmas = []
            for row in range(i_widget.tableWidget_fields_to_include_lemmas.rowCount()):
                if i_widget.tableWidget_fields_to_include_lemmas.item(row, 0):
                    fields_for_lemmas.append(i_widget.tableWidget_fields_to_include_lemmas.item(row, 0).text())
            fields_for_raw = []
            for row in range(i_widget.tableWidget_fields_to_include_raw.rowCount()):
                if i_widget.tableWidget_fields_to_include_raw.item(row, 0):
                    fields_for_raw.append(i_widget.tableWidget_fields_to_include_raw.item(row, 0).text())
            filtering_condition = ""  # TODO

            i_dict = {'identifier_field': identifier_field,
                      'fields_for_lemmas': fields_for_lemmas,
                      'fields_for_raw': fields_for_raw,
                      'filtering_condition': filtering_condition
                      }
            dict_to_tm_corpus[i_dts_id] = i_dict

        dtsName = self.lineEdit_trdts_name.text()
        dtsDesc = self.textEdit_trdts_description.toPlainText()
        privacy = self.comboBox_privacy_level.currentText()

        # Create TMCorpus
        self.status = self.tm.createTMCorpus(dict_to_tm_corpus, dtsName, dtsDesc, privacy)

        # Hide window
        self.hide()

        # Remove widgets
        while self.stackedWidget_trdts.count() > 0:
            i_widget = self.stackedWidget_trdts.widget(0)
            self.stackedWidget_trdts.removeWidget(i_widget)
            i_widget.deleteLater()

        while self.stackedWidget_dts_name.count() > 0:
            i_widget = self.stackedWidget_dts_name.widget(0)
            self.stackedWidget_dts_name.removeWidget(i_widget)
            i_widget.deleteLater()







