"""
* *IntelComp H2020 project*

Class that defines the subwindow for the Interactive Topic Model Trainer App for the creation of a new topic modeling corpus from one or several of the available local dataset selected by the user in the GUI's main window.

"""
import json

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget
from PyQt6.uic import loadUi
from src.gui.utils.constants import Constants
from src.gui.manage_corpus.widget_create_tm_corpus import WidgetCreateTMCorpus


class GenerateTMCorpus(QtWidgets.QDialog):

    def __init__(self, dts_ids_list, tm):
        """
        Initializes the application's subwindow from which the user can access the functionalities for creating a new topic modeling corpus.

        Parameters
        ----------
        dts_ids_list: list
            List of ids describing the datasets that are going to be used for the generation of the new topic modeling training corpus
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
        self.pushButton_trdts_back.clicked.connect(
            self.clicked_pushButton_trdts_back)
        self.pushButton_trdts_next.clicked.connect(
            self.clicked_pushButton_trdts_next)
        self.pushButton_create_tm_corpus.clicked.connect(
            self.clicked_pushButton_create_tm_corpus)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e., icon of the application, the application's title, and the position of the window at its opening.
        """

        self.setWindowIcon(QtGui.QIcon(
            'src/gui/resources/images/fuzzy_training.png'))
        self.setWindowTitle(Constants.SMOOTH_SPOON_TITLE)
        self.center()

    def initialize_stackedWidget_trdts(self):
        """It creates as many widgets as datasets selected for the training corpus creation. At each of the widgets, a QLabel is added to specify the name of the dataset to which the widget refers to.
        """        
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
        """Method for going back to the widget referring to the previous dataset when the button 'pushButton_trdts_back' is clicked.
        """      
         
        if self.current_dts + 1 > 0:
            self.current_dts -= 1
            self.current_stack = self.stackedWidget_trdts_widgets[self.current_dts]
            self.current_stack_name = self.stackedWidget_dts_name_widgets[self.current_dts]
            self.update_stacks()

        return

    def clicked_pushButton_trdts_next(self):
        """Method for moving to the widget referring to the next dataset when the button 'pushButton_trdts_next' is clicked.
        """ 

        if self.current_dts + 1 < len(self.dts_ids_list):
            self.current_dts += 1
            self.current_stack = self.stackedWidget_trdts_widgets[self.current_dts]
            self.current_stack_name = self.stackedWidget_dts_name_widgets[self.current_dts]
            self.update_stacks()

        return

    def update_stacks(self):
        """Method for updating the content of the widget that refers to the dataset selected with the buttons 'pushButton_trdts_next' and 'pushButton_trdts_back'.
        """    

        self.stackedWidget_trdts.setCurrentWidget(self.current_stack)
        self.stackedWidget_dts_name.setCurrentWidget(self.current_stack_name)

        return

    def clicked_pushButton_create_tm_corpus(self):
        """Method for controlling the clicking of the button 'pushButton_create_tm_corpus'. It takes the information from each of the widgets referring to the datasets selected for the creation of the training corpus, and once all the data is available, it invokes the task manager function in charge of creating the TM corpus. Once it is completed, the window is closed and the widgets created for describing each of the datasets are removed.
        """  

        dict_to_tm_corpus = {}
        for i in range(len(self.dts_ids_list)):
            i_dts_id = self.dts_ids_list[i]
            i_widget = self.stackedWidget_trdts_widgets[i]
            identifier_field = i_widget.comboBox_trdts_id.currentText()
            fields_for_lemmas = []
            for row in range(i_widget.tableWidget_fields_to_include_lemmas.rowCount()):
                if i_widget.tableWidget_fields_to_include_lemmas.item(row, 0):
                    fields_for_lemmas.append(
                        i_widget.tableWidget_fields_to_include_lemmas.item(row, 0).text())
            fields_for_raw = []
            for row in range(i_widget.tableWidget_fields_to_include_raw.rowCount()):
                if i_widget.tableWidget_fields_to_include_raw.item(row, 0):
                    fields_for_raw.append(
                        i_widget.tableWidget_fields_to_include_raw.item(row, 0).text())
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
        self.status = self.tm.createTMCorpus(
            dict_to_tm_corpus, dtsName, dtsDesc, privacy)

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
        
        return
