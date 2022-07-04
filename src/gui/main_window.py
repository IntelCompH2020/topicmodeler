"""
* *IntelComp H2020 project*

Class that defines the Graphical user interface's main window for the Interactive Topic Model Trainer App
It implements the functions needed to

    - Configure the GUI widgets defined in the corresponding UI file (main_window.ui)
    - Connect the GUI and the project manager (ITMTTaskManager)
    - Implement handler functions to control the interaction between the user and the different widgets
    - Control the opening of secondary windows for specific functionalities (training, curation, etc.)
"""

import os
import pathlib
from functools import partial

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThreadPool, QUrl
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QPushButton
from PyQt6.uic import loadUi
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Local imports
from src.gui.manage_lists.create_sw_lst_window import CreateSwLstWindow
from src.gui.manage_lists.edit_sw_lst_window import EditSwLstWindow
from src.gui.manage_corpus.generate_tm_corpus_window import GenerateTMCorpus
from src.gui.topic_modeling.train_model_window import TrainModelWindow
from src.gui.utils import utils
from src.gui.utils.constants import Constants
from src.gui.utils.output_wrapper import OutputWrapper
from src.project_manager.itmt_task_manager import ITMTTaskManagerGUI


class MainWindow(QMainWindow):
    def __init__(self, project_folder=None, parquet_folder=None, wordlists_folder=None):
        """
        Initializes the application's main window.

        Parameters
        ----------
        project_folder : pathlib.Path (default=None)
           Path to the application project
        parquet_folder : pathlib.Path (default=None)
           Path to the folder containing the parquet files
        wordlists_folder : pathlib.Path (default=None)
           Path to the folder hosting the wordlists (stopwords, keywords, etc)
        """

        super(MainWindow, self).__init__()

        #####################################################################################
        # Load UI
        #####################################################################################
        loadUi("src/gui/uis/main_window.ui", self)
        self.setWindowIcon(QtGui.QIcon(
            'src/gui/resources/images/fuzzy_training.png'))
        self.setWindowTitle(Constants.SMOOTH_SPOON_TITLE)
        #####################################################################################
        # Attributes
        #####################################################################################
        # Attributes to redirect stdout and stderr
        self.stdout = OutputWrapper(self, True)
        self.stderr = OutputWrapper(self, False)

        # Attributes for creating TM object
        self.project_folder = project_folder
        self.parquet_folder = parquet_folder
        self.wordlists_folder = wordlists_folder
        self.tm = None
        if self.project_folder and self.parquet_folder and self.wordlists_folder:
            self.configure_tm()

        # Attributes for displaying PyLDAvis in home page
        self.web = None

        # Other attributes
        self.previous_page_button = self.findChild(
            QPushButton, "menu_button_1")
        self.previous_corpus_button = self.findChild(
            QPushButton, "corpus_button_1")

        # Get home in any operating system
        self.home = str(pathlib.Path.home())

        # Creation of subwindows
        self.train_model_subwindow = TrainModelWindow()
        self.create_tm_corpus_subwindow = None
        self.create_stopwords_list_subwindow = None
        self.edit_stopwords_list_subwindow = None

        # Threads for executing in parallel
        self.thread_pool = QThreadPool()
        print("Multithreading with maximum"
              " %d threads" % self.thread_pool.maxThreadCount())

        #####################################################################################
        # Connect pages
        #####################################################################################
        self.menu_buttons = []
        for id_button in np.arange(Constants.MAX_MENU_BUTTONS):
            menu_button_name = "menu_button_" + str(id_button + 1)
            menu_button_widget = self.findChild(QPushButton, menu_button_name)
            self.menu_buttons.append(menu_button_widget)

        for menu_button in self.menu_buttons:
            menu_button.clicked.connect(
                partial(self.clicked_change_menu_button, menu_button))

        # PAGE 1: Home
        ###############
        self.menu_button_1.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_home))

        self.home_recent_buttons = []
        for id_button in np.arange(Constants.MAX_RECENT_PROJECTS):
            home_recent_button_name = "pushButton_recent_project_folder_" + \
                str(id_button + 1)
            home_recent_button_widget = self.findChild(
                QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)
        for id_button in np.arange(Constants.MAX_RECENT_PARQUETS):
            home_recent_button_name = "pushButton_recent_parquet_folder_" + \
                str(id_button + 1)
            home_recent_button_widget = self.findChild(
                QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)
        for id_button in np.arange(Constants.MAX_RECENT_WORDLISTS):
            home_recent_button_name = "pushButton_recent_wordlists_folder_" + \
                str(id_button + 1)
            home_recent_button_widget = self.findChild(
                QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)

        for home_recent_button in self.home_recent_buttons:
            home_recent_button.clicked.connect(
                partial(self.get_folder_from_recent, home_recent_button))

        # PAGE 2: Corpus
        #################
        self.menu_button_2.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_corpus))

        self.corpus_buttons = []
        for id_button in np.arange(Constants.MAX_CORPUS_BUTTONS):
            corpus_button_name = "corpus_button_" + str(id_button + 1)
            corpus_button_widget = self.findChild(
                QPushButton, corpus_button_name)
            self.corpus_buttons.append(corpus_button_widget)

        for corpus_button in self.corpus_buttons:
            corpus_button.clicked.connect(
                partial(self.clicked_change_corpus_button, corpus_button))

        self.corpus_button_1.clicked.connect(
            lambda: self.corpus_tabs.setCurrentWidget(self.page_local_corpus))
        self.corpus_button_2.clicked.connect(
            lambda: self.corpus_tabs.setCurrentWidget(self.page_training_datasets))

        # PAGE 3: Wordlists
        #####################
        self.menu_button_3.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_wordlists))

        # PAGE 4: Models
        #################
        self.menu_button_4.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_models))

        # PAGE 5: Settings
        ###################
        self.menu_button_5.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_general_settings))

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################

        # MENU BUTTONS
        # When the app is first opened, menu buttons are disabled until the user selects properly the project and
        # parquet folders
        for menu_button in self.menu_buttons:
            menu_button.setEnabled(False)

        # PAGE 1: Home
        utils.set_recent_buttons(self)
        self.previous_page_button.setStyleSheet(
            Constants.HOME_BUTTON_SELECTED_STYLESHEET)

        # PAGE 2: Corpus
        # Configure tables
        utils.configure_table_header(Constants.CORPUS_TABLES, self)
        self.previous_corpus_button.setStyleSheet(
            Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        # PAGE 3: Wordlists
        # Configure tables
        utils.configure_table_header(Constants.WORDLISTS_TABLES, self)

        # PAGE 4: Models
        # Configure tables

        # PAGE 5: Settings
        utils.configure_table_header(Constants.MODELS_TABLES, self)

        #####################################################################################
        # Connect buttons
        #####################################################################################
        self.pushButton_open_project_folder.clicked.connect(
            self.get_project_folder)
        self.pushButton_open_parquet_folder.clicked.connect(
            self.get_parquet_folder)
        self.pushButton_open_wordlists_folder.clicked.connect(
            self.get_wordlists_folder)

        self.pushButton_generate_training_dataset.clicked.connect(
            self.clicked_pushButton_generate_training_dataset)
        self.pushButton_train_trdtst.clicked.connect(
            self.clicked_train_dataset)
        self.pushButton_delete_trdtst.clicked.connect(
            self.clicked_delete_dataset)

        self.pushButton_create_wordlist.clicked.connect(
            self.clicked_pushButton_create_wordlist)
        self.pushButton_edit_wordlist.clicked.connect(
            self.clicked_pushButton_edit_wordlist)
        self.pushButton_delete_wordlist.clicked.connect(
            self.clicked_pushButton_delete_wordlist)

        self.treeView_trained_models.clicked.connect(
            self.clicked_treeView_trained_models)
        self.pushButton_models_pyldavis.clicked.connect(
            self.clicked_pushButton_models_pyldavis)
        self.pushButton_return_pyldavis.clicked.connect(
            self.clicked_pushButton_return_pyldavis)
        self.pushButton_train_submodel.clicked.connect(
            self.clicked_pushButton_train_submodel)
        self.pushButton_delete_model.clicked.connect(
            self.clicked_pushButton_delete_model)

    #####################################################################################
    # TASK MANAGER COMMUNICATION METHODS
    #####################################################################################
    def configure_tm(self):
        """
        Once proper project and parquet folders have been selected by the user, it instantiates a task manager object and its corresponding configuration functions are invoked according to whether the project selected already existed or was just created.After this, the selected folders are saved in the dictionary of recent folders and the menu buttons are unlocked so the user can proceed with the interaction with the GUI.
        """

        if self.project_folder and self.parquet_folder and self.wordlists_folder:
            # A ITMTTaskManagerGUI is instantiated and configured according to whether the selected project is a new or
            # an already utilized one
            self.tm = ITMTTaskManagerGUI(
                self.project_folder, self.parquet_folder, self.wordlists_folder)
            if len(os.listdir(self.project_folder)) == 0:
                print("A new project folder was selected. Proceeding with "
                      "its configuration...")
                self.tm.create()
                # self.tm.setup()
            else:
                print("An existing project folder was selected. Proceeding with "
                      "its loading...")
            self.tm.load()
            # Project and parquet folders are saved in the dict of recent folders to future user-gui interactions
            utils.save_recent(self.project_folder,
                              self.parquet_folder, self.wordlists_folder)
            self.init_user_interaction()

            return

    def load_data(self):
        """
        It loads the data and its associated metadata (local datasets available in the parquet folder, corpus for training and trained models) into the corresponding GUI's tables.
        """

        # Load datasets available in the parquet folder into "table_available_local_corpus"
        self.tm.listDownloaded(self)
        # Add checkboxes in the last column of "table_available_local_corpus" so the user can select from which of the
        # datasets he wants to create a training corpus
        utils.add_checkboxes_to_table(self.table_available_local_corpus)

        # Load available training corpus (if any) into "table_available_training_datasets"
        self.tm.listTMCorpus(self)

        # Update the style of the tables in the corpus page
        utils.configure_table_header(Constants.CORPUS_TABLES, self)

        # Load available wordlists (if any) into "table_available_wordlists"
        self.tm.listAllWdLists(self)

        return

    def init_user_interaction(self):
        """
        Unlocks the clicking of the menu buttons so the user can proceed with the interaction with the GUI different from the selection of the project and parquet folders.
        """

        # Once project and parquet folders are properly selected, app's functionalities are enabled
        for menu_button in self.menu_buttons:
            menu_button.setEnabled(True)

        # Already available data is visualized in the corresponding tables
        self.load_data()

        return

    #####################################################################################
    # HANDLERS
    #####################################################################################
    def get_project_folder(self):
        """
        Method to control the clicking of the button "pushButton_open_project_folder. When this button is clicked, the folder selector of the user's OS is open so the user can select the project folder. Once the project is selected, if a proper parquet and wordlists folders were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.project_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select an existing project or create a new one', self.home))
        self.lineEdit_current_project.setText(self.project_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_parquet_folder(self):
        """
        Method to control the clicking of the button "pushButton_open_parquet_folder. When this button is clicked, the folder selector of the user's OS is open so the user can select the parquet folder. Once the parquet folder is selected, if a proper project and wordlists folders were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.parquet_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select the folder with the parquet files', self.home))
        self.lineEdit_current_parquet.setText(self.parquet_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_wordlists_folder(self):
        """
        Method to control the clicking of the button "pushButton_open_wordlists_folder. When this button is clicked, the folder selector of the user's OS is open so the user can select the folder hosting the wordlists (stopwords, keywords, etc). Once the wordlists folder is selected, if a proper project and parquet folders were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.wordlists_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select the folder hosting the wordlists (stopwords, keywords, etc)', self.home))
        self.lineEdit_current_wordlist.setText(
            self.wordlists_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_folder_from_recent(self, recent_button):
        """
        Method to control the clicking of one of the recent buttons. If the recent button relates to a project folder, such a project folder is loaded directly as the selected project folder and updated in the corresponding line edit. Otherwise, the same applies but for the parquet folder. If both proper project and parquet folders have been selected, the GUI's associated Task Manager object is configured.
        """

        if "project" in recent_button.objectName():
            self.project_folder = pathlib.Path(recent_button.text())
            self.lineEdit_current_project.setText(recent_button.text())
        elif "parquet" in recent_button.objectName():
            self.parquet_folder = pathlib.Path(recent_button.text())
            self.lineEdit_current_parquet.setText(recent_button.text())
        elif "wordlists" in recent_button.objectName():
            self.wordlists_folder = pathlib.Path(recent_button.text())
            self.lineEdit_current_wordlist.setText(recent_button.text())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def clicked_change_menu_button(self, menu_button):
        """
        Method to control the selection of one of the buttons in the menu bar.
        """

        # Put unpressed color for the previous pressed menu button
        if self.previous_page_button:
            if self.previous_page_button.objectName() == "menu_button_1":
                self.previous_page_button.setStyleSheet(
                    Constants.HOME_BUTTON_UNSELECTED_STYLESHEET)
            else:
                self.previous_page_button.setStyleSheet(
                    Constants.OTHER_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_page_button = menu_button
        if self.previous_page_button.objectName() == "menu_button_1":
            self.previous_page_button.setStyleSheet(
                Constants.HOME_BUTTON_SELECTED_STYLESHEET)
        else:
            self.previous_page_button.setStyleSheet(
                Constants.OTHER_BUTTONS_SELECTED_STYLESHEET)
        return

    def clicked_change_corpus_button(self, corpus_button):
        """
        Method to control the selection of one of the buttons in the train bar.
        """

        # Put unpressed color for the previous pressed train button
        if self.previous_corpus_button:
            self.previous_corpus_button.setStyleSheet(
                Constants.TRAIN_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_corpus_button = corpus_button
        self.previous_corpus_button.setStyleSheet(
            Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        return

    # CORPUS FUNCTIONS
    def clicked_pushButton_generate_training_dataset(self):
        """Method for controlling the clicking of the 'pushButton_generate_training_dataset' Once this button is clicked, the datasets to be used for the creation of a new topic modeling corpus are collected by checking which of the checkboxes at the last column of the available local datasets are selected, and a list is created with its corresponding ids. Such a list is utilized for creating a "GenerateTMCorpus" subwindow, from which the user can specify his desired characteristics for the training corpus to be created. Once the former subwindow is closed, the training corpus generation is completed and based on the status returned by the task manager function in charge of the creation, an informative or warning message is shown to the user.
        """

        # Get ids of the datasets that are going to be used for the training corpus generation
        checked_list = []
        for i in range(self.table_available_local_corpus.rowCount()):
            item = self.table_available_local_corpus.item(
                i, self.table_available_local_corpus.columnCount() - 1)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked_list.append(i)

        self.create_tm_corpus_subwindow = GenerateTMCorpus(
            checked_list, self.tm)
        self.create_tm_corpus_subwindow.exec()

        # Update data in table
        self.load_data()

        # Show information message about the TM corpus creation completion
        if self.create_tm_corpus_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.TM_CORPUS_MSG_STATUS_0)
        elif self.create_tm_corpus_subwindow.status == 1:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_1)
        elif self.create_tm_corpus_subwindow.status == 2:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_2)

        return

    def clicked_delete_dataset(self):
        """It controls the clicking of the 'pushButton_delete_trdtst' button. When such a button is selected, it is first checked whether the user has chosen a corpus to be deleted. In the affirmative case, the TM function in charge of the deletion of training corpus is invoked.
        """

        # Get selected dataset for deletion
        r = self.table_available_training_datasets.currentRow()
        print(self.table_available_training_datasets.item(
            r, 0).text())

        # If no training corpus is selected for deletion before clicking the 'pushButton_delete_trdtst' button,
        # a warning message is shown to the user
        if r is None:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.TM_DELETE_NO_CORPUS_MSG)
            return
        # Get name of the corpus to be deleted
        corpus_to_delete = self.table_available_training_datasets.item(
            r, 0).text()

        # Delete corpus by invoking the corresponding task manager function
        self.tm.deleteTMCorpus(corpus_to_delete, self)

        # Reload the listing of the training corpora
        self.tm.listTMCorpus(self)

        return

    def clicked_pushButton_create_wordlist(self):
        """It controls the clicking of the 'pushButton_create_wordlist' button. When such a button is clicked, a message informing about the format for each wordlist type is shown to the user. Immediately after, a new window appears from which the user can insert the wordlist characteristics. Once this window is closed and the wordlist creation is completed, an informative message is shown specifying whether the creation was successful or not.
        """

        # Show instruction for creation of wordlists
        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.MSG_INSTRUCTIONS_NEW_WORDLIST)

        # Invoke window
        self.create_stopwords_list_subwindow = CreateSwLstWindow(self.tm)
        self.create_stopwords_list_subwindow.exec()

        # Update data in table
        self.tm.listAllWdLists(self)

        # Show information message about the TM corpus creation completion
        if self.create_stopwords_list_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.WORDLIST_CREATION_MSG_STATUS_0)
        elif self.create_stopwords_list_subwindow.status == 1:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_CREATION_MSG_STATUS_1)
        elif self.create_stopwords_list_subwindow.status == 2:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_CREATION_MSG_STATUS_2)

        return

    def clicked_pushButton_edit_wordlist(self):
        """It controls the clicking of the 'pushButton_edit_wordlist' button. When such a button is clicked, a new window appears from which the user can edit the selected wordlist. Once this window is closed and the wordlist edition is completed, an informative message is shown specifying whether the edition was successful or not.
        """

        # Get wordlist selected for edition
        r = self.table_available_wordlists.currentRow()

        # If no wordlist for edition is selected  before clicking the 'pushButton_edit_wordlist' button, a warning message is shown to the user
        if not r:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.EDIT_WORDLIST_NOT_SELECTED_MSG)
            return

        wlst_to_edit = self.table_available_wordlists.item(r, 0).text()
        wdList_info = self.tm.get_wdlist_info(wlst_to_edit)
        self.edit_stopwords_list_subwindow = EditSwLstWindow(
            self.tm, wdList_info)
        self.edit_stopwords_list_subwindow.exec()

        # Update data in table
        self.tm.listAllWdLists(self)

        # Show information message about the TM corpus edition completion
        if self.edit_stopwords_list_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.WORDLIST_EDITION_MSG_STATUS_0)
        elif self.edit_stopwords_list_subwindow.status == 1:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_EDITION_MSG_STATUS_1)
        elif self.edit_stopwords_list_subwindow.status == 2:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_EDITION_MSG_STATUS_2)

        return

    def clicked_pushButton_delete_wordlist(self):
        """It controls the clicking of the 'pushButton_delete_wordlist' button. When such a button is selected, it is first checked whether the user has chosen a wordlist to be deleted. In the affirmative case, the TM function in charge of the deletion of a wordlist is invoked.
        """

        # Get wordlist selected for deletion
        r = self.table_available_wordlists.currentRow()

        # If no training corpus is selected for deletion before clicking the 'pushButton_delete_wordlist' button, a warning message is shown to the user
        if not r:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.DELETE_WORDLIST_NOT_SELECTED_MSG)
            return

        wlst_to_delete = self.table_available_wordlists.item(r, 0).text()
        self.tm.DelWdList(wlst_to_delete, self)

        # Update data in wordlists table
        self.tm.listAllWdLists(self)

        return

    def clicked_train_dataset(self):
        self.train_model_subwindow.exec()

        return

    # MODELS FUNCTIONS
    def clicked_treeView_trained_models(self):
        """Method to control the clicking of an item within the QTreeWidget 'treeView_trained_models' in the 'Models' tab. At the time one of the items in the QTreeWidget is selected, the information of the model associated with the clicked item, as well as its topics' chemical description is shown in the tables 'table_available_trained_models_desc' and 'tableWidget_trained_models_topics', respectively.
        """

        if self.treeView_trained_models.currentItem() is None or \
                self.treeView_available_models_topic_desc.currentItem().text(0).lower().startswith("models"):
            return
        else:
            model_selected = self.treeView_trained_models.currentItem().text(0)

            for model in self.tm.models.keys():
                if self.tm.models[model]['name'] == model_selected:
                    self.table_available_trained_models_desc.setRowCount(1)
                    self.table_available_trained_models_desc.setItem(0, 0, QtWidgets.QTableWidgetItem(
                        self.tm.models[model]['name']))
                    self.table_available_trained_models_desc.setItem(0, 1, QtWidgets.QTableWidgetItem(
                        self.tm.models[model]['model_type']))
                    self.table_available_trained_models_desc.setItem(0, 2, QtWidgets.QTableWidgetItem(
                        self.tm.models[model]['hierarchy_level']))
                    self.table_available_trained_models_desc.setItem(0, 3, QtWidgets.QTableWidgetItem(
                        self.tm.models[model]['hierarchical_type']))
                    self.table_available_trained_models_desc.setItem(0, 4, QtWidgets.QTableWidgetItem(
                        self.tm.models[model]['creation_date']))

            # Show topics in table
            self.tableWidget_trained_models_topics.clearContents()
            self.tableWidget_trained_models_topics.setRowCount(
                int(model.num_topics))
            self.tableWidget_trained_models_topics.setColumnCount(2)

            list_description = []
            for i in np.arange(0, len(model.topics_models), 1):
                if str(type(model.topics_models[i])) == "<class 'src.htms.topic.Topic'>":
                    description = ' '.join(
                        str(x) for x in model.topics_models[i].description)
                    list_description.append(description)
            for i in np.arange(0, len(list_description), 1):
                item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
                self.tableWidget_trained_models_topics.setItem(
                    i, 0, item_topic_nr)
                item_topic_description = QtWidgets.QTableWidgetItem(
                    str(list_description[i]))
                self.tableWidget_trained_models_topics.setItem(
                    i, 1, item_topic_description)

            # Show PyLDAvis
            if self.web:
                self.web.setParent(None)
            self.web = QWebEngineView()
            # self.web.setZoomFactor(0.25)
            url = QUrl.fromLocalFile(pathlib.Path(
                model.model_path, "pyLDAvis.html").as_posix())
            self.web.load(url)
            self.layout_plot_pyldavis.addWidget(self.web)
            self.web.show()

        return

    def clicked_pushButton_models_pyldavis(self):
        """Method to control the change within the Models' page to the PyLDAvis view. 
        """

        self.models_tabs.setCurrentWidget(self.page_models_pyldavis)

        return

    def clicked_pushButton_return_pyldavis(self):
        """Method to control the change from the PyLDAvis view back to the Models' page main view. 
        """

        self.models_tabs.setCurrentWidget(self.page_models_main)

        return

    def clicked_pushButton_train_submodel(self):
        """Method to control the clicking of the button 'pushButton_train_submodel', which is in charge of starting the training of a submodel.
        """

        return

    def clicked_pushButton_delete_model(self):
        """Method to control the clicking of the button 'pushButton_train_submodel', which is in charge of starting the deletion of a model.
        """

        return
