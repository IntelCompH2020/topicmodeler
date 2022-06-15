"""
* *IntelComp H2020 project*

Class that defines the Graphical user interface's main window for the Interactive Topic Model Trainer App
It implements the functions needed to

    - Configure the GUI widgets defined in the corresponding UI file (main_window.ui)
    - Connect the GUI and the project manager (ITMTTaskManager)
    - Implement handler functions to control the interaction between the user and the different widgets
    - Control the opening of secondary windows for specific functionalities (training, curation, etc.)
"""

import numpy as np
import os
import pathlib

from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QPushButton, QMessageBox
from PyQt6.uic import loadUi
from PyQt6.QtCore import QUrl, QThreadPool
from PyQt6.QtWebEngineWidgets import QWebEngineView
from functools import partial

# Local imports
from src.gui.create_sw_lst_window import CreateSwLstWindow
from src.gui.edit_sw_lst_window import EditSwLstWindow
from src.gui.generate_tm_corpus_window import GenerateTMCorpus
from src.gui.train_model_window import TrainModelWindow
from src.gui.utils import utils
from src.gui.utils.output_wrapper import OutputWrapper
from src.gui.utils.constants import Constants
from src.gui.utils.utils import execute_in_thread
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
        self.previous_page_button = self.findChild(QPushButton, "menu_button_1")
        self.previous_corpus_button = self.findChild(QPushButton, "corpus_button_1")

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
            menu_button.clicked.connect(partial(self.clicked_change_menu_button, menu_button))

        # PAGE 1: Home
        ###############
        self.menu_button_1.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_home))

        self.home_recent_buttons = []
        for id_button in np.arange(Constants.MAX_RECENT_PROJECTS):
            home_recent_button_name = "pushButton_recent_project_folder_" + str(id_button + 1)
            home_recent_button_widget = self.findChild(QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)
        for id_button in np.arange(Constants.MAX_RECENT_PARQUETS):
            home_recent_button_name = "pushButton_recent_parquet_folder_" + str(id_button + 1)
            home_recent_button_widget = self.findChild(QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)
        for id_button in np.arange(Constants.MAX_RECENT_WORDLISTS):
            home_recent_button_name = "pushButton_recent_wordlists_folder_" + str(id_button + 1)
            home_recent_button_widget = self.findChild(QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)

        for home_recent_button in self.home_recent_buttons:
            home_recent_button.clicked.connect(partial(self.get_folder_from_recent, home_recent_button))

        # PAGE 2: Corpus
        #################
        self.menu_button_2.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_corpus))

        self.corpus_buttons = []
        for id_button in np.arange(Constants.MAX_CORPUS_BUTTONS):
            corpus_button_name = "corpus_button_" + str(id_button + 1)
            corpus_button_widget = self.findChild(QPushButton, corpus_button_name)
            self.corpus_buttons.append(corpus_button_widget)

        for corpus_button in self.corpus_buttons:
            corpus_button.clicked.connect(partial(self.clicked_change_corpus_button, corpus_button))

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
        self.previous_page_button.setStyleSheet(Constants.HOME_BUTTON_SELECTED_STYLESHEET)

        # PAGE 2: Corpus
        # Configure tables
        utils.configure_table_header(Constants.CORPUS_TABLES, self)
        self.previous_corpus_button.setStyleSheet(Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

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
        self.pushButton_open_project_folder.clicked.connect(self.get_project_folder)
        self.pushButton_open_parquet_folder.clicked.connect(self.get_parquet_folder)
        self.pushButton_open_wordlists_folder.clicked.connect(self.get_wordlists_folder)

        self.pushButton_generate_training_dataset.clicked.connect(self.clicked_pushButton_generate_training_dataset)
        self.pushButton_train_trdtst.clicked.connect(self.clicked_train_dataset)
        self.pushButton_delete_trdtst.clicked.connect(self.clicked_delete_dataset)

        self.pushButton_create_wordlist.clicked.connect(self.clicked_pushButton_create_wordlist)
        self.pushButton_edit_wordlist.clicked.connect(self.clicked_pushButton_edit_wordlist)
        self.pushButton_delete_wordlist.clicked.connect(self.clicked_pushButton_delete_wordlist)

    #####################################################################################
    # TASK MANAGER COMMUNICATION METHODS
    #####################################################################################
    def configure_tm(self):
        """
        Once proper project and parquet folders have been selected by the user, it instantiates a task manager object
        and its corresponding configuration functions are invoked according to whether the project selected already
        existed or was just created. After this, the selected folders are saved in the dictionary of recent folders
        and the menu buttons are unlocked so the user can proceed with the interaction with the GUI.
        """

        if self.project_folder and self.parquet_folder and self.wordlists_folder:
            # A ITMTTaskManagerGUI is instantiated and configured according to whether the selected project is a new or
            # an already utilized one
            self.tm = ITMTTaskManagerGUI(self.project_folder, self.parquet_folder, self.wordlists_folder)
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
            utils.save_recent(self.project_folder, self.parquet_folder, self.wordlists_folder)
            self.init_user_interaction()

            return

    def load_data(self):
        """
        It loads the data and its associated metadata (local datasets available in the parquet folder, corpus for
        training and trained models) into the corresponding GUI's tables.
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
        Unlocks the clicking of the menu buttons so the user can proceed with the interaction with the GUI different
        from the selection of the project and parquet folders.
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
        Method to control the clicking of the button "pushButton_open_project_folder. When this button is clicked,
        the folder selector of the user's OS is open so the user can select the project folder. Once the project is
        selected, if a proper parquet and wordlists folders were also already selected, the GUI's associated Task
        Manager object is configured.
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
        Method to control the clicking of the button "pushButton_open_parquet_folder. When this button is clicked,
        the folder selector of the user's OS is open so the user can select the parquet folder. Once the parquet
        folder is selected, if a proper project and wordlists folders were also already selected, the GUI's associated
        Task Manager object is configured.
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
        Method to control the clicking of the button "pushButton_open_wordlists_folder. When this button is clicked,
        the folder selector of the user's OS is open so the user can select the folder hosting the wordlists
        (stopwords, keywords, etc). Once the wordlists folder is selected, if a proper project and parquet folders
        were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.wordlists_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select the folder hosting the wordlists (stopwords, keywords, etc)', self.home))
        self.lineEdit_current_wordlist.setText(self.wordlists_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_folder_from_recent(self, recent_button):
        """
        Method to control the clicking of one of the recent buttons. If the recent button relates to a project
        folder, such a project folder is loaded directly as the selected project folder and updated in the
        corresponding line edit. Otherwise, the same applies but for the parquet folder. If both proper project and
        parquet folders have been selected, the GUI's associated Task Manager object is configured.
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
                self.previous_page_button.setStyleSheet(Constants.HOME_BUTTON_UNSELECTED_STYLESHEET)
            else:
                self.previous_page_button.setStyleSheet(Constants.OTHER_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_page_button = menu_button
        if self.previous_page_button.objectName() == "menu_button_1":
            self.previous_page_button.setStyleSheet(Constants.HOME_BUTTON_SELECTED_STYLESHEET)
        else:
            self.previous_page_button.setStyleSheet(Constants.OTHER_BUTTONS_SELECTED_STYLESHEET)
        return

    def clicked_change_corpus_button(self, corpus_button):
        """
        Method to control the selection of one of the buttons in the train bar.
        """

        # Put unpressed color for the previous pressed train button
        if self.previous_corpus_button:
            self.previous_corpus_button.setStyleSheet(Constants.TRAIN_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_corpus_button = corpus_button
        self.previous_corpus_button.setStyleSheet(Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        return

    # CORPUS FUNCTIONS
    def clicked_pushButton_generate_training_dataset(self):
        checked_list = []
        for i in range(self.table_available_local_corpus.rowCount()):
            item = self.table_available_local_corpus.item(i, self.table_available_local_corpus.columnCount() - 1)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked_list.append(i)

        self.create_tm_corpus_subwindow = GenerateTMCorpus(checked_list, self.tm)
        self.create_tm_corpus_subwindow.exec()

        # Update data in table
        self.load_data()

        # Show information message about the TM corpus creation completion
        # @ TODO: Add this to TM
        if self.create_tm_corpus_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_0)
        elif self.create_tm_corpus_subwindow.status == 1:
            QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_1)
        elif self.create_tm_corpus_subwindow.status == 2:
            QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_2)

        return

    def clicked_delete_dataset(self):
        r = self.table_available_training_datasets.currentRow()
        corpus_to_delete = self.table_available_training_datasets.item(r, 0).text()
        self.tm.deleteTMCorpus(corpus_to_delete, self)

        self.tm.listTMCorpus(self)

        return

    def clicked_pushButton_create_wordlist(self):
        QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.MSG_INSTRUCTIONS_NEW_WORDLIST)

        # Invoke window
        self.create_stopwords_list_subwindow = CreateSwLstWindow(self.tm)
        self.create_stopwords_list_subwindow.exec()

        # Update data in table
        self.tm.listAllWdLists(self)

        # Show information message about the TM corpus creation completion
        # @ TODO: Add this to TM
        if self.create_stopwords_list_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_0)
        elif self.create_stopwords_list_subwindow.status == 1:
            QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_1)
        elif self.create_stopwords_list_subwindow.status == 2:
            QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_2)
        return

    def clicked_pushButton_edit_wordlist(self):

        # Get wordlist selected for edition
        r = self.table_available_wordlists.currentRow()
        wlst_to_edit = self.table_available_wordlists.item(r, 0).text()

        wdList_info = self.tm.get_wdlist_info(wlst_to_edit)
        self.edit_stopwords_list_subwindow = EditSwLstWindow(self.tm,wdList_info)
        self.edit_stopwords_list_subwindow.exec()

        # Update data in table
        self.tm.listAllWdLists(self)

        # Show information message about the TM corpus edition completion
        # @ TODO: Add this to TM
        if self.edit_stopwords_list_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_0)
        elif self.edit_stopwords_list_subwindow.status == 1:
            QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_1)
        elif self.edit_stopwords_list_subwindow.status == 2:
            QMessageBox.information(self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_2)

        return

    def clicked_pushButton_delete_wordlist(self):
        r = self.table_available_wordlists.currentRow()
        wlst_to_delete = self.table_available_wordlists.item(r, 0).text()
        self.tm.DelWdList(wlst_to_delete, self)

        self.tm.listAllWdLists(self)
        return

    def clicked_train_dataset(self):
        self.train_model_subwindow.exec()

        return

    # MODELS FUNCTIONS

    def get_pyldavis_home(self):
        self.web = QWebEngineView()
        cwd = os.getcwd()
        url = QUrl.fromLocalFile(cwd + "/src/gui/resources/pyLDAvis.html")
        print(url)
        self.web.load(url)
        self.layout_plot_pyldavis.addWidget(self.web)
        self.web.show()
