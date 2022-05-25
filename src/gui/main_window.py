import numpy as np
import os
import pathlib

from PyQt6 import QtGui
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QPushButton
from PyQt6.uic import loadUi
from PyQt6.QtCore import QUrl, QThreadPool
from PyQt6.QtWebEngineWidgets import QWebEngineView
from functools import partial

from src.gui.train_model_window import TrainModelWindow
from src.gui.utils import utils
from src.gui.utils.output_wrapper import OutputWrapper
from src.gui.utils.constants import Constants
from src.gui.utils.utils import execute_in_thread
from src.project_manager.itmt_task_manager import ITMTTaskManagerGUI


class MainWindow(QMainWindow):
    def __init__(self, project_folder=None, parquet_folder=None):

        """
        Initializes the application's main window.
        
        Parameters
        ----------
        project_folder : pathlib.Path (default=None)
           Path to the application project
        parquet_folder : pathlib.Path (default=None)
           Path to the folder containing the parquet files
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
        self.tm = None
        if self.project_folder and self.parquet_folder:
            self.configure_tm()

        # Attributes for displaying PyLDAvis in home page
        self.web = None

        # Other attributes
        self.previous_page_button = None
        self.previous_corpus_button = None

        # Get home in any operating system
        self.home = str(pathlib.Path.home())

        # Creation of subwindows
        self.train_model_subwindow = TrainModelWindow()

        # Threads for executing in parallel
        self.thread_pool = QThreadPool()
        print("Multithreading with maximum"
              " %d threads" % self.thread_pool.maxThreadCount())

        #####################################################################################
        # Connect pages
        #####################################################################################
        menu_buttons = []
        for id_button in np.arange(Constants.MAX_MENU_BUTTONS):
            menu_button_name = "menu_button_" + str(id_button + 1)
            menu_button_widget = self.findChild(QPushButton, menu_button_name)
            menu_buttons.append(menu_button_widget)

        for menu_button in menu_buttons:
            menu_button.clicked.connect(partial(self.clicked_change_menu_button, menu_button))

        # PAGE 1: Home
        ###############
        self.menu_button_1.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_home))

        # PAGE 2: Corpus
        #################
        self.menu_button_2.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_corpus))

        corpus_buttons = []
        for id_button in np.arange(Constants.MAX_CORPUS_BUTTONS):
            corpus_button_name = "corpus_button_" + str(id_button + 1)
            corpus_button_widget = self.findChild(QPushButton, corpus_button_name)
            corpus_buttons.append(corpus_button_widget)

        for corpus_button in corpus_buttons:
            corpus_button.clicked.connect(partial(self.clicked_change_corpus_button, corpus_button))

        self.corpus_button_1.clicked.connect(
            lambda: self.corpus_tabs.setCurrentWidget(self.page_local_corpus))
        self.corpus_button_2.clicked.connect(
            lambda: self.corpus_tabs.setCurrentWidget(self.page_training_datasets))

        # PAGE 3: Models
        #################
        self.menu_button_3.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_models))

        # PAGE 4: Settings
        ###################
        self.menu_button_4.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_general_settings))

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        # PAGE 1: Home

        # PAGE 2: Corpus
        # Configure tables
        utils.configure_table_header(Constants.CORPUS_TABLES, self)

        # PAGE 3: Models
        # Configure tables

        # PAGE 4: Settings
        utils.configure_table_header(Constants.MODELS_TABLES, self)

        #####################################################################################
        # Connect buttons
        #####################################################################################
        self.pushButton_open_project_folder.clicked.connect(self.get_project_folder)
        self.pushButton_open_parquet_folder.clicked.connect(self.get_parquet_folder)

        self.pushButton_train_dataset.clicked.connect(self.clicked_train_dataset)

    #####################################################################################
    # TASK MANAGER COMMUNICATION METHODS
    #####################################################################################
    def configure_tm(self):
        if self.project_folder and self.parquet_folder:
            self.tm = ITMTTaskManagerGUI(self.project_folder, self.parquet_folder)

            if len(os.listdir(self.project_folder)) == 0:
                print("A new project folder was selected. Proceeding with "
                      "its configuration...")
                self.tm.create()
                self.tm.setup()
            else:
                print("An existing project folder was selected. Proceeding with "
                      "its loading...")
                self.tm.load()

            self.load_data()
            return

    def load_data(self):
        self.tm.listDownloaded(self)
        self.tm.listTMCorpus(self)
        return

    #####################################################################################
    # HANDLERS
    #####################################################################################
    def get_project_folder(self):
        self.project_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select an existing project or create a new one', self.home))
        self.lineEdit_current_project.setText(self.project_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

    def get_parquet_folder(self):
        self.parquet_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select the folder with the parquet files', self.home))
        self.lineEdit_current_parquet.setText(self.parquet_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

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
    def clicked_train_dataset(self):
        self.train_model_subwindow.exec()

    # MODELS FUNCTIONS

    def get_pyldavis_home(self):
        self.web = QWebEngineView()
        cwd = os.getcwd()
        url = QUrl.fromLocalFile(cwd + "/src/gui/resources/pyLDAvis.html")
        print(url)
        self.web.load(url)
        self.layout_plot_pyldavis.addWidget(self.web)
        self.web.show()
