import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton
from PyQt6.uic import loadUi
from functools import partial

from src.gui.utils import utils
from src.gui.utils.constants import Constants


class TrainModelWindow(QtWidgets.QDialog):
    """
    @ TODO: Describe
    """

    def __init__(self):
        """
        @ TODO: Describe

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        """

        super(TrainModelWindow, self).__init__()

        # Load UI and configure default geometry of the window
        # #####################################################################
        loadUi("src/gui/uis/train_window.ui", self)

        #####################################################################################
        # ATTRIBUTES
        #####################################################################################
        self.previous_train_button = None

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        # Initialize progress bars
        utils.initialize_progress_bar(Constants.TRAIN_LOADING_BARS, self)

        # Configure tables
        utils.configure_table_header(Constants.TRAIN_MODEL_TABLES, self)

        #####################################################################################
        # Connect buttons
        #####################################################################################
        train_buttons = []
        for id_button in np.arange(Constants.MAX_TRAIN_OPTIONS):
            train_button_name = "train_button_" + str(id_button + 1)
            train_button_widget = self.findChild(QPushButton, train_button_name)
            train_buttons.append(train_button_widget)

        for train_button in train_buttons:
            train_button.clicked.connect(partial(self.clicked_change_train_button, train_button))

        # PAGE 1: LDA-Mallet
        self.train_button_1.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainLDA))

        # PAGE 2: ProdLDA
        self.train_button_2.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainAVITM))

        # PAGE 3: CTM
        self.train_button_3.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainCTM))

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        # @ TODO: When icons ready
        # self.setWindowIcon(QIcon('UIs/Images/dc_logo.png'))
        # self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def clicked_change_train_button(self, train_button):
        """
        Method to control the selection of one of the buttons in the train bar.
        """

        # Put unpressed color for the previous pressed train button
        if self.previous_train_button:
            self.previous_train_button.setStyleSheet(Constants.TRAIN_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_train_button = train_button
        self.previous_train_button.setStyleSheet(Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        return
