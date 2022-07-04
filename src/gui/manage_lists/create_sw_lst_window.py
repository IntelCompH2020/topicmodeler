"""
* *IntelComp H2020 project*

Class that defines the subwindow for the Interactive Topic Model Trainer App for the creation of a new wordlist.

"""
from PyQt6 import QtGui, QtWidgets
from PyQt6.uic import loadUi
from src.gui.utils.constants import Constants


class CreateSwLstWindow(QtWidgets.QDialog):

    def __init__(self, tm):
        """
        Initializes the application's subwindow from which the user can access the functionalities for creating a new wordlist.

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        """

        super(CreateSwLstWindow, self).__init__()

        # Load UI and configure default geometry of the window
        # #####################################################################
        loadUi("src/gui/uis/create_wordlist.ui", self)

        #####################################################################################
        # ATTRIBUTES
        #####################################################################################
        self.tm = tm
        self.status = 0
        self.list_type = None

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        self.textEdit_wordlst.setPlainText("stw1,stw2, ...")

        #####################################################################################
        # Connect buttons
        #####################################################################################
        self.pushButton_create_wordlist.clicked.connect(
            self.clicked_pushButton_create_wordlist)
        self.comboBox_wordlst_type.currentIndexChanged.connect(
            self.changed_comboBox_wordlst_type)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e., icon of the application, the application's title, and the position of the window at its opening.
        """

        self.setWindowIcon(QtGui.QIcon(
            'src/gui/resources/images/fuzzy_training.png'))
        self.setWindowTitle(Constants.SMOOTH_SPOON_TITLE)
        self.center()

    def changed_comboBox_wordlst_type(self):
        """It adapts the format of the wordlist that needs to be introdued in the 'textEdit_wordlst' for the wordlist type selected in the 'comboBox_wordlst_type'.
        """

        self.list_type = self.comboBox_wordlst_type.currentText()
        if self.list_type == "Stopwords":
            msg = "stw1,stw2, ..."
        elif self.list_type == "Keywords":
            msg = "key1,key2, ..."
        elif self.list_type == "Equivalences":
            msg = "orig1: tgt1, orig2: tgt2, ..."
        else:
            return
        self.textEdit_wordlst.setPlainText(msg)
        return

    def clicked_pushButton_create_wordlist(self):
        """It controls the clicking of the 'pushButton_create_wordlist'. Once the button is clicked, the wordlist inserted by the user is taken from the 'textEdit_wordlst', as well as the wordlist characteristics (name, description, and privacy level). Once the wordlist characteristics are attained, the task manager function in charge of creating the new wordlist is invoked. After the wordlist creation completion, the window is closed.
        """

        # Get wordlist characteristics
        wds = self.textEdit_wordlst.toPlainText()
        wds = [el.strip() for el in wds.split(',') if len(el)]
        wds = sorted(list(set(wds)))
        wlst_name = self.lineEdit_wordlst_name.text()
        wlst_privacy = self.comboBox_privacy_level.currentText()
        wlst_desc = self.textEdit_wordlst_description.toPlainText()

        # Create TMCorpus
        self.status = self.tm.NewWdList(
            self.list_type, wds, wlst_name, wlst_privacy, wlst_desc)

        # Hide window
        self.hide()

        return
