from PyQt6 import QtWidgets
from PyQt6.uic import loadUi


class CreateSwLstWindow(QtWidgets.QDialog):
    """
    @ TODO: Describe
    """

    def __init__(self, tm):
        """
        @ TODO: Describe

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
        self.pushButton_create_wordlist.clicked.connect(self.clicked_pushButton_create_wordlist)
        self.comboBox_wordlst_type.currentIndexChanged.connect(self.changed_comboBox_wordlst_type)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        # @ TODO: When icons ready
        # self.setWindowIcon(QIcon('UIs/Images/dc_logo.png'))
        # self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def changed_comboBox_wordlst_type(self):
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
        wds = self.textEdit_wordlst.toPlainText()
        wds = [el.strip() for el in wds.split(',') if len(el)]
        wds = sorted(list(set(wds)))

        wlst_name = self.lineEdit_wordlst_name.text()
        wlst_privacy = self.comboBox_privacy_level.currentText()
        wlst_desc = self.textEdit_wordlst_description.toPlainText()

        # Create TMCorpus
        self.status = self.tm.NewWdList(self.list_type, wds, wlst_name, wlst_privacy, wlst_desc)

        # Hide window
        self.hide()
