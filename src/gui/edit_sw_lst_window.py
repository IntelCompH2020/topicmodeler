from PyQt6 import QtWidgets
from PyQt6.uic import loadUi


class EditSwLstWindow(QtWidgets.QDialog):
    """
    @ TODO: Describe
    """

    def __init__(self, tm, dict_wordlist):
        """
        @ TODO: Describe

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        """

        super(EditSwLstWindow, self).__init__()

        # Load UI and configure default geometry of the window
        # #####################################################################
        loadUi("src/gui/uis/edit_wordlist.ui", self)

        #####################################################################################
        # ATTRIBUTES
        #####################################################################################
        self.tm = tm
        self.dict_wordlist = dict_wordlist
        self.status = 0
        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        self.label_wordlist_type.setText(self.dict_wordlist['valid_for'])
        self.textEdit_wordlst.setPlainText(",".join(self.dict_wordlist['wordlist']))
        self.label_wordlist_name.setText(self.dict_wordlist['name'])
        self.label_wordlist_privacy_level.setText(self.dict_wordlist['visibility'])
        self.textEdit_wordlst_description.setPlainText(self.dict_wordlist['description'])

        #####################################################################################
        # Connect buttons
        #####################################################################################
        self.pushButton_edit_wordlist.clicked.connect(self.clicked_pushButton_edit_wordlist)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e.
        icon of the application, the application's title, and the position of the window at
        its opening.
        """
        # @ TODO: When icons ready
        # self.setWindowIcon(QIcon('UIs/Images/dc_logo.png'))
        # self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def clicked_pushButton_edit_wordlist(self):
        wds = self.textEdit_wordlst.toPlainText()
        print(wds)
        wds = [el.strip() for el in wds.split(',') if len(el)]
        wds = sorted(list(set(wds)))

        self.dict_wordlist['wordlist'] = wds

        # Edit list and save replacing existing list
        self.status = self.tm.EditWdList(self.dict_wordlist)

        # Hide window
        self.hide()

        return
