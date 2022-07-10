"""
* *IntelComp H2020 project*

Class that defines the subwindow for the Interactive Topic Model Trainer App for the edition of an existing wordlist selected by the user in the GUI's main window.

"""
from PyQt6 import QtWidgets
from PyQt6.uic import loadUi


class EditSwLstWindow(QtWidgets.QDialog):

    def __init__(self, tm, dict_wordlist):
        """
        Initializes the application's subwindow from which the user can access the functionalities for editing a new wordlist.

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        dict_wordlist: Dictionary
            Dictionary with all the information (name, word list, privacy level and description) that defines the wordlist selected by the user in the GUI's main page to be edited.
        """

        super(EditSwLstWindow, self).__init__()

        # Load UI
        # #####################################################################
        loadUi("src/gui/uis/edit_wordlist.ui", self)

        ########################################################################
        # ATTRIBUTES
        ########################################################################
        self.tm = tm
        self.dict_wordlist = dict_wordlist
        self.status = 0

        ########################################################################
        # Widgets initial configuration
        ########################################################################
        self.label_wordlist_type.setText(self.dict_wordlist['valid_for'])
        self.textEdit_wordlst.setPlainText(
            ",".join(self.dict_wordlist['wordlist']))
        self.label_wordlist_name.setText(self.dict_wordlist['name'])
        self.label_wordlist_privacy_level.setText(
            self.dict_wordlist['visibility'])
        self.textEdit_wordlst_description.setPlainText(
            self.dict_wordlist['description'])

        ########################################################################
        # Connect buttons
        ########################################################################
        self.pushButton_edit_wordlist.clicked.connect(
            self.clicked_pushButton_edit_wordlist)

    def clicked_pushButton_edit_wordlist(self):
        """It controls the clicking of the 'pushButton_edit_wordlist'. Once the button is clicked, the updates on the wordlist are taken from the 'textEdit_wordlst'. Once the wordlist new words are attained, these are updated 'dict_wordlist' describing the wordlist being edited and the task manager function in charge of editing the wordlist is invoked. After the wordlist edition completion, the window is closed.
        """

        # Get wordlist updates
        wds = self.textEdit_wordlst.toPlainText()
        wds = [el.strip() for el in wds.split(',') if len(el)]
        wds = sorted(list(set(wds)))
        self.dict_wordlist['wordlist'] = wds

        # Edit list and save replacing existing list
        self.status = self.tm.EditWdList(self.dict_wordlist)

        # Hide window
        self.hide()

        return
