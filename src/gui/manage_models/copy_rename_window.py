"""
* *IntelComp H2020 project*

Class that defines the subwindow for the Interactive Topic Model Trainer App that allows the user to select the new name for a (copied) topic model.

"""
from PyQt6 import QtWidgets
from PyQt6.uic import loadUi
from PyQt6.QtWidgets import QMessageBox
from src.gui.utils.constants import Constants

class CopyRenameWindow(QtWidgets.QDialog):

    def __init__(self, tm, type, model_name):
        """
        Initializes the application's subwindow from which the user can access the insert the new name for the model.

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        type: str
            Type of function being carried out through the subwindow
        """

        super(CopyRenameWindow, self).__init__()

        # Load UI
        # #####################################################################
        loadUi("src/gui/uis/rename_copy_model.ui", self)

        ########################################################################
        # ATTRIBUTES
        ########################################################################
        self.tm = tm
        self.type = type
        self.model_name = model_name
        self.new_name = None

        ########################################################################
        # Widgets initial configuration
        ########################################################################
        if self.type == 'copy':
            self.pushButton_rename_model.hide()
            self.pushButton_create_copy.show()
        elif self.type == 'rename':
            self.pushButton_rename_model.show()
            self.pushButton_create_copy.hide()

        ########################################################################
        # Connect buttons
        ########################################################################
        self.pushButton_rename_model.clicked.connect(
            self.clicked_pushButton_rename_model)
        self.pushButton_create_copy.clicked.connect(
            self.clicked_pushButton_create_copy)

    def clicked_pushButton_rename_model(self):
        """It controls the clicking of the 'clicked_pushButton_rename_model'. Once the button is clicked, if a correct name has been introduced, the renaming is carried out. Otherwise, a warning message is shown.
        """
        
        new_name = self.lineEdit_new_name.text()

        if new_name == "":
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.NEW_TM_NAME_NOT_SELECTED_MSG)
            return
        
        self.tm.renameTM(self.model_name, new_name, self)

        # Hide window
        self.hide()

        return
    
    def clicked_pushButton_create_copy(self):
        """It controls the clicking of the 'pushButton_create_copy'. Once the button is clicked, if a correct name has been introduced, the copying is carried out. Otherwise, a warning message is shown.
        """
        
        new_name = self.lineEdit_new_name.text()

        if new_name == "":
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.NEW_TM_NAME_NOT_SELECTED_MSG)
            return
        
        self.tm.copyTM(self.model_name, new_name, self)

        # Hide window
        self.hide()

        return
