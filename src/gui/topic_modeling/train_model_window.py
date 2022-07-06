"""
* *IntelComp H2020 project*

Class that defines the subwindow for the Interactive Topic Model Trainer App for the training of a new topic model.

"""

import configparser
import pathlib
import re
from functools import partial

import numpy as np
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QThreadPool, QUrl, pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QLabel, QMessageBox, QPushButton, QWidget
from PyQt6.uic import loadUi
from src.gui.utils import utils
from src.gui.utils.constants import Constants


class TrainModelWindow(QtWidgets.QDialog):

    def __init__(self, tm, stdout, stderr, training_corpus):
        """
        Initializes the application's subwindow from which the user can train a new topic model.

        Parameters
        ----------
        tm : TaskManager
            TaskManager object associated with the project
        stdout : OutputWrapper
            Attribute in which stdout is redirected
        stderr : OutputWrapper
            Attribute in which stderr is redirected
        training_corpus : str
            Name of the training dataset
        """

        super(TrainModelWindow, self).__init__()

        # Load UI and configure default geometry of the window
        # #####################################################################
        loadUi("src/gui/uis/train_window.ui", self)

        #####################################################################################
        # ATTRIBUTES
        #####################################################################################
        self.tm = tm
        self.stdout = stdout
        self.stderr = stderr
        self.TrDts_name = training_corpus
        self.previous_train_button = self.findChild(
            QPushButton, "train_button_1")
        self.trainer = None
        self.TrDts_name = None
        self.preproc_settings = None
        self.training_params = None 
        self.modelname = None
        self.ModelDesc = None
        self.privacy = None

        #####################################################################################
        # Widgets initial configuration
        #####################################################################################
        # Initialize progress bars
        utils.initialize_progress_bar(Constants.TRAIN_LOADING_BARS, self)

        # Configure tables
        utils.configure_table_header(Constants.TRAIN_MODEL_TABLES, self)

        # Initialize checkboxes
        self.lda_mallet_checkboxes_params = []
        for id_check in np.arange(Constants.NR_PARAMS_TRAIN_LDA_MALLET):
            checkbox_name = "checkBox_lda_" + str(id_check + 1) + "_good"
            checkbox_widget = self.findChild(QLabel, checkbox_name)
            self.lda_mallet_checkboxes_params.append(checkbox_widget)
            checkbox_name = "checkBox_lda_" + str(id_check + 1) + "_bad"
            checkbox_widget = self.findChild(QLabel, checkbox_name)
            self.lda_mallet_checkboxes_params.append(checkbox_widget)
        for checkbox in self.lda_mallet_checkboxes_params:
            checkbox.hide()

        self.prodlda_checkboxes_params = []
        for id_check in np.arange(Constants.NR_PARAMS_TRAIN_PRODLDA):
            checkbox_name = "checkBox_prod_" + str(id_check + 1) + "_good"
            checkbox_widget = self.findChild(QLabel, checkbox_name)
            self.prodlda_checkboxes_params.append(checkbox_widget)
            checkbox_name = "checkBox_prod_" + str(id_check + 1) + "_bad"
            checkbox_widget = self.findChild(QLabel, checkbox_name)
            self.prodlda_checkboxes_params.append(checkbox_widget)
        for checkbox in self.prodlda_checkboxes_params:
            checkbox.hide()

        self.ctm_checkboxes_params = []
        for id_check in np.arange(Constants.NR_PARAMS_TRAIN_CTM):
            checkbox_name = "checkBox_ctm_" + str(id_check + 1) + "_good"
            checkbox_widget = self.findChild(QLabel, checkbox_name)
            self.ctm_checkboxes_params.append(checkbox_widget)
            checkbox_name = "checkBox_ctm_" + str(id_check + 1) + "_bad"
            checkbox_widget = self.findChild(QLabel, checkbox_name)
            self.ctm_checkboxes_params.append(checkbox_widget)
        for checkbox in self.ctm_checkboxes_params:
            checkbox.hide()

        # Initialize settings
        self.get_lda_mallet_params()
        self.get_prodlda_params()
        self.get_ctm_params()
        self.get_sparklda_params()

        #####################################################################################
        # Connect buttons
        #####################################################################################
        train_buttons = []
        for id_button in np.arange(Constants.MAX_TRAIN_OPTIONS):
            train_button_name = "train_button_" + str(id_button + 1)
            train_button_widget = self.findChild(
                QPushButton, train_button_name)
            train_buttons.append(train_button_widget)

        for train_button in train_buttons:
            train_button.clicked.connect(
                partial(self.clicked_change_train_button, train_button))
        self.previous_train_button.setStyleSheet(
            Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        # PAGE 1: LDA-Mallet
        self.train_button_1.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainLDA))

        # PAGE 2: ProdLDA
        self.train_button_2.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainAVITM))

        # PAGE 3: CTM
        self.train_button_3.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainCTM))

        # PAGE 4: SparkLDA
        self.train_button_4.clicked.connect(
            lambda: self.train_tabs.setCurrentWidget(self.page_trainSparkLDA))

        #####################################################################################
        # Connect buttons
        #####################################################################################
        self.pushButton_train.clicked.connect(
            self.clicked_pushButton_train)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e., icon of the application, the application's title, and the position of the window at its opening.
        """

        self.setWindowIcon(QtGui.QIcon(
            'src/gui/resources/images/fuzzy_training.png'))
        self.setWindowTitle(Constants.SMOOTH_SPOON_TITLE)
        self.center()

    def clicked_change_train_button(self, train_button):
        """
        Method to control the selection of one of the buttons in the train bar.
        """

        # Put unpressed color for the previous pressed train button
        if self.previous_train_button:
            self.previous_train_button.setStyleSheet(
                Constants.TRAIN_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_train_button = train_button
        self.previous_train_button.setStyleSheet(
            Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        return

    # Mallet functions
    def get_lda_mallet_params(self):
        """Method that reads from the config file the default parameters for training a LDA Mallet model, and sets the corresponding values in the 'tabWidget_settingsLDA'.
        """

        # Get config object
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        # Read values and set them in the correponding widget
        self.lineEdit_nr_topics_lda.setText(
            str(cf.get('TM', 'ntopics')))
        self.lineEdit_alpha_lda.setText(
            str(cf.get('MalletTM', 'alpha')))
        self.lineEdit_optimize_interval_lda.setText(
            str(cf.get('MalletTM', 'optimize_interval')))
        self.lineEdit_nr_threads_lda.setText(
            str(cf.get('MalletTM', 'num_threads')))
        self.lineEdit_nr_iterations_lda.setText(
            str(cf.get('MalletTM', 'num_iterations')))
        self.lineEdit_doc_top_thr_lda.setText(
            str(cf.get('MalletTM', 'doc_topic_thr')))
        self.lineEdit_thetas_thr_lda.setText(
            str(cf.get('MalletTM', 'thetas_thr')))
       
        return

    def get_mallet_params_from_user(self):
        """Method that gets the parameters from the 'tabWidget_settingsLDA' that have been modified by the user and checks whether the updated value is valid for the conditions that each parameter must meet. For each of the parameters that are not configured appropriately, the information that is going to be shown to the user in a warning message is updated accordingly. Moreover, for the parameters wrong configured, a red cross is shown on the right of each of them, while those properly configured are accompanied by a red tick. Once all the parameters are checked, a warning message is shown and the method is exited in case one or more of them are wrong configured. 

        Returns
        -------
        Okay : Bool
            True if all the parameters have been properly configured
        """

        okay = True
        self.training_params = {}
        messages = ""

        if int(self.lineEdit_nr_topics_lda.text()) < 0:
            self.checkBox_lda_1_bad.show()
            messages += Constants.WRONG_NR_TOPICS_LDA_MSG + "\n"
        else:
            self.checkBox_lda_1_good.show()
            self.training_params['ntopics'] = self.lineEdit_nr_topics_lda.text()

        if int(self.lineEdit_alpha_lda.text()) < 0:
            self.checkBox_lda_2_bad.show()
            messages += Constants.WRONG_ALPHA_LDA_MSG + "\n"
        else:
            self.checkBox_lda_2_good.show()
            self.training_params['alpha'] = self.lineEdit_alpha_lda.text()

        if int(self.lineEdit_optimize_interval_lda.text()) < 0:
            self.checkBox_lda_3_bad.show()
            messages += Constants.WRONG_OI_LDA_MSG + "\n"
        else:
            self.checkBox_lda_3_good.show()
            self.training_params['optimize_interval'] = self.lineEdit_optimize_interval_lda.text()

        if int(self.lineEdit_nr_threads_lda.text()) < 0:
            self.checkBox_lda_4_bad.show()
            messages += Constants.WRONG_NR_THREADS_LDA_MSG + "\n"
        else:
            self.checkBox_lda_4_good.show()
            self.training_params['num_threads'] = self.lineEdit_nr_threads_lda.text()

        if int(self.lineEdit_nr_iterations_lda.text()) < 0:
            self.checkBox_lda_5_bad.show()
            messages += Constants.WRONG_NR_ITER_LDA_MSG + "\n"
        else:
            self.checkBox_lda_5_good.show()
            self.training_params['num_iterations'] = self.lineEdit_nr_iterations_lda.text()

        if float(self.lineEdit_doc_top_thr_lda.text()) > 1:
            self.checkBox_lda_6_bad.show()
            messages += Constants.WRONG_DOC_TPC_THR_LDA_MSG + "\n"
        else:
            self.checkBox_lda_6_good.show()
            self.training_params['doc_topic_thr'] = self.lineEdit_doc_top_thr_lda.text()

        if float(self.lineEdit_thetas_thr_lda.text()) > 1:
            self.checkBox_lda_7_bad.show()
            messages += Constants.WRONG_THETAS_THR_LDA_MSG + "\n"
        else:
            self.checkBox_lda_7_good.show()
            self.training_params['thetas_thr'] = self.lineEdit_thetas_thr_lda.text()

        if len(self.training_params.keys()) != Constants.NR_PARAMS_TRAIN_LDA_MALLET:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, messages)
            return False
        
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)
        self.training_params["mallet_path"] = cf.get('MalletTM', 'mallet_path')
        self.training_params["token_regexp"] = cf.get('MalletTM', 'token_regexp')

        # Hide checkboxes
        for checkbox in self.lda_mallet_checkboxes_params:
            checkbox.hide()
        
        return okay

    # PRODLDA
    def get_prodlda_params(self):
        """Method that reads from the config file the default parameters for training a AVITM  model, and sets the corresponding values in the 'tabWidget_settings_avitm'.
        """

        # Get config object
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        self.lineEdit_nr_topics_prod.setText(
            str(cf.get('ProdLDA', 'n_components')))
        self.comboBox_model_type_prod.setCurrentText(
            str(cf.get('ProdLDA', 'model_type')))
        self.lineEdit_hidden_prod.setText(
            str(cf.get('ProdLDA', 'hidden_sizes')))
        self.comboBox_prod_activation.setCurrentText(
            str(cf.get('ProdLDA', 'activation')))
        self.lineEdit_dropout_prod.setText(
            str(cf.get('ProdLDA', 'dropout')))
        self.comboBox_prod_learn_priors.setCurrentText(
            str(cf.get('ProdLDA', 'learn_priors')))
        self.lineEdit_lr_prod.setText(
            str(cf.get('ProdLDA', 'lr')))
        self.lineEdit_momentum_prod.setText(
            str(cf.get('ProdLDA', 'momentum')))
        self.comboBox_prod_solver.setCurrentText(
            str(cf.get('ProdLDA', 'solver')))
        self.lineEdit_nr_epochs_prod.setText(
            str(cf.get('ProdLDA', 'num_epochs')))
        self.comboBox_prod_reduce_on_plateau.setCurrentText(
            str(cf.get('ProdLDA', 'reduce_on_plateau')))
        self.lineEdit_batch_size_prod.setText(
            str(cf.get('ProdLDA', 'batch_size')))

        return

    def get_prodlda_params_from_user(self):
        """Method that gets the parameters from the 'tabWidget_settings_avitm' that have been modified by the user and checks whether the updated value is valid for the conditions that each parameter must meet. For each of the parameters that are not configured appropriately, the information that is going to be shown to the user in a warning message is updated accordingly. Moreover, for the parameters wrong configured, a red cross is shown on the right of each of them, while those properly configured are accompanied by a red tick. Once all the parameters are checked, a warning message is shown and the method is exited in case one or more of them are wrong configured; otherwise, the parameters are saved in the training configuration file. 

        Returns
        -------
        Okay : Bool
            True if all the parameters have been properly configured
        """

        okay = True
        dict_params = {}
        messages = ""

        if int(self.lineEdit_nr_topics_prod.text()) < 0:
            self.checkBox_prod_1_bad.show()
            messages += Constants.WRONG_NR_TOPIC_MSG + "\n"
        else:
            self.checkBox_prod_1_good.show()
            dict_params['num_topics'] = self.lineEdit_ctm_nr_topics.text()

        if self.comboBox_model_type_prod.currentText() not in \
                ['prodLDA', 'lda']:
            self.checkBox_prod_2_bad.show()
            messages += Constants.WRONG_UNDERLYING_MODEL_TYPE_MSG + "\n"
        else:
            self.checkBox_prod_2_good.show()
            dict_params['model_type'] = self.comboBox_model_type_prod.currentText()

        if int(self.lineEdit_nr_epochs_prod.text()) < 0:
            self.checkBox_prod_3_bad.show()
            messages += Constants.WRONG_NR_EPOCHS_MSG + "\n"
        else:
            self.checkBox_prod_3_good.show()
            dict_params['num_epochs'] = self.lineEdit_nr_epochs_prod.text()

        if int(self.lineEdit_batch_size_prod.text()) < 0:
            self.checkBox_prod_4_bad.show()
            messages += Constants.WRONG_BATCH_SIZE_MSG + "\n"
        else:
            self.checkBox_prod_4_good.show()
            dict_params['batch_size'] = self.lineEdit_batch_size_prod.text()

        if type(tuple(map(int, self.lineEdit_hidden_prod.text()[1:-1].split(',')))) != tuple:
            self.checkBox_prod_5_bad.show()
            messages += Constants.WRONG_HIDDEN_SIZES_MSG + "\n"
        else:
            self.checkBox_prod_5_good.show()
            dict_params['hidden_sizes'] = self.lineEdit_ctm_hidden_sizes.text()

        if self.comboBox_prod_activation.currentText() not in \
                ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu']:
            self.checkBox_prod_6_bad.show()
            messages += Constants.WRONG_ACTIVATION_MSG + "\n"
        else:
            self.checkBox_prod_6_good.show()
            dict_params['activation'] = self.comboBox_prod_activation.currentText()

        if float(self.lineEdit_dropout_prod.text()) < 0:
            self.checkBox_prod_7_bad.show()
            messages += Constants.WRONG_DROPOUT_MSG + "\n"
        else:
            self.checkBox_ctm_7_good.show()
            dict_params['dropout'] = self.lineEdit_dropout_prod.text()

        if self.comboBox_prod_learn_priors.currentText() != "True" and self.comboBox_prod_learn_priors.currentText() != "False":
            self.checkBox_prod_8_bad.show()
            messages += Constants.WRONG_LEARN_PRIORS_MSG + "\n"
        else:
            self.checkBox_prod_8_good.show()
            dict_params['learn_priors'] = self.comboBox_prod_learn_priors.currentText()

        if float(self.lineEdit_lr_prod.text()) < 0 or float(self.lineEdit_lr_prod.text()) > 1:
            self.checkBox_ctm_9_bad.show()
            messages += Constants.WRONG_LR_MSG + "\n"
        else:
            self.checkBox_prod_9_good.show()
            dict_params['lr'] = self.lineEdit_lr_prod.text()

        if float(self.lineEdit_momentum_prod.text()) < 0 or float(self.lineEdit_momentum_prod.text()) > 1:
            self.checkBox_prod_10_bad.show()
            messages += Constants.WRONG_MOMENTUM_MSG + "\n"
        else:
            self.checkBox_prod_10_good.show()
            dict_params['momentum'] = self.lineEdit_prod_momentum.text()

        if self.comboBox_prod_solver.currentText() not in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop']:
            self.checkBox_prod_11_bad.show()
            messages += Constants.WRONG_SOLVER_MSG + "\n"
        else:
            self.checkBox_prod_11_good.show()
            dict_params['solver'] = self.comboBox_prod_solver.currentText()

        if self.comboBox_prod_reduce_on_plateau.currentText() != "True" and self.comboBox_prod_reduce_on_plateau.currentText() != "False":
            self.checkBox_prod_12_bad.show()
            messages += Constants.WRONG_REDUCE_ON_PLATEAU_MSG + "\n"
        else:
            self.checkBox_prod_12_good.show()
            dict_params['reduce_on_plateau'] = self.comboBox_prod_reduce_on_plateau.currentText()

        if len(dict_params.keys()) != Constants.NR_PARAMS_TRAIN_PRODLDA:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, messages)
            return False

        # Hide checkboxes
        for checkbox in self.prodlda_checkboxes_params:
            checkbox.hide()

        return okay

    # CTM
    def get_ctm_params(self):
        """Method that reads from the config file the default parameters for training a CTM  model, and sets the corresponding values in the 'tabWidget_settings_ctm'.
        """

        # Get config object
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        # Read values and set them in the correponding widget
        self.comboBox_ctm_model_type.setCurrentText(
            str(cf.get('CTM', 'ctm_model_type')))
        self.lineEdit_nr_topics_ctm.setText(str(cf.get('CTM', 'num_topics')))
        self.lineEdit_nr_epochs_ctm.setText(str(cf.get('CTM', 'num_epochs')))
        self.lineEdit_batch_size_ctm.setText(str(cf.get('CTM', 'batch_size')))
        self.lineEdit_ctm_hidden_sizes.setText(cf.get('CTM', 'hidden_sizes'))
        self.comboBox_ctm_activation.setCurrentText(
            str(cf.get('CTM', 'activation')))
        self.lineEdit_ctm_dropout.setText(str(cf.get('CTM', 'dropout')))
        self.comboBox_ctm_learn_priors.setCurrentText(
            cf.get('CTM', 'learn_priors'))
        self.lineEdit_ctm_learning_rate.setText(str(cf.get('CTM', 'lr')))
        self.lineEdit_ctm_momentum.setText(str(cf.get('CTM', 'momentum')))
        self.comboBox_ctm_solver.setCurrentText(str(cf.get('CTM', 'solver')))
        self.comboBox_ctm_reduce_on_plateau.setCurrentText(
            cf.get('CTM', 'reduce_on_plateau'))
        self.lineEdit_ctm_topic_prior_mean.setText(
            str(cf.get('CTM', 'topic_prior_mean')))
        self.lineEdit_ctm_topic_prior_variance.setText(
            str(cf.get('CTM', 'topic_prior_variance')))

        return

    def get_ctm_params_from_user(self):
        """Method that gets the parameters from the 'tabWidget_settings_ctm' that have been modified by the user and checks whether the updated value is valid for the conditions that each parameter must meet. For each of the parameters that are not configured appropriately, the information that is going to be shown to the user in a warning message is updated accordingly. Moreover, for the parameters wrong configured, a red cross is shown on the right of each of them, while those properly configured are accompanied by a red tick. Once all the parameters are checked, a warning message is shown and the method is exited in case one or more of them are wrong configured; otherwise, the parameters are saved in the training configuration file. 

        Returns
        -------
        Okay : Bool
            True if all the parameters have been properly configured
        """

        okay = True
        dict_params = {}
        messages = ""
        if self.comboBox_ctm_model_type.currentText() != "CombinedTM" and "ZeroShotTM":
            self.checkBox_ctm_1_bad.show()
            messages += Constants.WRONG_MODEL_TYPE_MSG + "\n"
        else:
            self.checkBox_ctm_1_good.show()
            dict_params['ctm_model_type'] = self.comboBox_ctm_model_type.currentText()

        if int(self.lineEdit_nr_topics_ctm.text()) < 0:
            self.checkBox_ctm_2_bad.show()
            messages += Constants.WRONG_NR_TOPIC_MSG + "\n"
        else:
            self.checkBox_ctm_2_good.show()
            dict_params['num_topics'] = self.lineEdit_nr_topics_ctm.text()

        if int(self.lineEdit_nr_epochs_ctm.text()) < 0:
            self.checkBox_ctm_3_bad.show()
            messages += Constants.WRONG_NR_EPOCHS_MSG + "\n"
        else:
            self.checkBox_ctm_3_good.show()
            dict_params['num_epochs'] = self.lineEdit_nr_epochs_ctm.text()

        if int(self.lineEdit_batch_size_ctm.text()) < 0:
            self.checkBox_ctm_4_bad.show()
            messages += Constants.WRONG_BATCH_SIZE_MSG + "\n"
        else:
            self.checkBox_ctm_4_good.show()
            dict_params['batch_size'] = self.lineEdit_batch_size_ctm.text()

        if type(tuple(map(int, self.lineEdit_ctm_hidden_sizes.text()[1:-1].split(',')))) != tuple:
            self.checkBox_ctm_5_bad.show()
            messages += Constants.WRONG_HIDDEN_SIZES_MSG + "\n"
        else:
            self.checkBox_ctm_5_good.show()
            dict_params['hidden_sizes'] = self.lineEdit_ctm_hidden_sizes.text()

        if self.comboBox_ctm_activation.currentText() not in \
                ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu']:
            self.checkBox_ctm_6_bad.show()
            messages += Constants.WRONG_ACTIVATION_MSG + "\n"
        else:
            self.checkBox_ctm_6_good.show()
            dict_params['activation'] = self.comboBox_ctm_activation.currentText()

        if float(self.lineEdit_ctm_dropout.text()) < 0:
            self.checkBox_ctm_7_bad.show()
            messages += Constants.WRONG_DROPOUT_MSG + "\n"
        else:
            self.checkBox_ctm_7_good.show()
            dict_params['dropout'] = self.lineEdit_ctm_dropout.text()

        if self.comboBox_ctm_learn_priors.currentText() != "True" and self.comboBox_ctm_learn_priors.currentText() != "False":
            self.checkBox_ctm_8_bad.show()
            messages += Constants.WRONG_LEARN_PRIORS_MSG + "\n"
        else:
            self.checkBox_ctm_8_good.show()
            dict_params['learn_priors'] = self.comboBox_ctm_learn_priors.currentText()

        if float(self.lineEdit_ctm_learning_rate.text()) < 0:
            self.checkBox_ctm_9_bad.show()
            messages += Constants.WRONG_LR_MSG + "\n"
        else:
            self.checkBox_ctm_9_good.show()
            dict_params['lr'] = self.lineEdit_ctm_learning_rate.text()

        if float(self.lineEdit_ctm_momentum.text()) < 0 or float(self.lineEdit_ctm_momentum.text()) > 1:
            self.checkBox_ctm_10_bad.show()
            messages += Constants.WRONG_MOMENTUM_MSG + "\n"
        else:
            self.checkBox_ctm_10_good.show()
            dict_params['momentum'] = self.lineEdit_ctm_momentum.text()

        if self.comboBox_ctm_solver.currentText() not in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop']:
            self.checkBox_ctm_11_bad.show()
            messages += Constants.WRONG_SOLVER_MSG + "\n"
        else:
            self.checkBox_ctm_11_good.show()
            dict_params['solver'] = self.comboBox_ctm_solver.currentText()

        if self.comboBox_ctm_reduce_on_plateau.currentText() != "True" and self.comboBox_ctm_reduce_on_plateau.currentText() != "False":
            self.checkBox_ctm_12_bad.show()
            messages += Constants.WRONG_REDUCE_ON_PLATEAU_MSG + "\n"
        else:
            self.checkBox_ctm_12_good.show()
            dict_params['reduce_on_plateau'] = self.comboBox_ctm_reduce_on_plateau.currentText()

        if re.match(r'^-?\d+(?:\.\d+)$', self.lineEdit_ctm_topic_prior_mean.text()) is None:
            self.checkBox_ctm_13_bad.show()
            messages += Constants.WRONG_TOPIC_PRIOR_MEAN_MSG + "\n"
        else:
            self.checkBox_ctm_13_good.show()
            dict_params['topic_prior_mean'] = self.lineEdit_ctm_topic_prior_mean.text()

        # if re.match(r'^-?\d+(?:\.\d+)$', self.lineEdit_ctm_topic_prior_variance.text()) is None:
        #    self.checkBox_ctm_14_bad.show()
        #    messages += Constants.WRONG_TOPIC_PRIOR_VAR_MSG + "\n"
        # else:
        self.checkBox_ctm_14_good.show()
        dict_params['topic_prior_variance'] = self.lineEdit_ctm_topic_prior_variance.text()

        if len(dict_params.keys()) != Constants.NR_PARAMS_TRAIN_CTM:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, messages)
            return False

        # Hide checkboxes
        for checkbox in self.ctm_checkboxes_params:
            checkbox.hide()

        return okay

    # SPARKLDA
    def get_sparklda_params(self):
        return

    def get_sparklda_params_from_user(self):
        return
    
    def additional_training_params(self):

        # Model name
        if self.lineEdit_model_name.text() is None or self.lineEdit_model_name.text() == "":
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.NO_NAME_FOR_MODEL)
            return
        else:
            self.modelname = self.lineEdit_model_name.text()

        # Model description
        if self.textEdit_model_description.toPlainText() is None or self.textEdit_model_description.toPlainText() == "":
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.NO_DESC_FOR_MODEL)
            return
        else:
            self.ModelDesc = self.textEdit_model_description.toPlainText()

        # Privacy level
        self.privacy = self.comboBox_model_privacy_level.currentText()

        return
        

    @pyqtSlot(str)
    def append_text_train(self, text):
        """
        Method to redirect the stdout and stderr in the "text_logs_train"
        while the training of a topic model is being performed.
        """

        self.text_logs_train.moveCursor(QTextCursor.MoveOperation.End)
        self.text_logs_train.insertPlainText(text)

        return

    def execute_train_CTM(self):
        """Method to control the execution of the training of a topic model. To do so, it shows log information in the 'text_logs_train' QTextEdit and calls the corresponding TaskManager function that is in charge of training a the correspinding topic model type.
        """

        # Connect pyslots for the stdout and stderr redirection during the time the training is being performed
        self.stdout.outputWritten.connect(self.append_text_train)
        self.stderr.outputWritten.connect(self.append_text_train)

        # Train the topic model by invoking the task manager method
        self.tm.trainTM(self.trainer, self.TrDts_name, self.preproc_settings, self.training_params, self.modelname, self.ModelDesc, self.privacy)

        return

    def do_after_execute_train_CTM(self):
        """Method after the completition of a CTM's model training. Among other things, it is in charge of:
        - Making the training loading bar invisible for the user.
        - Showing messages about the completion of the training.
        - Show the topics' chemical description of the just trained model.
        """

        # Hide progress bar
        self.progress_bar_CTM.setVisible(False)

        # Show messages in the status bar and pop up window
        msg = "'The model " + self.model_name + "' was trained."
        self.statusBar().showMessage(msg, Constants.LONG_TIME_SHOW_SB)
        QtWidgets.QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, msg)

        # Show model's topics' chemical description
        self.table_training_results_CTM.clearContents()
        # self.table_training_results_CTM.setRowCount(int(num_topics))
        # self.table_training_results_CTM.setColumnCount(2)

        # for i in np.arange(0, len(list_description), 1):
        #    item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
        #    self.table_training_results_CTM.setItem(i, 0, item_topic_nr)
        #    item_topic_description = QtWidgets.QTableWidgetItem(
        #        str(list_description[i]))
        #    self.table_training_results_CTM.setItem(
        #        i, 1, item_topic_description)

        self.table_training_results_CTM.resizeRowsToContents()

        return

    def clicked_pushButton_train(self):
        """Method to control the clicking of the 'pushButton_trainCTM' which is in charge of starting the training of a CTM model. Once the buttons have been clicked, it checks whether an appropriate corpus for training has been selected, and CTM training parameters have been configured appropriately. In case both former conditions are fulfilled, the training of a CTM model is carried out in a secondary thread, while the execution of the GUI is kept in the main one.
        """

        if self.training_corpus is None:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WARNING_NO_TR_CORPUS)
            return

        if self.train_tabs.currentWidget() == self.findChild(
            QWidget, "page_trainLDA"):
            self.trainer = "mallet"
        elif self.train_tabs.currentWidget() == self.findChild(
            QWidget, "page_trainSparkLDA"):
            self.trainer = "sparkLDA"
        elif self.train_tabs.currentWidget() == self.findChild(
            QWidget, "page_trainAVITM"):
            self.trainer = "prodLDA"
        elif self.train_tabs.currentWidget() == self.findChild(
            QWidget, "page_trainCTM"):
            self.trainer = "ctm"


        okay = self.get_ctm_params_from_user()

        if okay:
            utils.execute_in_thread(
                self, self.execute_train_CTM,
                self.do_after_execute_train_CTM,
                self.progress_bar_CTM)
        else:
            return