class Constants:
    MAX_MENU_BUTTONS = 5
    MAX_TRAIN_OPTIONS = 3
    MAX_CORPUS_BUTTONS = 2
    MAX_RECENT_PROJECTS = 2
    MAX_RECENT_PARQUETS = 2
    MAX_RECENT_WORDLISTS = 2
    MAX_SETTINGS_BUTTONS = 5
    MAX_TM_SETTINGS_SUBBUTTONS = 6

    LONG_TIME_SHOW_SB = 10000

    SMOOTH_SPOON_TITLE = 'Smooth Spoon'
    SMOOTH_SPOON_MSG = 'Smooth Spoon message'
    TM_CORPUS_MSG_STATUS_0 = "The dataset could not be created."
    TM_CORPUS_MSG_STATUS_1 = "The dataset was created successfully."
    TM_CORPUS_MSG_STATUS_2 = "The dataset replaced an existing dataset"
    TM_DELETE_NO_CORPUS_MSG = "A corpus to be deleted must be selected first."
    WORDLIST_CREATION_MSG_STATUS_0 = "The wordlist could not be created."
    WORDLIST_CREATION_MSG_STATUS_1 = "The wordlist was created successfully."
    WORDLIST_CREATION_MSG_STATUS_2 = "The wordlist replaced an existing dataset"
    WORDLIST_EDITION_MSG_STATUS_0 = "The wordlist could not be created."
    WORDLIST_EDITION_MSG_STATUS_1 = "The wordlist was created successfully."
    WORDLIST_EDITION_MSG_STATUS_2 = "The wordlist replaced an existing dataset"
    EDIT_WORDLIST_NOT_SELECTED_MSG = "A wordlist to be edited must be selected first."
    DELETE_WORDLIST_NOT_SELECTED_MSG = "A wordlist to be deleted must be selected first."
    RESTORE_DFT_GUI_SETTINGS = "GUI settings were restored to its default value"
    UPDATE_GUI_SETTINGS = "GUI settings were to the selected values"
    RESTORE_DFT_LOG_SETTINGS = "Topic Modeling's logging settings were restored to its default value"
    UPDATE_LOG_SETTINGS = "Topic Modeling's logging settings were to the selected values"
    RESTORE_DFT_MALLET_SETTINGS = "Topic Modeling's Mallet settings were restored to its default value"
    UPDATE_MALLET_SETTINGS = "Topic Modeling's Mallet settings were to the selected values"
    RESTORE_DFT_PRODLDA_SETTINGS = "Topic Modeling's ProdLDA settings were restored to its default value"
    UPDATE_PRODLDA_SETTINGS = "Topic Modeling's ProdLDA settings were to the selected values"
    RESTORE_DFT_CTM_SETTINGS = "Topic Modeling's CTM settings were restored to its default value"
    UPDATE_CTM_SETTINGS = "Topic Modeling's CTM settings were to the selected values"
    RESTORE_DFT_TM_GENERAL_SETTINGS = "Topic Modeling's general settings were restored to its default value"
    UPDATE_TM_GENERAL_SETTINGS = "Topic Modeling's general settings were to the selected values"
    RESTORE_DFT_SPARK_SETTINGS = "Spark settings were restored to its default value"
    UPDATE_SPARK_SETTINGS = "Spark settings were to the selected values"
    RESTORE_DFT_PREPROC_SETTINGS = "Preprocessing settings were restored to its default value"
    UPDATE_PREPROC_SETTINGS = "Preprocessing settings were to the selected values"
    RESTORE_DFT_ALL_SETTINGS = "All settings were restored to its default value"
    RESTORE_DFT_MSGS = [RESTORE_DFT_GUI_SETTINGS, RESTORE_DFT_LOG_SETTINGS, RESTORE_DFT_MALLET_SETTINGS, RESTORE_DFT_PRODLDA_SETTINGS,
                        RESTORE_DFT_CTM_SETTINGS, RESTORE_DFT_TM_GENERAL_SETTINGS, RESTORE_DFT_SPARK_SETTINGS, RESTORE_DFT_PREPROC_SETTINGS]
    SETTINGS_OPTIONS = ["gui", "log", "mallet", "prodlda",
                        "ctm", "tm_general", "spark", "preproc"]

    CORPUS_TABLES = ["table_available_local_corpus",
                     "table_available_training_datasets"]
    DOWNLOAD_CORPUS_TABLES = ["table_available_fields",
                              "table_fields_to_include", "table_filters"]
    CREATE_TM_CORPUS_TABLES = ["tableWidget_available_fields_raw",
                               "tableWidget_fields_to_include_raw",
                               "tableWidget_available_fields_lemmas", "tableWidget_fields_to_include_lemmas"]

    WORDLISTS_TABLES = ["table_available_wordlists"]
    MSG_INSTRUCTIONS_NEW_WORDLIST = "To generate a new wordlists:\n - Stopwords or keywords: \
                                    Introduce the words " \
                                    "separated by commas (stw1,stw2, ...)\n - Equivalences: Introduce equivalences " \
                                    "separated by commas in the format orig:target (orig1:tgt1, orig2:tgt2, ...) "

    MODELS_TABLES = ["table_available_trained_models_desc",
                     "tableWidget_trained_models_topics"]
    TRAIN_MODEL_TABLES = ["table_training_resultsLDA",
                          "table_training_results_AVITM", "table_training_results_CTM"]
    TRAIN_LOADING_BARS = ["progress_bar_LDA",
                          "progress_bar_AVITM", "progress_bar_CTM"]

    HOME_BUTTON_SELECTED_STYLESHEET = \
        """ QPushButton {	
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
            font: 87 13pt "Avenir";
            font-weight: bold;
            color: #1A2E40;
            border-right: 4px solid #4D6F8C;
        }"""

    OTHER_BUTTONS_SELECTED_STYLESHEET = \
        """ QPushButton {	
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
            font: 87 13pt "Avenir";
            font-weight: bold;
            color: #FFFFFF;
            border-right: 4px solid #4D6F8C;
        }"""

    HOME_BUTTON_UNSELECTED_STYLESHEET = \
        """ QPushButton {	
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
            font: 87 13pt "Avenir";
            font-weight: bold;
            color: #1A2E40;
        } 
        QPushButton[Active=true] {
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            border-right: 4px solid #4D6F8C;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
        }
        QPushButton:pressed {
            border-right: 4px solid #4D6F8C;
            background-color: transparent;
        }"""

    OTHER_BUTTONS_UNSELECTED_STYLESHEET = \
        """ QPushButton {	
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
            font: 87 13pt "Avenir";
            font-weight: bold;
            color: #FFFFFF;
        } 
        QPushButton[Active=true] {
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            border-right: 4px solid #4D6F8C;
            background-color: transparent;
            text-align: left;
            padding-left: 10px;
        }
        QPushButton:pressed {
            border-right: 4px solid #4D6F8C;
            background-color: transparent;
        }"""

    TRAIN_BUTTONS_SELECTED_STYLESHEET = \
        """QPushButton {	
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            background-color: #F2F2F2;
            text-align: center;
            padding-left: 0px;
            font: 87 13pt "Avenir";
            font-weight: bold;
            color: #1A2E40;
            border-bottom: 4px solid #4D6F8C;
        }"""

    TRAIN_BUTTONS_UNSELECTED_STYLESHEET = \
        """QPushButton {	
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border: none;
            background-color: #F2F2F2;
            text-align: center;
            padding-left: 0px;
            font: 87 13pt "Avenir";
            font-weight: bold;
            color: #1A2E40;
        }

        QPushButton[Active=true] {
            background-image: ICON_REPLACE;
            background-position: left center;
            background-repeat: no-repeat;
            border-bottom: 4px solid #4D6F8C;
            background-color: #F2F2F2;
            text-align: center;
            padding-left: 0px;
        }

        QPushButton:pressed {
            border-bottom: 4px solid #4D6F8C;
            background-color: #F2F2F2;
        }"""

    Q_LABEL_EDIT_STYLESHEET = \
        """QLabel {	
            background-color: #FFFFFF;
            font: 87 italic 12pt "Avenir";
            border-radius: 5px;
            border: 1px solid #1A2E40;
            color:black;
        }"""

    SETTINGS_SUBBUTTON_SELECTED_STYLESHEET = \
        """
        QPushButton {
	        background-color: #4D6F8C;
	        font: 87 11pt "Avenir";
	        font-weight: bold;
	        color: #FFFFFF;
        }
        QPushButton:hover {
	        background-color: #4D6F8C;
        }
        QPushButton:pressed {	
	        background-color: #4D6F8C;
        }
        """

    SETTINGS_SUBBUTTON_UNSELECTED_STYLESHEET = \
        """
        QPushButton {
	        background-color: #1A2E40;
	        font: 87 11pt "Avenir";
	        font-weight: bold;
	        color: #FFFFFF;
        }
        QPushButton:hover {
	        background-color: #4D6F8C;
        }
        QPushButton:pressed {	
	        background-color: #4D6F8C;
        }
        """

    SETTINGS_SELECTED_STYLESHEET = \
        """
        QPushButton {
	        background-color: #1A2E40;
	        font: 87 11pt "Avenir";
	        font-weight: bold;
	        color: #FFFFFF;
            border-right: 4px solid #4D6F8C;
        }
        QPushButton:hover {
	        background-color: #4D6F8C;
        }
        QPushButton:pressed {	
	        background-color: #4D6F8C;
        }
        """

    SETTINGS_UNSELECTED_STYLESHEET = \
        """
        QPushButton {
	        background-color: #1A2E40;
	        font: 87 11pt "Avenir";
	        font-weight: bold;
	        color: #FFFFFF;
        }
        QPushButton:hover {
	        background-color: #4D6F8C;
        }
        QPushButton:pressed {	
	        background-color: #4D6F8C;
        }
        """
