class Constants:
    MAX_MENU_BUTTONS = 4
    MAX_TRAIN_OPTIONS = 3
    MAX_CORPUS_BUTTONS = 2

    LONG_TIME_SHOW_SB = 10000

    SMOOTH_SPOON_MSG = 'Smooth Spoon message'

    CORPUS_TABLES = ["table_available_local_corpus", "table_available_training_datasets"]
    DOWNLOAD_CORPUS_TABLES = ["table_available_fields", "table_fields_to_include", "table_filters"]
    MODELS_TABLES = ["table_available_models", "table_topics_edit_model"]
    TRAIN_MODEL_TABLES = ["table_training_resultsLDA", "table_training_results_AVITM", "table_training_results_CTM"]
    TRAIN_LOADING_BARS = ["progress_bar_LDA", "progress_bar_AVITM", "progress_bar_CTM"]

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
        }
"""