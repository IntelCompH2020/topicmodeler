class Constants:

    # GUI constants
    SMOOTH_SPOON_TITLE = 'Smooth Spoon'
    SMOOTH_SPOON_MSG = 'Smooth Spoon message'
    LONG_TIME_SHOW_SB = 10000

    # Buttons-related constants
    MAX_MENU_BUTTONS = 5
    MAX_TRAIN_OPTIONS = 4
    MAX_CORPUS_BUTTONS = 2
    MAX_RECENT_PROJECTS = 2
    MAX_RECENT_PARQUETS = 2
    MAX_RECENT_WORDLISTS = 2
    MAX_SETTINGS_BUTTONS = 5
    MAX_TM_SETTINGS_SUBBUTTONS = 6

    # Corpus management-related constants
    TM_CORPUS_MSG_STATUS_0 = "The dataset could not be created."
    TM_CORPUS_MSG_STATUS_1 = "The dataset was created successfully."
    TM_CORPUS_MSG_STATUS_2 = "The dataset replaced an existing dataset"
    TM_DELETE_NO_CORPUS_MSG = "A corpus to be deleted must be selected first."

    CORPUS_TABLES = ["table_available_local_corpus",
                     "table_available_training_datasets"]
    DOWNLOAD_CORPUS_TABLES = ["table_available_fields",
                              "table_fields_to_include", "table_filters"]
    CREATE_TM_CORPUS_TABLES = ["tableWidget_available_fields_raw",
                               "tableWidget_fields_to_include_raw",
                               "tableWidget_available_fields_lemmas", "tableWidget_fields_to_include_lemmas"]

    # Wordlists management-related constants
    WORDLIST_CREATION_MSG_STATUS_0 = "The wordlist could not be created."
    WORDLIST_CREATION_MSG_STATUS_1 = "The wordlist was created successfully."
    WORDLIST_CREATION_MSG_STATUS_2 = "The wordlist replaced an existing dataset"
    WORDLIST_EDITION_MSG_STATUS_0 = "The wordlist could not be created."
    WORDLIST_EDITION_MSG_STATUS_1 = "The wordlist was created successfully."
    WORDLIST_EDITION_MSG_STATUS_2 = "The wordlist replaced an existing dataset"
    EDIT_WORDLIST_NOT_SELECTED_MSG = "A wordlist to be edited must be selected first."
    DELETE_WORDLIST_NOT_SELECTED_MSG = "A wordlist to be deleted must be selected first."

    WORDLISTS_TABLES = ["table_available_wordlists"]
    MSG_INSTRUCTIONS_NEW_WORDLIST = "To generate a new wordlists:\n - Stopwords or keywords: \
                                    Introduce the words " \
                                    "separated by commas (stw1,stw2, ...)\n - Equivalences: Introduce equivalences " \
                                    "separated by commas in the format orig:target (orig1:tgt1, orig2:tgt2, ...) "

    # Models management-related constants
    MODELS_TABLES = ["table_available_trained_models_desc",
                     "tableWidget_trained_models_topics"]

    # Settings management-realated constants
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

    # Training related constants
    NR_PARAMS_TRAIN_LDA_MALLET = 7
    NR_PARAMS_TRAIN_PRODLDA = 17
    NR_PARAMS_TRAIN_CTM = 21
    NR_PARAMS_PREPROC = 4
    
    WRONG_NR_TOPICS_LDA_MSG = "The number of training topics must be larger than 0"
    WRONG_ALPHA_LDA_MSG = "The sum over topics of smoothing over doc-topic distributions (alpha) must be larger than 0."
    WRONG_OI_LDA_MSG = "The number of iterations between reestimating dirichlet hyperparameters (optimize interval) must be larger than 0. "
    WRONG_NR_THREADS_LDA_MSG = "The number of threads for parallel training must be larger than 0"
    WRONG_NR_ITER_LDA_MSG = "The number of iterations of Gibbs sampling must be larger than 0."
    WRONG_DOC_TPC_THR_LDA_MSG = "The hreshold for topic activation in a document during training (doc-topic thr) cannot be larger than 1."
    WRONG_THETAS_THR_LDA_MSG = "The threshold for topic activation in a doc during sparsification (thetas thr) cannot be larger than 1."
    WRONG_UNDERLYING_MODEL_TYPE_MSG = "The AVITM model type must be either 'prodLDA' or 'lda'."
    WRONG_MODEL_TYPE_MSG = "The CTM model type must be either 'CombinedTM or 'ZeroShotTM."
    WRONG_NR_TOPIC_MSG = "The number of training topics must be larger than 0."
    WRONG_NR_EPOCHS_MSG = "The number of epochs must be larger than 0."
    WRONG_BATCH_SIZE_MSG = "The batch size must be bigger than 0."
    WRONG_HIDDEN_SIZES_MSG = "Hidden_sizes must be type tuple."
    WRONG_ACTIVATION_MSG = "Activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'."
    WRONG_DROPOUT_MSG = "Dropout must be larger or equal to 0."
    WRONG_LEARN_PRIORS_MSG = "Learn_priors must be type bool."
    WRONG_LR_MSG = "The learning rate must a value in the range (0,1]"
    WRONG_MOMENTUM_MSG = "The learning momentum must a value in the range (0,1]"
    WRONG_SOLVER_MSG = "The NN optimizer to be used (solver) must be chosen from 'adagrad', 'adam', 'sgd', 'adadelta' or 'rmsprop'"
    WRONG_REDUCE_ON_PLATEAU_MSG = "Reduce_on_plateau must be type bool."
    WRONG_TOPIC_PRIOR_MEAN_MSG = ""
    WRONG_TOPIC_PRIOR_VAR_MSG = ""
    WRONG_NR_SAMPLES = "The number of samples must be a positive integer larger than 0."
    WRONG_NR_WORKERS = "The number of data loader workers must be a positive integer larger than 0."
    WRONG_LABEL_SIZE = "The label size must be equals to the number of documents."
    WRONG_LABEL_SIZE_FOR_SUPERCTM = "The label size cannot be 0 if SuperCTM is being used. It must be equal to the number of documents in the training corpus."
    WRONG_LOSS_WEIGTHS_FOR_BETACTM = "The weight loss cannot be None for a BetaCTM model."
    WARNING_NO_TR_CORPUS = "An appropiate training dataset must be selected to proceed."
    NO_NAME_FOR_MODEL = "A name for training the model must be specified"
    NO_DESC_FOR_MODEL = "A description for training the model must be specified"

    TRAIN_LOADING_BARS = ["progress_bar_train"]
    PREPROC_TABLES = ["table_available_stopwords","table_available_equivalences"]

    # Stylesheets
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
