import datetime
import json
import logging
import os
from typing import Optional
import colored


def printgr(text):
    print(colored.stylize(text, colored.fg('green')))


def printred(text):
    print(colored.stylize(text, colored.fg('red')))


def printmag(text):
    print(colored.stylize(text, colored.fg('magenta')))


def is_integer(user_str):
    """Check if input string can be converted to integer
    param user_str: string to be converted to an integer value

    Returns
    -------
    :
        The integer conversion of user_str if possible; None otherwise
    """
    try:
        return int(user_str)
    except ValueError:
        return None


def is_float(user_str):
    """Check if input string can be converted to float
    param user_str: string to be converted to an integer value

    Returns
    -------
    :
        The integer conversion of user_str if possible; None otherwise
    """
    try:
        return float(user_str)
    except ValueError:
        return None


def var_num_keyboard(vartype,default,question):
    """Read a numeric variable from the keyboard

    Parameters
    ----------        
    vartype:
        Type of numeric variable to expect 'int' or 'float'
    default:
        Default value for the variable
    question:
        Text for querying the user the variable value
    
    Returns
    -------
    :
        The value provided by the user, or the default value
    """
    aux = input(question + ' [' + str(default) + ']: ')
    if vartype == 'int':
        aux2 = is_integer(aux)
    else:
        aux2 = is_float(aux)
    if aux2 is None:
        if aux != '':
            print('The value you provided is not valid. Using default value.')
        return default
    else:
        if aux2 >= 0:
            return aux2
        else:
            print('The value you provided is not valid. Using default value.')
            return default
            

def var_arrnum_keyboard(vartype,default,question):
    """Read a list with numeric values from the keyboard

    Parameters
    ----------        
    vartype:
        Type of numeric variables to expect 'int' or 'float'
    default:
        Default value for the variable
    question:
        Text for querying the user
    
    Returns
    -------
    :
        A list with the values provided by the user. 
        If only one element is to be returned, we return just a number
        If no feasible values is provided, we return the default value
    """
    aux = input(question + ' [' + str(default) + ']: ')
    # We generate a list with all values that were given separated by commas
    aux2 = aux.split(',')
    if vartype == 'int':
        aux2 = [is_integer(el) for el in aux2 if is_integer(el) and is_integer(el)>=0]
    else:
        aux2 = [is_float(el) for el in aux2 if is_float(el) and is_float(el)>=0]
    if not len(aux2):
        print('The value you provided is not valid. Using default value.')
        return default
    elif len(aux2)==1:
        return aux2[0]
    else:
        return aux2


def var_string_keyboard(option, default, question):
    """Read a string variable from the keyboard

    Parameters
    ----------      
    option: str
        Expected format of the string provided  
    default: str
        Default value for the variable
    question: str
        Text for querying the user the variable value
    
    Returns
    -------
    :
        The value provided by the user, or the default value
    """
    
    aux = input(question + ' [' + str(default) + ']: ')
    if option == "comma_separated":
        aux2 = [el.strip() for el in aux.split(',') if len(el)]
        if aux2 is not None and len(aux2) <= 1:
            print('The value you provided is not valid. Using default value.')
            return default
        else:
            aux2 = tuple(map(int, aux[1:-1].split(',')))
    elif option == "bool":
        if aux is not None and aux != "True" and aux != "False":
            print('The value you provided is not valid. Using default value.')
            aux2 = None
            return default
        else:
            aux2 = True if aux == "True" else False
    elif option == "dict":
        if aux != '':
            if aux == "None":
                return default
            else:
                try:
                    aux2 = json.loads(aux)
                except:
                    print('The value you provided is not valid. Using default value.')
                    return default
    elif option == "str":
        aux2 = str(aux) if aux != '' else None
    if aux2 is None:
        if aux != '':
            print('The value you provided is not valid. Using default value.')
        return default
    else:
        return aux2


def request_confirmation(msg="     Are you sure?"):

    # Iterate until an admissible response is got
    r = ''
    while r not in ['yes', 'no']:
        r = input(msg + ' (yes | no): ')

    return r == 'yes'


def query_options(options, msg):
    """
    Prints a heading and the options, and returns the one selected by the user

    Parameters
    ----------
    options:
        Complete list of options
    msg:
        Heading message to be printed before the list of
        available options
    """

    print(msg)

    count = 0
    for n in range(len(options)):
        #Print active options without numbering lags
        print(' {}. '.format(count) + options[n])
        count += 1

    range_opt = range(len(options))

    opcion = None
    while opcion not in range_opt:
        opcion = input('What would you like to do? [{0}-{1}]: '.format(
            str(range_opt[0]), range_opt[-1]))
        try:
            opcion = int(opcion)
        except:
            print('Write a number')
            opcion = None

    return opcion


def format_title(tgt_str):
    #sentences = sent_tokenize(tgt_str)
    #capitalized_title = ' '.join([sent.capitalize() for sent in sentences])
    capitalized_title = tgt_str
    #Quitamos " y retornos de carro
    return capitalized_title.replace('"','').replace('\n','')


def init_logger(
    config_file: str,
    name: str = None
) -> logging.Logger:
    """
    Initialize a logger based on the provided configuration.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """

    logger_config = load_yaml_config_file(config_file, "logger", logger=None)
    name = name if name else logger_config.get("logger_name", "default_logger")
    log_level = logger_config.get("log_level", "INFO").upper()
    dir_logger = pathlib.Path(logger_config.get("dir_logger", "logs"))
    N_log_keep = int(logger_config.get("N_log_keep", 5))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create path_logs dir if it does not exist
    dir_logger.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in {dir_logger}")

    # Generate log file name based on the data
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_log_{current_date}.log"
    log_file_path = dir_logger / log_file_name

    # Remove old log files if they exceed the limit
    log_files = sorted(dir_logger.glob("*.log"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    if len(log_files) >= N_log_keep:
        for old_file in log_files[N_log_keep - 1:]:
            old_file.unlink()

    # Create handlers based on config
    if logger_config.get("file_log", True):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if logger_config.get("console_log", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger

def load_yaml_config_file(config_file: str, section: str, logger:logging.Logger) -> Dict:
    """
    Load a YAML configuration file and return the specified section.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    section : str
        Section of the configuration file to return.

    Returns
    -------
    Dict
        The specified section of the configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the specified section is not found in the configuration file.
    """

    if not pathlib.Path(config_file).exists():
        log_or_print(f"Config file not found: {config_file}", level="error", logger=logger)
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    section_dict = config.get(section, {})

    if section == {}:
        log_or_print(f"Section {section} not found in config file.", level="error", logger=logger)
        raise ValueError(f"Section {section} not found in config file.")

    log_or_print(f"Loaded config file {config_file} and section {section}.", logger=logger)

    return section_dict

def log_or_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Helper function to log or print messages.

    Parameters
    ----------
    message : str
        The message to log or print.
    level : str, optional
        The logging level, by default "info".
    logger : logging.Logger, optional
        The logger to use for logging, by default None.
    """
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
    else:
        print(message)