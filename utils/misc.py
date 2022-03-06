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
    Return: The integer conversion of user_str if possible; None otherwise
    """
    try:
        return int(user_str)
    except ValueError:
        return None


def is_float(user_str):
    """Check if input string can be converted to float
    param user_str: string to be converted to an integer value
    Return: The integer conversion of user_str if possible; None otherwise
    """
    try:
        return float(user_str)
    except ValueError:
        return None


def var_num_keyboard(vartype,default,question):
    """Read a numeric variable from the keyboard
    param vartype: Type of numeric variable to expect 'int' or 'float'
    param default: Default value for the variable
    param question: Text for querying the user the variable value
    Return: The value provided by the user, or the default value
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


def request_confirmation(msg="     Are you sure?"):

    # Iterate until an admissible response is got
    r = ''
    while r not in ['yes', 'no']:
        r = input(msg + ' (yes | no): ')

    return r == 'yes'


def query_options(options, msg):
    """
    Prints a heading and the options, and returns the one selected by the user
    Args:
        options        : Complete list of options
        msg            : Heading message to be printed before the list of
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
