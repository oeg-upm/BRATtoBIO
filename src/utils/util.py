"""
Utility functions to preprocess a dataset such us write a dataframe into a csv, read a text or make directories.
It also contains a class with colors to print strings
"""
import pandas

from os import mkdir
from os.path import exists


class Bcolors:
    """
    A class used to color the printed strings

    ...

    Attributes
    ----------
    HEADER : str
    OKBLUE : str
    OKCYAN : str
    OKGREEN : str
    WARNING : str
    FAIL : str
    ENDC : str
        ends the colored string
    BOLD : str
    UNDERLINE : str
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def correct_sentences(sentences: list[str]):
    """
    There are some scenarios where the sentence segmenter divices an entity into two sentences.
    For example, in the text '

        '[...]. Por la presencia de C. koseri en [...]', C. koseri is an entity.

    After the first processing the text, the result is

        '[...]. Por la presencia de $INI$C.$STR$SPECIE$END$ $INIkoseri$ITR$SPECIE$END$ en [...]'

    The result after the segmentation of the text is two sentences:

        Sentence 1: 'Por la presencia de $INI$C.'
        Sentence 2: '$STR$SPECIE$END$ $INIkoseri$ITR$SPECIE$END$ en [...]'

    To solve this problem, this function checks the first word of every sentence. If $END$ is present but $INI$ is not
    present, it means that the entity has been divided. The solution is to concatenate this sentence to the previous.

        Final sentence: ''Por la presencia de $INI$C.$STR$SPECIE$END$ $INIkoseri$ITR$SPECIE$END$ en [...]'

    Parameters
    ----------
    sentences

    Returns
    -------

    """
    new_sentences = []
    for sentence in sentences:
        if "$INI$" not in sentence.split()[0] and "$END$" in sentence.split()[0]:
            new_sentences[-1] += sentence
        else:
            new_sentences.append(sentence)
    return new_sentences


def split_token(token):
    """

    Parameters
    ----------
    token

    Returns
    -------

    """
    import re

    # Definir los patrones de expresiones regulares para los casos
    patterns = [
        r'\$INI\$',  # Separador "$INIT$"
        r'\$END\$',  # Separador "$END$"
    ]

    # Unir los patrones en una expresión regular
    pattern = '|'.join(patterns)

    # Dividir la cadena utilizando la expresión regular
    tokens = re.split(pattern, token)

    return [tok for tok in tokens if tok != '']


def write_csv(path_file, df):
    """
    Write a pandas dataframe into a csv file

    Parameters
    ----------
    path_file : srt
        path to save dir
    df : pandas.DataFrame
        Dataframe to save
    """

    df.to_csv(path_file, index=False, header=True, encoding="utf-8")


def read_txt(file):
    """
    Return the content of a txt file in a string

    Parameters
    ----------
    file : str
        path to the txt

    Returns
    ------
    str
        string with the text

    """
    with open(file, 'r', encoding="utf8") as file:
        text = file.read()

    return text


def mkdirs(paths):
    """
    Make directories

    Parameters
    ----------
    paths : list[str]
         path to the base directory
    """

    state = 0
    for path in paths:
        if not exists(path):
            try:
                mkdir(path)
                print(f"Successfully mkdir {path}")
            except FileNotFoundError as e:
                print(f"{Bcolors.FAIL}{e}{Bcolors.ENDC}")
                state = 1
        else:
            print(f"{Bcolors.WARNING}Directory {path} already exists{Bcolors.ENDC}")

    return state
