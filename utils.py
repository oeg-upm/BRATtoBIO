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
