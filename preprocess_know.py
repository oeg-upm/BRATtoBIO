"""
This python file contains the necessary function to process annotated data in Brat format.

 - Variable functions that depends on each case, so it may be modified:
    -> read_ann()
    -> process_all_files()
    -> process_data_parallel()
    -> __main__

"""

import pandas as pd
import argparse
import pysbd
import ast
import utils
import multiprocessing

from os import listdir
from random import randint
from os.path import isfile, join


def read_tsv(file):
    """
    Return a pandas.DataFrame with the annotation of the .tsv file.

    This function depends on the tsv file format, and it may be modified.
    The file is expected to be structured as follows:

    filename (tab) mark (tab) label (tab) off0 (tab) off1 (tab) span

    Example:

    filename    id  label   off0    off1    span
    ----------------------------------------------------------
    fl_example  T1	HUMAN   112     118	    hombre
    fl_example  T2	HUMAN   1025    1033	paciente


    Paramteres
    ----------
    file : str
        path to the tsv file

    Returns
    -------
    df : pandas.DataFrame
        pandas.DataFrame with the colums: [label, start, end, word]
    """

    relevant_columns = ["filename", "label", "off0", "off1", "span"]

    # Read txt sep by tabs
    df = pd.read_csv(file, sep="\t")

    # Drop not relevant columns
    for col in df.columns:
        if col not in relevant_columns:
            df.drop(col, axis=1, inplace=True)

    # Convert offsets to int
    df['off0'] = pd.to_numeric(df['off0'], errors="coerce")
    df['off1'] = pd.to_numeric(df['off1'], errors="coerce")

    # Sort by off0
    df = df.sort_values(by=['filename'])

    return df


def process_text(text, df_ann, ann_labels):
    """
    Process the input text to generate a dataset for the NER task.
    The text is divided into sentences. Each word has a label attached following the BIO annotation format.
    Finally, return a pandas.DataFrame with the dataset.

    Example of dataset:
    |tokens
    |------------------------------------------------------------------------------------------------
    |['Mujer', 'de', '67', 'años', 'con', 'antecedentes', 'personales', 'de', 'hipotiroidismo', '.']
    |
    |labels
    |------------------------------------------------------------------------------------------------
    |['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MORFOLOGIA_NEOPLASIA', 'O']

    Parameters
    ----------
    text : str
        text to process
    df_ann : pandas.DataFrame
        pandas.DataFrame with info about the spans (span, offset, word)
    ann_labels : list[str]
       list of lenght four with the labels of the ann file. Format must be:
            ann_labels[0] --> offset init label (E.g: 'start', 'off0', ...)
            ann_labels[1] --> offset final label (E.g: 'end', 'off1', ...)
            ann_labels[2] --> label for the column 'label' (E.g: 'label', ...)
            ann_labels[3] --> labelled word (E.g: 'word', 'span', ...)
            example       --> ['off0', 'off1', 'label', 'span']

    Returns
    -------
    pandas.DataFrame
        pandas.dataFrame with the text processed and labelled
    """

    columns = ['tokens', 'labels']
    df = pd.DataFrame(columns=columns)
    ann = df_ann.sort_values(by=[ann_labels[0]]).reset_index(drop=True)
    offset = 0
    str_char = "$STR$"
    itr_char = "$ITR$"
    len_char = len(str_char)
    seg = pysbd.Segmenter(language="es", clean=False)

    # Goes throw all the annotated word in the ann dataframe
    for index, row in ann.iterrows():
        off1 = row[ann_labels[0]]
        off2 = row[ann_labels[1]]
        label = row[ann_labels[2]]
        span = row[ann_labels[3]]

        # Check if the word is in the correct offset
        if text[off1 + offset:off2 + offset] == span:
            # Divid the span in words for cases as 'Torre Eiffel', labelled as place
            spans = span.split()
            if len(spans) >= 2:
                # Enter the if when the span has two or more words

                # To identity the labelled spans in the text it is added a mark with the string str_char (for the
                # beggining word) and itr_char (for the rest of words).
                # For each labelled word in the span, the special string and the label are added following the words
                # as follows. For the sentence 'The Eiffel Tower is un Paris' where 'Eiffel Tower' is labelled a PLACE,
                # the result is 'The Eiffel$STR$PLACE$ Tower$ITR$PLACE$ is un Paris'.

                # For the first word it is used the str_char
                off_aux = off1
                off_aux += len(spans[0])
                text = text[:offset + off_aux] + str_char + label + '$' + text[off_aux + offset:]
                off_aux += 1  # +1 bc of blank space
                offset += len_char + len(label) + 1

                # For every word except the first one
                for i in range(1, len(spans)):
                    # For the rest of the words it is used the itr_char
                    off_aux += len(spans[i])
                    text = text[:off_aux + offset] + itr_char + label + '$' + text[off_aux + offset:]
                    off_aux += 1  # +1 bc of blank space
                    offset += len_char + len(label) + 1
            else:
                # If there is only in word in the span, only it is needed to add the str_char
                text = text[:off2 + offset] + str_char + label + '$' + text[off2 + offset:]
                offset += len_char + len(label) + 1

    # Once the labelled word are marked, the text can be divided in sentences and assing labels to every word of the
    # sentences.

    sentences = seg.segment(text)
    all_tokens = []
    all_labels = []

    for sentence in sentences:
        new_sentence = sentence.split()
        labels = []
        tokens = []
        special_chars = ['\ufeff']
        for token in new_sentence:
            # Remove special characters as \ufeff
            for schar in special_chars:
                token = token.replace(schar, '')
            if "$STR$" in token:
                split_token = token.split("$STR$")
                # Sometimes the word has a parenthesis
                if '(' in split_token:
                    split_token = split_token.split('(')[1]
                    tokens.append('(')
                    labels.append('O')
                tokens.append(split_token[0])
                labels.append('B-' + split_token[1].split('$')[0])
            elif "$ITR$" in token:
                split_token = token.split("$ITR$")
                if '(' in split_token:
                    split_token = split_token.split('(')[1]
                    tokens.append('(')
                    labels.append('O')
                tokens.append(split_token[0])
                labels.append('I-' + split_token[1].split('$')[0])
            else:
                tokens.append(token)
                labels.append('O')
        all_tokens.append(tokens)
        all_labels.append(labels)

    df['tokens'] = all_tokens
    df['labels'] = all_labels

    for i in range(df.shape[0]):
        if len(df['tokens'][0]) != len(df['labels'][0]):
            print("MAAAL")

    return df


def process_data_parallel(txt_path_types, tsv_path_types, save_path_df, ann_labels, numthreads=8):
    """
    Manage a thread pool to parallel execution of process_all_files().

    Parameters
    ----------
    txt_path_types : list[str]
        path to train, dev and test text directories
    tsv_path_types : list[str]
        path to train, dev and test tsv directories
    save_path_df : list[str]
        path to train, dev and test save directories
    ann_labels : list[str]
        annotation labels for process_text()
    numthreads : int, optional
        number of threads to launch. Deafult = 8
    """

    pool = multiprocessing.Pool(numthreads)

    for path_txt, path_tsv, path_save in zip(txt_path_types, tsv_path_types, save_path_df):
        files_tsv = [f for f in listdir(path_tsv) if isfile(join(path_tsv, f))]
        df_tsv = pd.DataFrame()
        for file in files_tsv:
            df_tsv = pd.concat([df_tsv, read_tsv(path_tsv + file)]).reset_index(drop=True)

        txt_files = df_tsv['filename'].unique()
        for txt in txt_files:
            df_aux = df_tsv.loc[df_tsv['filename'] == txt].reset_index(drop=True)
            pool.apply_async(process_all_files, (path_txt, df_aux, path_save, ann_labels,))

    pool.close()
    pool.join()


def process_all_files(path_txt, df_tsv, path_save, ann_labels):
    """
    Main function to process all files.

    This function depends on the organization of the files, and it may be modified.
    In this case the files are organized as follows:
        unique_label_file.tsv

        text_file1.txt
            ...
        text_fileN.txt

    Parameters
    ----------
    path_txt : str
        path to the directory containing text files
    df_tsv : pandas.DataFrame
        pandas.DataFrame with all the annotated info
    path_save : str
        path to the directory to save the processed data
    ann_labels : list[str]
        annotation labels for process_text()
    """

    try:
        id_thread = multiprocessing.current_process().name
        print(f"Starting thread {id_thread}...")
    except:
        pass

    file = df_tsv['filename'][0] + '.txt'
    text = utils.read_txt(path_txt + file)

    df = process_text(text, df_tsv, ann_labels)

    file = file[:-4] if file[-4:] == ".txt" else file
    utils.write_csv(path_save + file + '.csv', df)

    try:
        print(f"Thread {id_thread} finished.")
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess data in brat format and save it in csv files. The files must be stored in a directory '
                    'with the name "raw_data".')
    parser.add_argument('path', help='Base path to data directory.')
    parser.add_argument('-al', '--ann_labels',
                        default="off0,off1,label,span",
                        help="Names of the offset start, offset end, label and span columns. "
                             "Input must be the names sep by a coma. E.g: -ann start,end,label,word. Default is "
                             "'off0,off1,label,span'.")
    parser.add_argument('-tr', '--train_dir', default="train-set",
                        help='Directory from path where train data is stored. Default is "train-set".')
    parser.add_argument('-de', '--dev_dir', default="dev-set",
                        help='Directory from path where evaluation data is stored. Default is "dev-set".')
    parser.add_argument('-te', '--test_dir', default="test-set",
                        help='Directory from path where test data is stored. Default is "test-dev".')
    parser.add_argument('-sn', '--save_name', default="processed_data",
                        help='Directory from path where processed data is stored in the train, dev and test'
                             'directories. Default is "processed_data"')
    parser.add_argument('-an', '--ann_name', default="NER_ann",
                        help='Directory from path where annotation files are stored in the train, dev and test '
                             'directories. Default is "NER_ann".')
    parser.add_argument('-txt', '--txt_name', default="text-files",
                        help='Directory from path where text files are stored in the train, dev and test directories. '
                             'Default is "text-files".')
    parser.add_argument('-n', '--num_threads', type=int, default=8,
                        help='Number of threads generated to process the data. Default is 8')

    args = parser.parse_args()

    path_ = args.path + '/' if args.path[-1] != '/' else args.path

    # dirs = [args.train_dir, args.dev_dir, args.test_dir]
    dirs = [args.train_dir, args.dev_dir]
    path_txt_types_ = []
    path_tsv_types_ = []
    save_path_df_ = []

    for dir_ in dirs:
        path_txt_types_.append(f"{path_}{dir_}/{args.txt_name}/")
        path_tsv_types_.append(f"{path_}{dir_}/{args.ann_name}/")
        save_path_df_.append(f"{path_}{dir_}/{args.save_name}/")

    num_threads = args.num_threads
    ann_labels_ = args.ann_labels.split(',')

    if utils.mkdirs(save_path_df_):
        exit(1)

    process_data_parallel(path_txt_types_,
                          path_tsv_types_,
                          save_path_df_,
                          ann_labels_,
                          numthreads=num_threads,
                          )


def get_all_labels(path_dir, col_labels):
    """
    DEPRECATED - NOT USED
    Return all the unique labels in a set of ann files.

    Parameters
    ----------
    path_dir : str
        path to the dir with the ann files
    col_labels : list[str]
        columns name for the labels in the ann files

    Returns
    -------
    list[str]
        all the labels in the ann files
    """

    labels = set()
    files = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
    for file in files:
        if 'tsv' in file:
            for element in read_tsv(path_dir + file).iterrows():
                labels.add(element[1][col_labels])

    final_labels = list(labels)

    # Add 'begin' and 'intern' label for every label
    for i in range(0, len(labels) * 2, 2):
        begin = f'B-{final_labels[i]}'
        intern = f'I-{final_labels[i]}'
        final_labels[i] = begin
        final_labels.insert(i + 1, intern)

    # Add 'O', [CLS] and [SEP] tokens
    final_labels.insert(0, 'O')
    final_labels += ['[CLS]', '[SEP]']

    return final_labels
