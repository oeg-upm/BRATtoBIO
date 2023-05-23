"""
This python file contains the necessary function to convert annotated data in Brat format to BIO format.

The files need to be in a specific structure. The directory dataset must contain at least one set (train, dev, test).
Each set directory must contain two main directories:
    1. First one with the text files (raw data)
    2. Second one with the file with the annotation. Only one file with all the annotations. The structure of this file
       is as follows (each column is sep by a tab):

            filename    mark  label   off0    off1    span
            ----------------------------------------------------------
            fl_example  T1	  HUMAN   112     118	  hombre
            fl_example  T2	  HUMAN   1025    1033	  paciente

Example of dataset organization:

/path/to/dataset
└── cantemist
    ├── train-set
    │   ├── cantemist-ner
    │   │   └── ner_annotations.tsv
    │   └── text-files
    │       ├── text-file_1.txt
    │       ├── text-file_2.txt
    │       └── ...
    │
    ├── dev-set
    │   ├── cantemist-ner
    │   │   └── ner_annotations.tsv
    │   └── text-files
    │       ├── text-file_1.txt
    │       ├── text-file_2.txt
    │       └── ...
    │
    └── test-set
        ├── cantemist-ner
        │   └── ner_annotations.tsv
        └── text-files
            ├── text-file_1.txt
            ├── text-file_2.txt
            └── ...

Each set has a directory 'text-files' with the raw data and a directory 'cantemist-ner' where the annotated data is
located

"""

import os
import argparse
import pysbd
import utils
import multiprocessing
import pandas as pd

from tqdm import tqdm
from os import listdir
from os.path import isfile, join


def read_tsv(file, ann_labels, header):
    """
    Return a pandas.DataFrame with the annotation of the .tsv file.

    This function depends on the tsv file format, and it may be modified.
    The file is expected to be structured as follows:

    filename (tab) mark (tab) label (tab) off0 (tab) off1 (tab) span

    Example:

    filename    id  label   off0    off1    span
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

    # relevant_columns = ["filename", "label", "off0", "off1", "span"]

    # Read txt sep by tabs
    df = pd.read_csv(file, sep="\t", names=header, header=0)

    # Drop not relevant columns
    for col in df.columns:
        if col not in ann_labels:
            df.drop(col, axis=1, inplace=True)

    # Convert offsets to numeric (conversion to int is later)
    # df['off0'] = pd.to_numeric(df['off0'], errors="coerce")
    # df['off1'] = pd.to_numeric(df['off1'], errors="coerce")
    # Convert offsets to numeric (conversion to int is later)
    df[ann_labels[2]] = pd.to_numeric(df[ann_labels[2]], errors="coerce")
    df[ann_labels[3]] = pd.to_numeric(df[ann_labels[3]], errors="coerce")

    # Sort by off0
    # df = df.sort_values(by=['filename'])
    df = df.sort_values(by=[header[0]])

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
    # ann = df_ann.sort_values(by=[ann_labels[0]]).reset_index(drop=True)
    offset = 0
    str_char = "$STR$"
    itr_char = "$ITR$"
    len_char = len(str_char)
    seg = pysbd.Segmenter(language="es", clean=False)

    # Goes throw all the annotated word in the ann dataframe
    for index, row in df_ann.iterrows():
        label = row[ann_labels[1]]
        off1 = int(row[ann_labels[2]])
        off2 = int(row[ann_labels[3]])
        span = row[ann_labels[4]]

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
        else:
            print(f"{utils.Bcolors.WARNING}WARNING: An span offset do not correspond its position on text --> "
                  f"Filename: {row['filename']}, Span: {span}, off0: {off1}, off1: {off2}{utils.Bcolors.ENDC}")

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


def process_file(path_txt, df_tsv, path_save, ann_labels, txt_tasks):
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
    ann_labels : List[str]
        annotation labels for process_text()
    txt_tasks : List[str]
    """

    global cont

    thread_id = multiprocessing.current_process().name
    # print(f"Id thread: {thread_id}")

    for txt in txt_tasks:
        # df_aux = df_tsv.loc[df_tsv['filename'] == txt].reset_index(drop=True)
        df_aux = df_tsv.loc[df_tsv[ann_labels[0]] == txt].sort_values(by=ann_labels[2]).reset_index(drop=True)
        # df_aux = df_tsv.loc[df_tsv[ann_labels[0]] == txt].reset_index(drop=True)

        file = f"{txt}.txt"
        text = utils.read_txt(path_txt + file)

        df = process_text(text, df_aux, ann_labels)

        file = file[:-4] if file[-4:] == ".txt" else file
        utils.write_csv(path_save + file + '.csv', df)

        with cont.get_lock():
            cont.value += 1


def distribute_tasks(num_tasks):
    threads = os.cpu_count()
    tasks_per_thread = (num_tasks + threads - 1) // threads
    tasks = [(i * tasks_per_thread, min((i + 1) * tasks_per_thread, num_tasks)) for i in range(threads)]
    tasks[-1] = (tasks[-1][1], tasks[-1][1]) if tasks[-1][0] > tasks[-1][1] else tasks[-1]
    return tasks


# error callback function
def handler(error):
    print(f'{utils.Bcolors.FAIL}Error: {error}{utils.Bcolors.ENDC}', flush=True)
    exit(1)


def init_pool_processes(cont_):
    """
    Initialize each process with a global variable lock.
    """

    global cont
    cont = cont_


def process_data_parallel(txt_path_types, tsv_path_types, save_path_df, ann_labels, header, numthreads=os.cpu_count()):
    """
    Manage a thread pool to parallel execution of process_file().

    Futuro: hacer una comprobación de que todos los textos se han procesdao bien. Un hilo pude devolver lo que se ha
    procesdao.

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
        number of threads to launch. Deafult = os.cpu_count()
    """

    print('-' * (len(txt_path_types[0]) + 35))
    print(f"Processing dataset")
    cont_ = multiprocessing.Value('i', 1)
    with multiprocessing.Pool(initializer=init_pool_processes, initargs=(cont_,), processes=numthreads) as pool:
        for path_txt, path_tsv, path_save in zip(txt_path_types, tsv_path_types, save_path_df):
            files_tsv = [f for f in listdir(path_tsv) if isfile(join(path_tsv, f))]
            df_tsv = pd.DataFrame()
            for file in files_tsv:
                df_tsv = pd.concat([df_tsv, read_tsv(path_tsv + file, ann_labels, header)]).reset_index(drop=True)

            # ORDENAR DATAFRAME

            # txt_files = df_tsv['filename'].unique()
            txt_files = df_tsv[header[0]].unique()
            tasks = distribute_tasks(len(txt_files))
            res = []

            # Cont start on 1 to stops the while loop where the tqdm is updated
            cont_.value = 1

            for task in tasks:
                txt_task = txt_files[task[0]:task[1]]
                # res = pool.apply_async(process_file, (path_txt, df_tsv, path_save, ann_labels, txt_task,),
                #                        error_callback=handler)
                res.append(pool.apply_async(process_file, (path_txt, df_tsv, path_save, ann_labels, txt_task,),
                                            error_callback=handler))

            # While loop to update the tqdm bar. The shared value cont_ is updated by each thread when a text if
            # processed. This loop checks the cont_.value every time and update the bar when it is modified.
            # The init cont_.value is 1 because to avoid satisfing the condition cont_.value <= len(txt_files) at the
            # end. If the init value is 0, the condition is always satified (max cont_.value is the amount of text
            # to process).
            prev = cont_.value
            pbar = tqdm(total=len(txt_files),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        desc=f"{path_txt.split('/')[-3]}")
            while cont_.value <= len(txt_files):
                if cont_.value != prev:
                    diff = cont_.value - prev
                    prev = cont_.value
                    pbar.update(diff)

            for r in res:
                r.wait()

            if cont_.value != prev:
                diff = cont_.value - prev
                pbar.update(diff)

            pbar.close()

    pool.join()
    print(f"{utils.Bcolors.OKGREEN}Dataset processed successfully{utils.Bcolors.ENDC}")


if __name__ == "__main__":
    debbug = True

    if debbug:
        header_ = "filename,mark,label,off0,off1,span".split(',')
        ann_labels_ = [header_[0], header_[2], header_[3], header_[4], header_[5]]
        df_tsv = read_tsv("/home/carlos/datasets/softcite/train-set/softcite-ner/ner_annotations.tsv",
                          ann_labels_,
                          header_).reset_index(drop=True)

        process_file("/home/carlos/datasets/softcite/train-set/text-files/",
                     df_tsv,
                     "/home/carlos/datasets/softcite/train-set/processed_data/",
                     ann_labels_,
                     ["PMC5144680"])

        exit(0)

    parser = argparse.ArgumentParser(
        description='Preprocess data in BRAT format and save it in csv files on BIO format. '
                    'The positional argument is the path to the dataset directory and then ONLY the name of the'
                    'set directories (train, dev and test). At least one must be provided.'
                    'The annotation must be in a SINGLE file with a header. '
                    'This header of the annotation file can be passed as argument if its different to the default '
                    'value. Also, it is mandatory to follow the order "filename, mark, start, end, label, span" and '
                    'only the names can be modified.')
    parser.add_argument('path', help='Base path to data directory.')
    parser.add_argument('-tr', '--train_dir',
                        help='Directory from path where train data is stored.')
    parser.add_argument('-de', '--dev_dir',
                        help='Directory from path where evaluation data is stored.')
    parser.add_argument('-te', '--test_dir',
                        help='Directory from path where test data is stored.')
    parser.add_argument('-an', '--ann_name',
                        help='Directory from path where annotation files are stored in the train, dev and test '
                             'directories.')
    parser.add_argument('-hd', '--header',
                        default="filename,mark,off0,off1,label,span",
                        help="Header of the annotation file. The column name can differ from the default but the order "
                             "must be the same. E.g, the file name must be the first column."
                             "Default is filename,mark,label,off0,off1,span.")
    # parser.add_argument('-ac', '--ann_cols',
    #                     default="off0,off1,label,span",
    #                     help="Names of the offset start, offset end, label and span columns."
    #                          "Input must be the names sep by a coma and in the same order and name than in the header"
    #                          ". E.g: -ann start,end,label,word. Default is 'off0,off1,label,span'.")
    parser.add_argument('-sn', '--save_name', default="processed_data",
                        help='Directory from path where processed data is stored in the train, dev and test'
                             'directories. Default is "processed_data"')
    parser.add_argument('-txt', '--txt_name', default="text-files",
                        help='Directory from path where text files are stored in the train, dev and test directories. '
                             'Default is "text-files".')
    parser.add_argument('-n', '--num_threads', type=int, default=os.cpu_count(),
                        help='Number of threads generated to process the data. Default is os.cpu_count()')

    args = parser.parse_args()

    path_ = args.path + '/' if args.path[-1] != '/' else args.path

    dirs = []
    if args.train_dir is not None:
        dirs.append(args.train_dir)
    if args.dev_dir is not None:
        dirs.append(args.dev_dir)
    if args.test_dir is not None:
        dirs.append(args.test_dir)

    if not dirs:
        print(f"{utils.Bcolors.FAIL}No set provided in dataset{utils.Bcolors.ENDC}")
        exit(0)
    if args.txt_name is None:
        print(f"{utils.Bcolors.FAIL}No text files provided{utils.Bcolors.ENDC}")
        exit(0)
    if args.ann_name is None:
        print(f"{utils.Bcolors.FAIL}No ann files provided{utils.Bcolors.ENDC}")
        exit(0)

    path_txt_types_ = []
    path_tsv_types_ = []
    save_path_df_ = []

    for dir_ in dirs:
        path_txt_types_.append(f"{path_}{dir_}/{args.txt_name}/")
        path_tsv_types_.append(f"{path_}{dir_}/{args.ann_name}/")
        save_path_df_.append(f"{path_}{dir_}/{args.save_name}/")

    num_threads = args.num_threads
    header_ = args.header.split(',')
    ann_labels_ = [header_[0], header_[2], header_[3], header_[4], header_[5]]

    if utils.mkdirs(save_path_df_):
        exit(1)

    # print(path_txt_types_,
    #                       path_tsv_types_,
    #                       save_path_df_,
    #                       ann_labels_,
    #                       num_threads,
    #                       sep='\n')
    # exit(0)

    process_data_parallel(path_txt_types_,
                          path_tsv_types_,
                          save_path_df_,
                          ann_labels_,
                          header_,
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
