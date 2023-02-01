"""
This python file contains the necessary function to process annotated data in Brat format.

It is structured as follows:
 - Standard functions that are general for almost every case:
    -> write_csv()
    -> read_txt()
    -> process_text()
    -> mkdir_processed()

 - Variable functions that depends on each case, so it may be modified:
    -> read_ann()
    -> process_all_files()
    -> process_data_parallel()
    -> __main__

"""

from os import listdir, mkdir
import multiprocessing
from os.path import isfile, join, exists
from random import randint
import pandas as pd
import argparse
import pysbd
import ast


def write_csv(path_file, df):
    """
        Write a pandas.dataFrame in a csv

        Input:
            - path_file:    path to save dir
            - df:           dataFrame to save

    """

    df.to_csv(path_file, index=False, header=True, encoding="utf-8")


def get_all_labels(path_dir, col_labels):
    """
        Return all the unique labels in a set of ann files

        Input:
            - path_dir:     path to the dir with the ann files
            - col_labels:   columns name for the labels in the ann files

        Return:
            - final_labels: all the labels in the ann files

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
        begin = 'B-' + final_labels[i]
        intern = 'I-' + final_labels[i]
        final_labels[i] = begin
        final_labels.insert(i + 1, intern)

    # Add 'O', [CLS] and [SEP] tokens
    final_labels.insert(0, 'O')
    final_labels += ['[CLS]', '[SEP]']

    return final_labels


def read_txt(file):
    """
    Return the content of a txt file in a string

    Input:
        - file: path to the txt

    Return:
        - text: string with the text

    """
    with open(file, 'r', encoding="utf8") as file:
        text = file.read()

    return text


def process_text(text, ann, ann_labels):
    columns = ['tokens', 'labels']
    df = pd.DataFrame(columns=columns)
    ann = ann.sort_values(by=[ann_labels[0]]).reset_index(drop=True)
    offset = 0
    str_char = "$STR$"
    itr_char = "$ITR$"
    len_char = len(str_char)
    seg = pysbd.Segmenter(language="es", clean=False)

    for index, row in ann.iterrows():
        off1 = row[ann_labels[0]]
        off2 = row[ann_labels[1]]
        label = row[ann_labels[2]]
        span = row[ann_labels[3]]

        # Check if offset is correct
        if text[off1 + offset:off2 + offset] == span:
            # Add label to the end of the word with and special character
            if len(span.split()) >= 2:
                spans = span.split()
                off_aux = off1
                for i in range(len(spans)):
                    if i == 0:
                        off_aux += len(spans[i])
                        text = text[:off_aux + offset] + str_char + label + '$' + text[off_aux + offset:]
                        off_aux += 1  # +1 bc of blank space
                        offset += len_char + len(label) + 1
                    else:
                        off_aux += len(spans[i])
                        text = text[:off_aux + offset] + itr_char + label + '$' + text[off_aux + offset:]
                        off_aux += 1    # +1 bc of blank space
                        offset += len_char + len(label) + 1
            else:
                text = text[:off2 + offset] + str_char + label + '$' + text[off2 + offset:]
                offset += len_char + len(label) + 1

    sentences = seg.segment(text)
    all_tokens = []
    all_labels = []

    for sentence in sentences:

        new_sentence = sentence.split()
        labels = []
        tokens = []
        for token in new_sentence:
            token = token.replace('\ufeff', '')
            if "$STR$" in token:
                split_token = token.split("$STR$")
                tokens.append(split_token[0])
                labels.append('B-' + split_token[1].split('$')[0])
            elif "$ITR$" in token:
                split_token = token.split("$ITR$")
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


def process_text_deprecated(text, ann, ann_labels):
    """
    Process the text with the annotations and return a pandas.dataFrame.

    Example:

    tokens
    ------------------------------------------------------------------------------------------------
    ['Mujer', 'de', '67', 'años', 'con', 'antecedentes', 'personales', 'de', 'hipotiroidismo', '.']

    labels
    ------------------------------------------------------------------------------------------------
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MORFOLOGIA_NEOPLASIA', 'O']


    Input:
        - text:         text to process
        - ann:          info about the spans (span, offset, word)
        - ann_labels:   list of labels of the ann file. Format must be:
                        ann_labels[0] --> offset init label (E.g: 'start', 'off0', ...)
                        ann_labels[1] --> offset final label (E.g: 'end', 'off1', ...)
                        ann_labels[2] --> label for the column 'label' (E.g: 'label', 'span', ...)
                        ann_labels[3] --> labeled word (E.g: 'word', 'span', ...)

    Return:
        - df:           pandas.dataFrame with the text processed and labeled

    """
    columns = ['tokens', 'labels']
    df = pd.DataFrame(columns=columns)
    ann = ann.sort_values(by=[ann_labels[0]]).reset_index(drop=True)
    labels_phrase = []
    phrase = []
    word = ""

    cont = 0
    label_index = 0
    for letter in text:
        # Si se ha terminado de etiquetar la palabra entonces hay que pasar a la siguiente etiqueta
        if cont == ann[ann_labels[1]][label_index] and label_index < len(ann) - 1:
            # Increment the label indexer to the next labeled word
            label_index += 1

        # Si hay un punto es un posible final de frase
        # Dot can represent the final of a sentence
        if letter == '.':
            is_int = False

            # Comprobar que se puede evaluar el siguiente caracter, es decir que no estamos al final del texto
            # Must check if there is a next character (avoid an error in the last char of the text)
            if cont + 1 < len(text):
                # Miramos a ver si el siguiente es un 'int', lo que quiere decir que estamos viendo un número decimal
                # If the next char is a number, the dot is part of a decimal numbr
                is_int = text[cont + 1].isdigit()

            # Si es un 'int' hay que continuar formando la palabra
            # If is_int then avoid the end of sentence code
            if is_int:
                word += letter

            # Final de frase
            # End of sentence
            else:
                # La palabra que estábamos formando no se ha añadido a la frase
                # Add the word being formed to the sencente
                if word:
                    phrase.append(word)

                # Añadimos el punto y su etiqueta a la frase
                # Add the dot and the null label to the sentence
                phrase.append(letter)
                labels_phrase.append('O')

                # Añadimos la frase al DataFrame
                # Add the sentence to the dataFrame
                new_df = pd.DataFrame({columns[0]: [phrase], columns[1]: [labels_phrase]})
                df = pd.concat([df, new_df])

                # Reiniciamos las variables
                # Reset the variable
                labels_phrase = []
                phrase = []
                word = ""

        # Si hay un salto de línea es un posible final de frase
        # New line can represent the final of a sentence
        elif letter == '\n':
            # Si no se está formando ninguna frase entonces no hay nada que hacer
            # Check if a sentence is being formed. If not, pass
            if phrase:
                is_finished = True

                # Comprobar que se puede evaluar el siguiente caracter, es decir que no estamos al final del texto
                # Must check if there is a next character (avoid an error in the last char of the text)
                if cont + 1 < len(text):
                    # Si la primera letra de la siguiente frase está en minúscula entonces hay que seguir con la frase
                    # If the next letter is in lower case, the sentence continues in the next line
                    is_finished = not text[cont + 1].isupper()

                # Si no es el final de la frase hay que seguir formándola
                # If sentence is not finished
                if not is_finished:
                    # Si hay una palabra, añadirla. Si no la hay puede ser que haya un espacio antes del enter
                    # --
                    # Add the word to the sentence if exists. If not, it is possible that there is a blank space
                    # before the new line
                    if word:
                        phrase.append(word)
                        word = ""

                # Hemos llegado al final de la frase
                # End of sentence reached
                else:
                    # La palabra que estábamos formando no se ha añadido a la frase
                    # If word exists, add it to the sentence
                    if word:
                        phrase.append(word)

                    # Añadimos la frase al DataFrame
                    # Add the sentence to the dataFrame
                    new_df = pd.DataFrame({columns[0]: [phrase], columns[1]: [labels_phrase]})
                    df = pd.concat([df, new_df])

                    # Reiniciamos las variables
                    # Reset the variables
                    labels_phrase = []
                    phrase = []
                    word = ""

        # Si hay un espacio significa que se ha terminado de formar la palabra
        # Blank represent the end of the word
        elif letter == ' ' and word:
            phrase.append(word)
            word = ""

        # Si no hay ninguna palabra y 'letter' es un caracter válido en una frase entonces estamos empezando
        # a formar una palabra
        # --
        # If not word and letter is a valid char (not a blank space) then a new word is being formed

        elif not word and letter != ' ':
            # Estamos empezando a etiquetar una palabra con 'Begin'
            # Is a begin labeled word
            if cont == ann[ann_labels[0]][label_index]:
                labels_phrase.append('B-' + ann[ann_labels[2]][label_index])

            # Estamos formando una palabra etiquetada que es 'Inside'
            # Is an inside labeled word
            elif cont in range(ann[ann_labels[0]][label_index], ann[ann_labels[1]][label_index]):
                labels_phrase.append('I-' + ann[ann_labels[2]][label_index])

            # Estamos formando una palabra no etiquetada, se le asigna 'O'
            # Is a null labeled word
            elif cont not in range(ann[ann_labels[0]][label_index], ann[ann_labels[1]][label_index]):
                labels_phrase.append('O')

            word += letter

        # Finalmente, se añade la letra a la palabra
        # Finally, add the letter to the word
        elif word and letter != ' ':
            word += letter

        cont += 1

    return df


def process_text_knowledge(text, ann, ann_labels, db_knowledge):
    columns = ['tokens', 'labels', 'knowledge']
    df = pd.DataFrame(columns=columns)
    ann = ann.sort_values(by=[ann_labels[0]]).reset_index(drop=True)
    offset = 0
    str_char = "$STR$"
    itr_char = "$ITR$"
    len_char = len(str_char)
    seg = pysbd.Segmenter(language="es", clean=False)

    for index, row in ann.iterrows():
        off1 = row[ann_labels[0]]
        off2 = row[ann_labels[1]]
        label = row[ann_labels[2]]
        span = row[ann_labels[3]]

        # Check if offset is correct
        if text[off1 + offset:off2 + offset] == span:
            # Add label to the end of the word with and special character
            if len(span.split()) >= 2:
                spans = span.split()
                off_aux = off1
                for i in range(len(spans)):
                    if i == 0:
                        off_aux += len(spans[i])
                        text = text[:off_aux + offset] + str_char + label + '$' + text[off_aux + offset:]
                        off_aux += 1  # +1 bc of blank space
                        offset += len_char + len(label) + 1
                    else:
                        off_aux += len(spans[i])
                        text = text[:off_aux + offset] + itr_char + label + '$' + text[off_aux + offset:]
                        off_aux += 1  # +1 bc of blank space
                        offset += len_char + len(label) + 1
            else:
                text = text[:off2 + offset] + str_char + label + '$' + text[off2 + offset:]
                offset += len_char + len(label) + 1

    sentences = seg.segment(text)
    all_tokens = []
    all_labels = []
    all_knowledge = []

    for sentence in sentences:

        new_sentence = sentence.split()
        labels = []
        tokens = []
        for token in new_sentence:
            token = token.replace('\ufeff', '')
            if "$STR$" in token:
                split_token = token.split("$STR$")
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

        sentence_knowledge = []
        for token in tokens:
            if token.lower() in list(db_knowledge['span']):
                knowledge = list(db_knowledge.loc[db_knowledge['span'] == token.lower()]['knowledge'].apply(ast.literal_eval))[0]
                knowledge = list(map(lambda x: x.replace('\n', ''), knowledge))
                sentence_knowledge += knowledge
        while len(sentence_knowledge) < 10:
            index = randint(0, len(db_knowledge) - 1)
            knowledge = list(db_knowledge.iloc[[index]]['knowledge'].apply(ast.literal_eval))[0]
            knowledge = list(map(lambda x: x.replace('\n', ''), knowledge))
            sentence_knowledge += knowledge
        if len(sentence_knowledge) > 10:
            sentence_knowledge = sentence_knowledge[:10]
        all_knowledge.append(sentence_knowledge)

    df['tokens'] = all_tokens
    df['labels'] = all_labels
    df['knowledge'] = all_knowledge

    for i in range(df.shape[0]):
        if len(df['tokens'][0]) != len(df['labels'][0]):
            print("MAAAL")

    return df


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


    Input:
        - file: path to the tsv file

    Return:
        - df:   pandas.DataFrame with the colums: [label, start, end, word]

    """
    columns = ["filename", "label", "off0", "off1", "span"]

    # Read txt sep by tabs
    df = pd.read_csv(file, sep="\t")

    for col in df.columns:
        if col not in columns:
            df.drop(col, axis=1, inplace=True)

    # Convert offsets to int
    df['off0'] = pd.to_numeric(df['off0'], errors="coerce")
    df['off1'] = pd.to_numeric(df['off1'], errors="coerce")

    # Sort by off0
    df = df.sort_values(by=['filename'])

    return df


def mkdir_processed(paths):
    """
        Make the directories to save processed data

        Input:
            - path:     path to the base directory
            - types:    path to train, dev and test directories
    """
    for path in paths:
        if not exists(path):
            mkdir(path)
            print(f"Created:\n{path}")


def process_all_files(path_txt, df_tsv, path_save, ann_labels, db_knowledge):
    """
        Main function to process all files.

        This function depends on the organization of the files, and it may be modified.
        In this case the files are organized as follows:
            unique_label_file.tsv

            text_file1.txt
            text_file2.txt
                    .
                    .
                    .

        Input:
            - path_txt:     path to the directory containing text files
            - df_tsv:       pandas.DataFrame with all the annotated info
            - path_save:    path to the directory to save the processed data
            - ann_labels:   annotation labels for process_text()

    """
    id_thread = multiprocessing.current_process().name
    print(f"Starting thread {id_thread}...")

    file = df_tsv['filename'][0] + '.txt'
    text = read_txt(path_txt + file)

    df = process_text_knowledge(text, df_tsv, ann_labels, db_knowledge)

    file = file[:-4] if file[-4:] == ".txt" else file
    write_csv(path_save + file + '.csv', df)

    print(f"Thread {id_thread} finished.")


def process_data_parallel(txt_path_types, tsv_path_types, save_df_path, ann_labels, path_knowledge, numthreads=8):
    """
        Manage a thread pool to parallel execution of process_all_files().

        Input:
            - txt_path_types:   path to train, dev and test text directories
            - tsv_path_types:   path to train, dev and test tsv directories
            - save_df_path:     path to train, dev and test save directories
            - ann_labels:       annotation labels for process_text()
            - numthreads:       number of threads to launch. Deafult = 8

    """
    pool = multiprocessing.Pool(numthreads)

    db_knowledge = pd.read_csv(path_knowledge, sep="\t")
    db_knowledge.pop('Unnamed: 0')

    for path_txt, path_tsv, path_save in zip(txt_path_types, tsv_path_types, save_df_path):
        files_tsv = [f for f in listdir(path_tsv) if isfile(join(path_tsv, f))]
        df_tsv = pd.DataFrame()
        for file in files_tsv:
            df_tsv = pd.concat([df_tsv, read_tsv(path_tsv + file)]).reset_index(drop=True)

        txt_files = df_tsv['filename'].unique()
        for txt in txt_files:
            df_aux = df_tsv.loc[df_tsv['filename'] == txt].reset_index(drop=True)
            pool.apply_async(process_all_files, (path_txt,
                                                 df_aux,
                                                 path_save,
                                                 ann_labels,
                                                 db_knowledge))

    pool.close()
    pool.join()


def test(txt_path_types, tsv_path_types, save_df_path, ann_labels, path_knowledge):
    db_knowledge = pd.read_csv(path_knowledge, sep="\t")
    db_knowledge.pop('Unnamed: 0')

    for path_txt, path_tsv, path_save in zip(txt_path_types, tsv_path_types, save_df_path):
        files_tsv = [f for f in listdir(path_tsv) if isfile(join(path_tsv, f))]
        df_tsv = pd.DataFrame()
        for file in files_tsv:
            df_tsv = pd.concat([df_tsv, read_tsv(path_tsv + file)]).reset_index(drop=True)

        txt_files = df_tsv['filename'].unique()
        for txt in txt_files:
            df_aux = df_tsv.loc[df_tsv['filename'] == txt].reset_index(drop=True)

            file = df_aux['filename'][0] + '.txt'
            text = read_txt(path_txt + file)

            df = process_text_knowledge(text, df_aux, ann_labels, db_knowledge)

            pass


if __name__ == "__main__":
    path_ = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/LivingNER/"

    parser = argparse.ArgumentParser(
        description='Preprocess data in brat format and save it in csv files. The files must be stored in a directory with the name "raw_data".')
    # parser.add_argument('path', help='Base path to data directory.')
    parser.add_argument('-ann', '--ann_labels',
                        default="off0,off1,label,span",
                        help="Names of the offset start, offset end, label and span columns. "
                             "Input must be the names sep by a coma. E.g: -ann start,end,label,word. Default is 'off0,off1,label,span'.")
    parser.add_argument('-tr', '--train_dir', default="data/train-set",
                        help='Directory where train data is stored. Default is "train-set".')
    parser.add_argument('-de', '--dev_dir', default="data/dev-set",
                        help='Directory where evaluation data is stored. Default is "dev-set".')
    parser.add_argument('-te', '--test_dir', default="data/test-set",
                        help='Directory where test data is stored. Default is "test-dev".')
    parser.add_argument('-kn', '--know_dir', default="data/knowledge/",
                        help='Directory where knowledge data is stored. Default is "knowledge".')
    parser.add_argument('-n', '--num_threads', type=int, default=8,
                        help='Number of threads generated to process the data. Default is 8')

    args = parser.parse_args()

    # path_ = args.path + '/' if args.path[-1] != '/' else args.path
    # path_ = args.path + '/'

    # Directory for data:           "data/text-files"
    # Directory for tsv data:       "data/subtask1-NER"
    # Directory for processed data: "data/processed"

    # dirs = [args.train_dir, args.dev_dir, args.test_dir]
    dirs = [args.train_dir, args.dev_dir]
    path_txt_types_ = []
    path_tsv_types_ = []
    save_df_path_ = []
    knowledge_dir = path_ + args.know_dir + "knowledge.tsv"

    for dir_ in dirs:
        path_txt_types_.append(path_ + dir_ + "/text-files/")
        path_tsv_types_.append(path_ + dir_ + "/subtask1-NER/")
        save_df_path_.append(path_ + dir_ + "/processed_knowledge/")

    num_threads = args.num_threads
    # num_threads = 8

    ann_labels_ = args.ann_labels.split(',')

    mkdir_processed(save_df_path_)

    # test(path_txt_types_, path_tsv_types_, save_df_path_, ann_labels_, knowledge_dir)
    # exit(0)

    process_data_parallel(path_txt_types_,
                          path_tsv_types_,
                          save_df_path_,
                          ann_labels_,
                          knowledge_dir,
                          num_threads,
                          )
