import multiprocessing
from os import listdir, mkdir
from os.path import isfile, join, exists
import pandas as pd
import argparse


def get_all_labels(path_dir, col_labels):
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


def read_txt(file: str) -> list[str]:
    with open(file, 'r', encoding="utf8") as file:
        text = file.read()

    return text


def read_tsv(file):
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


def process_text(text, ann):
    columns = ['tokens', 'labels']
    df = pd.DataFrame(columns=columns)
    ann = ann.sort_values(by=['off0']).reset_index(drop=True)
    labels_phrase = []
    phrase = []
    word = ""

    cont = 0
    label_index = 0
    for letter in text:
        # If we finish the labeled word, pass to next row in ann dataframe
        if cont == ann['off1'][label_index] and label_index < len(ann) - 1:
            label_index += 1

        # If letter is a point of \n we assume that a whole phrase is analyzed
        if letter == '.' or letter == '\n':
            # In other case, the point can be a numeric point so the phrase is not concluded
            is_int = False
            if cont + 1 < len(text):
                is_int = text[cont + 1].isdigit()

            if is_int:
                word += letter
            else:
                # Inser the last word in the phrase
                if word:
                    phrase.append(word)

                # Insert the final point of the phrase
                phrase.append('.')

                # If there is a blank line the code above will generate a phrase with only a point, like this: ['.'].
                # And we are not interested in adding this point to the data
                if phrase != ['.']:
                    # Add the label for the point and insert the row in the main datafame
                    labels_phrase.append('O')
                    new_df = pd.DataFrame({columns[0]: [phrase], columns[1]: [labels_phrase]})
                    df = pd.concat([df, new_df])
                    labels_phrase = []

                phrase = []
                word = ""

        # If there is a blank space we consider that the word is already formed, and it is added to the phrase
        # The condition 'and word' avoid the blank space at the beginning of a phrase
        elif letter == ' ' and word:
            phrase.append(word + ' ')
            word = ""

        # This is the case in which a new word is gone be formed
        else:
            if cont == ann['off0'][label_index]:
                # We are starting a labeled word
                labels_phrase.append('B-' + ann['label'][label_index])

            elif cont in range(ann['off0'][label_index], ann['off1'][label_index]) and not word:
                # We are already forming a labeled word
                labels_phrase.append('I-' + ann['label'][label_index])

            elif cont not in range(ann['off0'][label_index], ann['off1'][label_index]) and not word and letter != ' ':
                # We are forming a non labeled word.
                # The condition "letter != ' '" avoid the blank space at the beginning of a phrase
                labels_phrase.append('O')

            word += letter

        cont += 1

    return df


def write_csv(path_file: str, df: pd.DataFrame):
    df.to_csv(path_file, index=False, header=True, encoding="utf-8")


def process_all_files(path_txt, df_tsv, path_save):
    id_thread = multiprocessing.current_process().name
    print(f"Starting thread {id_thread}...")

    file = df_tsv['filename'][0] + '.txt'
    text = read_txt(path_txt + file)

    df = process_text(text, df_tsv)

    write_csv(path_save + file + '.csv', df)

    print(f"Thread {id_thread} finished.")


def mkdir_processed(paths):
    for path in paths:
        if not exists(path):
            mkdir(path)
            print(f"Created --> \n{path }")


def process_data_sec(txt_path_types, tsv_path_types, save_df_path):
    for path_txt, path_tsv, path_save in zip(txt_path_types, tsv_path_types, save_df_path):
        files_tsv = [f for f in listdir(path_tsv) if isfile(join(path_tsv, f))]
        df_tsv = pd.DataFrame()
        for file in files_tsv:
            df_tsv = pd.concat([df_tsv, read_tsv(path_tsv + file)]).reset_index(drop=True)

        txt_files = df_tsv['filename'].unique()
        for txt in txt_files:
            df_aux = df_tsv.loc[df_tsv['filename'] == txt].reset_index(drop=True)
            process_all_files(path_txt, df_aux, path_save)


def process_data_parallel(txt_path_types, tsv_path_types, save_df_path, numthreads=8):
    pool = multiprocessing.Pool(numthreads)
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
                                                 path_save))

    pool.close()
    pool.join()


def test2():
    path_ = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/LivingNER/sample_set/"
    path_txt_types_ = [path_ + "text-files/"]
    path_tsv_types_ = [path_ + "subtask1/"]
    save_df_path_ = [path_ + "processed/"]

    mkdir_processed(save_df_path_)

    process_data_sec(path_txt_types_, path_tsv_types_, save_df_path_)

    # process_data_parallel(path_txt_types_, path_tsv_types_, save_df_path_, numthreads=8)

    # df = read_tsv(path_ + "subtask1/sample_entities_subtask1.tsv")
    # df_files = df['filename'].unique()
    # df_1 = df.loc[df['filename'] == df_files[0]].reset_index(drop=True)


def test():
    path_ = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/Data/cantemist/NER/train-set/raw_data/"
    labels = get_all_labels(path_)
    label_map = {i: label for i, label in enumerate(labels)}
    label2id_ = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)
    print(labels)
    print(label_map)
    print(label2id_)
    print(num_labels)


if __name__ == "__main__":
    test2()
    path_ = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/Data/cantemist/NER"

    parser = argparse.ArgumentParser(description='Preprocess data in brat format and save it in csv files. The files must be stored in a directory with the name "raw_data".')
    parser.add_argument('path', help='Base path to data directory.')
    parser.add_argument('-tr', '--train_dir', default="train-set", help='Directory where train data is stored. Default is "train-set".')
    parser.add_argument('-de', '--dev_dir', default="dev-set", help='Directory where evaluation data is stored. Default is "dev-set".')
    parser.add_argument('-te', '--test_dir', default="test-set", help='Directory where test data is stored. Default is "test-dev".')
    parser.add_argument('-n', '--num_threads', type=int, default=8,
                        help='Number of threads generated to process the data. Default is 8')

    args = parser.parse_args()

    path_ = args.path + '/'
    train_data = args.train_dir + "/raw_data/"
    dev_data = args.dev_dir + "/raw_data/"
    test_data = args.test_dir + "/raw_data/"
    save_df_train = args.train_dir + "/processed/"
    save_df_dev = args.dev_dir + "/processed/"
    save_df_test = args.test_dir + "/processed/"
    num_threads = args.num_threads

    # types_path = [train_data, dev_data, test_data]
    types_path = [train_data]
    # save_df_types = [save_df_train, save_df_dev, save_df_test]
    save_df_types = [save_df_train]

    mkdir_processed(path_, save_df_types)

    process_data_parallel(path_, types_path, save_df_types, num_threads)
