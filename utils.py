from os import listdir
from os.path import isfile, join
from datasets import load_dataset
import argparse

# Globals
PATH = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/Data/cantemist/"
TRAIN = "train-set/cantemist-ner-processed/"
DEV1 = "dev-set1/cantemist-ner-processed/"
DEV2 = "dev-set2/cantemist-ner-processed/"
TEST = "test-set/cantemist-ner-processed/"


def get_files_path(path):
    return [path + f for f in listdir(path) if isfile(join(path, f))]


def get_dataset(path, train_path, dev_path, test_path):
    files_train = get_files_path(path + train_path)
    files_dev = get_files_path(path + dev_path)
    files_test = get_files_path(path + test_path)

    datasets = load_dataset('csv', data_files={'train': files_train, 'validation': files_dev, 'test': files_test})

    return datasets


if __name__ == "__main__":
    # path_ = "C:/Users/carlo/Documents/Máster/Trabajo-Fin-De-Master/Data/cantemist/NER"

    parser = argparse.ArgumentParser(description='Preprocess data in brat format and save it in csv files. The files must be stored in a directory with the name "raw_data".')
    parser.add_argument('path', help='Base path to data directory.')
    parser.add_argument('-tr', '--train_dir', default="train-set/", help='Directory where train data is stored. Default is "train-set/".')
    parser.add_argument('-de', '--dev_dir', default="dev-set/", help='Directory where evaluation data is stored. Default is "dev-set/".')
    parser.add_argument('-te', '--test_dir', default="test-set/", help='Directory where test data is stored. Default is "test-dev/".')

    args = parser.parse_args()

    path_ = args.path + '/'
    save_df_train = args.train_dir + "processed/"
    save_df_dev = args.dev_dir + "processed/"
    save_df_test = args.test_dir + "processed/"

    print(get_dataset(path_, save_df_train, save_df_dev, save_df_test))
