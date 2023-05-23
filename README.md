# Preprocess BRAT to BIO

## Table of Contents

- [Usage](#usage)
- [Requirements](#requirements)

This python file contains the necessary code to convert annotated data in Brat format to BIO format.

The files must be in a specific structure. The directory dataset must contain at least one set (train, dev, test).
Each set directory must contain two main directories:
    1. First one with the text files (raw data)
    2. Second one with the file with the annotation. Only one file with all the annotations. The structure of this file
       is as follows (each column is sep by a tab, so the pandas.DataFrame is readed as a tsv):

            filename    mark  label    off0    off1    span
            f1_example  T1	  HUMAN    112     118	  hombre
            f1_example  T2	  HUMAN    1025    1033	  paciente
            f2_example  T1	  HUMAN    112     118	  mujer
            f3_example  T1	  SPECIE   1025    1033	  coronavirus

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

# Requirements

The requirements are already in the requirements.txt file. These are:

    pandas==1.5.3
    pysbd==0.3.4
    tqdm==4.64.1

# Usage

    python preprocess.py <path-to-dataset> -tr <train-set-dir-name> \
    -de <dev-set-dir-name> -te <test-set-dir-name> -txt <text-files-dir-name> \
    -an <annotations-dir-name> -hd <list-of-columns> -sn <output-dir> \
    -n <number-of-process>

Example from the dataset tree in above:

- Dataset: "/home/carlos/datasets/cantemist"
    - Train: "train-set"
    - Evaluation: "dev-set"
    - Test: "test-set"
- Text files: "text-files"
- Annotations: "cantemist-ner"
- Header: "filename,mark,label,off0,off1,span"
- Output file: "processed_data",
- Number of processes: "8"

Example command:

    python preprocess.py /home/carlos/datasets/cantemist -tr train-set \
    -de dev-set -te test-set -txt text-files -an cantemist-ner \
    -hd filename,mark,label,off0,off1,span -sn processed_data -n 8

