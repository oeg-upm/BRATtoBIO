# Preprocess
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
