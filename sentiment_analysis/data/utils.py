import os
import csv
import pandas as pd

def preprocess_dataset(directory, output_file):
    texts = []
    labels = []

    for label in ['pos', 'neg']:
        dir_name = os.path.join(directory, label)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)

    df = pd.DataFrame({'label': labels, 'text': texts})
    df.to_csv(output_file, index=False, header=False)
