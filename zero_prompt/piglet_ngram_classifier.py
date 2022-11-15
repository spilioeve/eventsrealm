# Adapted from https://github.com/openai/gpt-2-output-dataset/blob/master/baseline.py

import os
import json

import fire
import numpy as np
from scipy import sparse

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import f1_score, recall_score, precision_score

def flatten_round(ll):
    return [round(elt) for l in ll for elt in l]

def f1_score_factory(label):
    def f1_score_bin(y_true, y_pred):
        return f1_score(
            y_true, 
            y_pred, 
            pos_label=label, 
            average="binary",
            zero_division=0,
        )
    return f1_score_bin

def precision_score_factory(label):
    def precision_score_bin(y_true, y_pred):
        return precision_score(
            y_true, 
            y_pred, 
            pos_label=label, 
            average="binary",
            zero_division=0,
        )
    return precision_score_bin

def recall_score_factory(label):
    def recall_score_bin(y_true, y_pred):
        return recall_score(
            y_true, 
            y_pred, 
            pos_label=label, 
            average="binary",
            zero_division=0,
        )
    return recall_score_bin
    

def load_split(data_dir, split, n=None):
    with open(f'{data_dir}/{split}.json') as infile:
        data = json.loads(infile.read())
    texts = []
    labels = []
    if n is not None:
        data = data[:n]
    for elt in data:
        texts.append(elt['text'])
        labels.append(elt['labels'])
    names = data[0]['label_names']
    return texts, np.array(labels), names

def main(
    data_dir='preprocessed_piglet_data',
    log_dir='output_simple_classifier',
    n_train=None,
    n_valid=None,
    n_jobs=10,
    verbose=False,
):
    train_texts, train_labels, train_label_names = load_split(data_dir, 'train', n=n_train)
    valid_texts, valid_labels, valid_label_names = load_split(data_dir, 'dev', n=n_valid)
    test_texts, test_labels, test_label_names = load_split(data_dir, 'test')

    vect = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=2**21)
    train_features = vect.fit_transform(train_texts)
    valid_features = vect.transform(valid_texts)
    test_features = vect.transform(test_texts)

    model = ClassifierChain(LogisticRegression(solver='liblinear'))
    params = {'base_estimator__C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
    split = PredefinedSplit([-1]*len(train_labels)+[0]*len(valid_labels))
    search = GridSearchCV(model, params, cv=split, n_jobs=n_jobs, verbose=verbose, refit=False)
    search.fit(sparse.vstack([train_features, valid_features]), np.vstack([train_labels,valid_labels]))
    model = model.set_params(**search.best_params_)
    model.fit(train_features, train_labels)
    valid_accuracy = model.score(valid_features, valid_labels)*100.
    test_accuracy = model.score(test_features, test_labels)*100.
    data = {
        'n_train':n_train,
        'valid_accuracy':valid_accuracy,
        'test_accuracy':test_accuracy
    }


    predictions = model.predict(test_features)

    def test_model(predictions, labels, attribute_list, split):
        print(f'Starting evaluation on {split} set.')
        predictions
        metrics = {
            "pos_f1": f1_score_factory(1),
            "pos_p": precision_score_factory(1),
            "pos_r": recall_score_factory(1),
            "neg_f1": f1_score_factory(0),
            "neg_p": precision_score_factory(0),
            "neg_r": recall_score_factory(0),
        }
        scores = {}
        for metric_name, metric in metrics.items():
            scores[metric_name] = metric(predictions.flatten(), labels.flatten())
            for i, attribute in enumerate(attribute_list):
                scores[f'{attribute}_{metric_name}'] = metric(
                    predictions[:,i].flatten(),
                    labels[:,i].flatten()
                )
            

        out_data = []
        for i, true_labels in enumerate(labels):
            for j, label in enumerate(true_labels):
                out_data.append({
                    'true_label':label,
                    'predicted_label':predictions[i][j],
                    'attribute':attribute_list[j]
                })
        return scores
    
    scores = test_model(predictions, test_labels, test_label_names, 'test')
    data.update(scores)

    print(data)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    json.dump(data, open(os.path.join(log_dir, f'piglet_simple_classifier_results.json'), 'w'), indent=4)

if __name__ == '__main__':
    fire.Fire(main)