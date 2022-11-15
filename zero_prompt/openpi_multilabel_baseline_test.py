from textwrap import indent
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import fire
from sklearn.metrics import f1_score, recall_score, precision_score
import json

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


with open('preprocessed_multilabel_openpi_data/attributes_51.txt') as infile:
    attributes_list = [line.strip().split(',')[0] for line in infile.readlines()]

def flatten_round(ll):
    return [round(elt) for l in ll for elt in l]

def f1_score_factory(label):
    def f1_score_bin(y_true, y_pred):
        return f1_score(
            flatten_round(y_true), 
            flatten_round(y_pred), 
            pos_label=label, 
            average="binary",
            zero_division=0,
        )
    return f1_score_bin

def precision_score_factory(label):
    def precision_score_bin(y_true, y_pred):
        return precision_score(
            flatten_round(y_true), 
            flatten_round(y_pred), 
            pos_label=label, 
            average="binary",
            zero_division=0,
        )
    return precision_score_bin

def recall_score_factory(label):
    def recall_score_bin(y_true, y_pred):
        return recall_score(
            flatten_round(y_true), 
            flatten_round(y_pred), 
            pos_label=label, 
            average="binary",
            zero_division=0,
        )
    return recall_score_bin

def f1_score_attr_factory(label):
    def f1_score_attr_bin(y_true, y_pred):
        f1_scores = {}
        for i in range(y_true.shape[1]):
            y_pred_i = [round(elt) for elt in y_pred[:, i]]
            score = f1_score(y_true[:,i], y_pred_i)
            f1_scores[attributes_list[i]] = score
        return f1_scores
    return f1_score_attr_bin

def test_model(model, df, checkpoint_dir, split):
    print(f'Starting evaluation on {split} set.')
    result, model_outputs, wrong_predictions = model.eval_model(
        df,
        pos_f1=f1_score_factory(1),
        pos_p=precision_score_factory(1),
        pos_r=recall_score_factory(1),
        neg_f1=f1_score_factory(0),
        neg_p=precision_score_factory(0),
        neg_r=recall_score_factory(0),
        pos_f1_attr = f1_score_attr_factory(1)
    )
    out_data = []
    attributes_of_interest = df['label_names'][0]
    for i, true_labels in enumerate(df['labels']):
        for j, label in enumerate(true_labels):
            out_data.append({
                'true_label':label,
                'predicted_label':round(model_outputs[i][j]),
                'attribute':attributes_of_interest[j]
            })
    with open(f'{checkpoint_dir}/predictions_{split}.json', 'w') as outfile:
        json.dump(out_data, outfile, indent=4)
    print(result)

def main (
        load_checkpoint_dir='openpi_output_multilabel_roberta_large_best',
        checkpoint_dir='openpi_output_multilabel_roberta_large_test',
        reprocess_input=True,
        overwrite_output_dir=True,
        n_gpu=2
    ):
    eval_df = pd.read_json(
        'preprocessed_multilabel_openpi_data/dev.json',
        orient='records'
    )
    test_df = pd.read_json(
        'preprocessed_multilabel_openpi_data/test.json',
        orient='records'
    )

    # Optional model configuration
    model_args = MultiLabelClassificationArgs(
        manual_seed=0,
        overwrite_output_dir=overwrite_output_dir,
        n_gpu=n_gpu,
        output_dir=checkpoint_dir,
        reprocess_input_data=reprocess_input,
    )

    attributes_of_interest = eval_df['label_names'][0]
    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        "roberta",
        load_checkpoint_dir,
        num_labels=len(attributes_of_interest),
        args=model_args,
    )

    test_model(model, eval_df, checkpoint_dir, 'dev')
    test_model(model, test_df, checkpoint_dir, 'test')

    # # Make predictions with the model
    # predictions, raw_outputs = model.predict(eval_df['text'].tolist())
    




if __name__ == '__main__':
    fire.Fire(main)