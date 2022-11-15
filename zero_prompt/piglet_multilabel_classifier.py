from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import fire
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import json

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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

def accuracy(y_true, y_pred):
    return accuracy_score(
        [[round(elt) for elt in ll] for ll in y_true],
        [[round(elt) for elt in ll] for ll in y_pred]
    )

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
        acc=accuracy,
    )
    print(result)
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

def main (
        checkpoint_dir='outputs-multilabel/multilabel_per_entity_state_change_all_attributes_roberta_large_2',
        best_model_dir='outputs-multilabel/multilabel_per_entity_state_change_all_attributes_roberta_large_best_2',
        model_name='roberta-large',
        n_epochs=30,
        learning_rate=4e-5,
        train_batch_size=20,
        eval_batch_size=64,
        evaluate_during_training=True,
        evaluate_during_training_steps=1000,
        evaluate_during_training_verbose=True,
        reprocess_input=True,
        overwrite_output_dir=True,
        n_gpu=2
    ):

    # Preparing train data
    train_df = pd.read_json(
        'preprocessed_piglet_data/train.json',
        orient='records'
    )
    eval_df = pd.read_json(
        'preprocessed_piglet_data/dev.json',
        orient='records'
    )
    test_df = pd.read_json(
        'preprocessed_piglet_data/test.json',
        orient='records'
    )

    # Optional model configuration
    model_args = MultiLabelClassificationArgs(
        num_train_epochs=n_epochs,
        evaluate_during_training=evaluate_during_training,
        evaluate_during_training_steps=evaluate_during_training_steps,
        evaluate_during_training_verbose=evaluate_during_training_verbose,
        best_model_dir=best_model_dir,
        manual_seed=0,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        overwrite_output_dir=overwrite_output_dir,
        n_gpu=n_gpu,
        output_dir=checkpoint_dir,
        reprocess_input_data=reprocess_input,
        learning_rate=learning_rate,
    )

    attributes_of_interest = train_df['label_names'][0]
    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        "roberta",
        model_name,
        num_labels=len(attributes_of_interest),
        args=model_args,
    )

    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        pos_f1=f1_score_factory(1),
        pos_p=precision_score_factory(1),
        pos_r=recall_score_factory(1),
        neg_f1=f1_score_factory(0),
        neg_p=precision_score_factory(0),
        neg_r=recall_score_factory(0),
        acc=accuracy,
    )

    # Evaluate the model
    test_model(model, eval_df, checkpoint_dir, 'dev')
    test_model(model, test_df, checkpoint_dir, 'test')
    
if __name__ == '__main__':
    fire.Fire(main)