from textwrap import indent
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import fire
from sklearn.metrics import f1_score, recall_score, precision_score
import json
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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
    
# def test_model(model, df, checkpoint_dir, split):
#     print(f'Starting evaluation on {split} set.')
#     result, model_outputs, wrong_predictions = model.eval_model(
#         df,
#         pos_f1=f1_score_factory(1),
#         pos_p=precision_score_factory(1),
#         pos_r=recall_score_factory(1),
#         neg_f1=f1_score_factory(0),
#         neg_p=precision_score_factory(0),
#         neg_r=recall_score_factory(0),
#     )
#     out_data = []
#     attributes_of_interest = df['label_names'][0]
#     for i, true_labels in enumerate(df['labels']):
#         for j, label in enumerate(true_labels):
#             out_data.append({
#                 'true_label':label,
#                 'predicted_label':round(model_outputs[i][j]),
#                 'attribute':attributes_of_interest[j]
#             })
#     with open(f'{checkpoint_dir}/predictions_{split}.json', 'w') as outfile:
#         json.dump(out_data, outfile, indent=4)
#     print(result)

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
    # with open(f'{log_dir}/predictions_{split}.json', 'w') as outfile:
    #     json.dump(out_data, outfile, indent=4)
    return scores

def main (
        load_checkpoint_dir='/usr0/home/espiliop/pet/virtual_events/outputs-multilabel/multilabel_per_entity_state_change_all_attributes_roberta_large_best_2',
        checkpoint_dir='/usr0/home/espiliop/pet/virtual_events/outputs-multilabel/multilabel_per_entity_state_change_all_attributes_roberta_large_best_2_test',
        reprocess_input=True,
        overwrite_output_dir=True,
        n_gpu=1
    ):
    # eval_df = pd.read_json(
    #     '/usr0/home/espiliop/pet/virtual_events/data/multilabel-per-entity-state-change-all-attributes/dev.json',
    #     orient='records'
    # )
    test_df = pd.read_json(
        '/usr0/home/espiliop/pet/virtual_events/data/multilabel-per-entity-state-change-all-attributes/test.json',
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

    attributes_of_interest = test_df['label_names'][0]
    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        "roberta",
        load_checkpoint_dir,
        num_labels=len(attributes_of_interest),
        args=model_args,
    )

    # test_model(model, eval_df, checkpoint_dir, 'dev')
    predictions, _ = model.predict(test_df['text'].tolist())
   
    labels = test_df['labels'].tolist()
    attribute_list = test_df['label_names'][0]
    split = 'test'
    scores = test_model(np.array(predictions), np.array(labels), attribute_list, split)

    # # Make predictions with the model
    # predictions, raw_outputs = model.predict(eval_df['text'].tolist())

    print(scores)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    json.dump(scores, open(os.path.join(checkpoint_dir, f'eval_scores.json'), 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)