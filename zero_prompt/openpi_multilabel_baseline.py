from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pandas as pd
import logging
import fire
from sklearn.metrics import f1_score, recall_score, precision_score

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
    

def main (
        checkpoint_dir='openpi_output_multilabel_roberta_large',
        best_model_dir='openpi_output_multilabel_roberta_large_best',
        model_name='roberta-large',
        n_epochs=20,
        learning_rate=1e-5,
        train_batch_size=32,
        eval_batch_size=32,
        evaluate_during_training=True,
        evaluate_during_training_steps=1000,
        evaluate_during_training_verbose=True,
        reprocess_input=True,
        overwrite_output_dir=True,
        n_gpu=2
    ):

    # Preparing train data
    train_df = pd.read_json(
        '/usr0/home/espiliop/pet/real_events/data/gold-v1.1-multilabel/train.json',
        orient='records'
    )
    eval_df = pd.read_json(
        '/usr0/home/espiliop/pet/real_events/data/gold-v1.1-multilabel/dev.json',
        orient='records'
    )
    test_df = pd.read_json(
        '/usr0/home/espiliop/pet/real_events/data/gold-v1.1-multilabel/test.json',
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
    )

    # Evaluate the model
    print('Starting evaluation on dev set.')
    result, model_outputs, wrong_predictions = model.eval_model(
        eval_df,
        pos_f1=f1_score_factory(1),
        pos_p=precision_score_factory(1),
        pos_r=recall_score_factory(1),
        neg_f1=f1_score_factory(0),
        neg_p=precision_score_factory(0),
        neg_r=recall_score_factory(0),
    )
    print(result)
    print('Starting evaluation on test set.')
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df,
        pos_f1=f1_score_factory(1),
        pos_p=precision_score_factory(1),
        pos_r=recall_score_factory(1),
        neg_f1=f1_score_factory(0),
        neg_p=precision_score_factory(0),
        neg_r=recall_score_factory(0),
    )
    print(result)

# # Make predictions with the model
# predictions, raw_outputs = model.predict(["Sam"])


if __name__ == '__main__':
    fire.Fire(main)