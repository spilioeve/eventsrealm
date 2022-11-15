#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import json
#from importlib_metadata import distribution
from numpy import require
import torch
import torch.nn as nn

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPromptMasking,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sklearn.metrics import f1_score, recall_score, precision_score

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


class MLM_ClassifierTrainer(Trainer):
    def __init__(self, label_token_ids_tensor, attr_distribution, attribute_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_token_ids_tensor = label_token_ids_tensor
        self.attr_distribution = attr_distribution
        self.attribute_loss = attribute_loss
        
        
        #dataset.set_format(type=dataset.format["type"], columns=list(dataset.features.keys()))
    # Link: https://discuss.huggingface.co/t/specify-loss-for-trainer-trainingarguments/10481
    # They get the logits and estimate loss on them. Maybe we can get logits AND we also need the labels as
    # global arguments?? 
    
    # Custom loss for our Trainer
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        #outputs = model(**inputs)        

        outputs = model(input_ids= inputs['input_ids'], attention_mask= inputs['attention_mask'])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # modify outputs logits and loss. We only keep the logits corresponding to the two labels.
        # import pdb; pdb.set_trace()
        mlm_logits = outputs['logits']
        outputs['logits'] = mlm_logits[:, :, self.label_token_ids_tensor]
        # distribution: pos = 16558, neg = 86334, 102892
        # 3.1, 0.6
        weight = torch.tensor([3.1, 0.6]).to(device=mlm_logits.device)


        # Define loss (TODO: do it only once instead of every time we compute the loss)
        # loss_fn = nn.CrossEntropyLoss(
        #     ignore_index=-100, 
        #     weight=weight,
        #     label_smoothing=self.args.label_smoothing_factor,
        #     reduction='none'
        # )
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, 
            label_smoothing=self.args.label_smoothing_factor,
            reduction='none'
        )
        # Compute loss (no reduction)
        loss = loss_fn(
            outputs['logits'].view(-1, len(self.label_token_ids_tensor)), 
            labels.view(-1)
        )

        # indices = labels != -100
        indices = torch.nonzero(
            torch.where(
                labels.view(-1) > -100, 
                torch.ones(len(labels.view(-1)), dtype=int).cuda(),
                torch.zeros(len(labels.view(-1)), dtype=int).cuda()
                )
            )
        
        # Attribute weight 
        if self.attribute_loss:
            attr_queries = inputs.pop("attribute_queries")
            for i, index in enumerate(indices):
                loss[index] = loss[index] * self.attr_distribution[attr_queries[i]]
            # TODO: turn everything into a tensor so that we avoid unwanted bugs... 
            # ... they are always there when you least expect them to be!
            # loss[indices] = loss[indices] * self.attr_distribution[torch.tensor(attr_queries)]
        
        loss = torch.mean(loss[indices])
        outputs['loss'] = loss

        return (loss, outputs) if return_outputs else loss


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # When using line_by_line, we just tokenize each nonempty line.
    padding = "max_length" if data_args.pad_to_max_length else False
    label_verbalizations = ['different', 'unchanged']
    label_token_ids = {
        # 'changed': tokenizer.convert_tokens_to_ids('Ġchanged'),
        # 'stable': tokenizer.convert_tokens_to_ids('Ġstable')
        # 'different': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' different'))[0],
        # 'unchanged': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' unchanged'))[0]
        verbalization: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f' {verbalization}'))[0] \
            for verbalization in label_verbalizations
    }
    label_index = {l:i for i,l in enumerate(label_verbalizations)}
    label_token_ids_tensor = torch.tensor(
        [label_token_ids[verbalization] for verbalization in label_verbalizations], 
        requires_grad=False
    )
    

    def tokenize_function(examples):
        # Join the data in a single 'text' column.
        # query0, template0, template1, query1, teamplate0 ...
        # template will have attribute masked. but we will keep it in a labels column.
        # import pdb; pdb.set_trace()
        mask_token = tokenizer.mask_token
        prompt = "The {} of the {} is {}. "
        texts = []
        texts_unmasked = []
        labels = []
        attribute_queries = []

        for i in range(len(examples['queries'])):
            text = ''
            text_unmasked = ''
            label = []
            queries = examples['queries'][i]
            for query in queries:
                attribute = query['attribute']
                text += f'{query["query"]} '
                text_unmasked += f'{query["query"]} '
                for entity, attribute_changed in zip(query['entities'], query['attribute_changed']):
                    text += prompt.format(attribute, entity, mask_token)
                    text_unmasked += prompt.format(attribute, entity, 'different' if attribute_changed else 'unchanged')
                    label.append(label_index['different'] if attribute_changed else label_index['unchanged'])
                    attribute_queries.append(attribute)
            texts.append(text)
            texts_unmasked.append(text_unmasked)
            labels.append(label)

        # # Remove empty lines
        # examples[text_column_name] = [
        #     line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        # ]
        # import pdb; pdb.set_trace()
        return {
            'texts':texts,
            'texts_unmasked':texts_unmasked,
            'labels':labels,
            'attribute_queries':attribute_queries,
            **tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        }

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        
        def preprocess_logits_for_metrics(logits, labels):
            # Note this is called after compute_loss.
            # import pdb; pdb.set_trace()
            # print(logits)
            return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            """
            actual eval loop at line 2429:
            transformers/src/transformers/trainer.py
            """
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]

            if len(preds) != len(eval_dataset["attribute_queries"]):
                raise Exception("Error: Predictions are not from the same eval dataset!")

            with open(f'{training_args.output_dir}/predictions.jsonl', 'w') as outfile:
                for l, p in zip(labels, preds):
                    outfile.write(f"{json.dumps({'label': int(l), 'pred': int(p)})}\n")

            metric_dict_output = metric.compute(predictions=preds, references=labels)

            score_names = ['f1', 'prec', 'rec']

            for label in label_verbalizations:
                for i, score in enumerate([
                        f1_score_factory(label_index[label]),
                        precision_score_factory(label_index[label]),
                        recall_score_factory(label_index[label])]
                    ):
                    computed_score = score(labels, preds)
                    print(f'{label}_{score_names[i]}', computed_score)
                    metric_dict_output[f'{label}_{score_names[i]}'] = computed_score

                    # New code here
            label = label_verbalizations[0]
            score = f1_score_factory(label_index[label])
            for attribute in set(eval_dataset["attribute_queries"]):
                indexes = [i for i, a in enumerate(eval_dataset["attribute_queries"]) if a == attribute]
                attribute_score = score(labels[indexes], preds[indexes])
                metric_dict_output[f'{label}_f1_{attribute}'] = attribute_score

            return metric_dict_output
            
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForPromptMasking(
        tokenizer=tokenizer,
    )

    with open('attr_distribution.json') as f:
        attr_distribution = [json.loads(line) for line in f.readlines()]
    total = sum(i["train"] for i in attr_distribution)
    attr_distribution = {i["attr"]: max(min(total/(i["train"]*len(attr_distribution)), 3.0), 0.1) for i in attr_distribution}
    

    trainer = MLM_ClassifierTrainer(
        label_token_ids_tensor=label_token_ids_tensor,
        model=model,
        attr_distribution=attr_distribution,
        attribute_loss= False,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
