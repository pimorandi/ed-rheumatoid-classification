from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
from torch import Tensor

from libs.weighted_trainer import WeightedTrainer, calculate_balance_weight
from libs.paths import RESULTS_PATH, DATA_PATH
from libs.eval import Evaluator
from libs.utils import write_json, read_config, mkdir_experiment
from libs.processing import prepare_model, compute_metrics

import os
from shutil import rmtree
import argparse


def main(train_path, test_path):

    config = read_config(__file__)
    
    model, tokenizer, data_collator = prepare_model(
        model_name=config["model_name"],
        freeze_base=False)

    train_tokenized = load_dataset(
        'json', 
        data_files=train_path
        )['train']
    test_tokenized = load_dataset(
        'json', 
        data_files=test_path
        )['train']

    experiment = mkdir_experiment()

    weights = None
    if config["use_balanced_weights"]:
        weights = calculate_balance_weight(train_tokenized['label'])
        weights = weights[1].reshape(1)
    training_args = TrainingArguments(
        output_dir = RESULTS_PATH / experiment / 'mdl_checkpoints',
        **config["train_arguments"]
    )
    trainer = WeightedTrainer(
        model=model,
        class_weights=weights,
        args=training_args,
        callbacks=[EarlyStoppingCallback(3)],
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    os.mkdir(RESULTS_PATH / experiment / 'best_model')
    trainer.save_model(RESULTS_PATH / experiment / 'best_model')
    rmtree(RESULTS_PATH / experiment / 'mdl_checkpoints')

    eval = Evaluator()
    train_res, test_res, best_thr = eval.run(trainer, train_tokenized, test_tokenized)

    _ = eval.save_results(train_res, RESULTS_PATH / experiment, prefix='train')
    _ = eval.save_results(test_res, RESULTS_PATH / experiment, prefix='test')

    exp_config = {}
    exp_config['train_args'] = config
    exp_config['calculated_threshold'] = best_thr
    exp_config['train_path'] = train_path
    exp_config['test_path'] = test_path
    _ = write_json(exp_config, RESULTS_PATH / experiment / 'exp_config.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", required=True)
    parser.add_argument("-te", "--test", required=True)
    args = parser.parse_args()
    main(
        args.train,
        args.test
    )
