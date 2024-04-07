"""
Compute some additional measures when creating models such as memory or cpu usage
"""


import pandas as pd
import datetime as DT
import multiprocessing as mp
import logging
import json
import sys
import time
import warnings
from getpass import getuser
from itertools import product
from pathlib import Path
from random import choices
from typing import Union

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())
from src.utils.mem_usage import Mem
from src.topicmodeling.topicmodeling import (CTMTrainer, MalletTrainer,
                                             ProdLDATrainer, BERTopicTrainer)
from src.topicmodeling.manageModels import TMmodel

warnings.filterwarnings(action="ignore")

####################################################
################### TO BE FILLED ###################
####################################################
# Parameters
models = Path(
    "/export/usuarios_ml4ds/lbartolome/Repos/intelcomp_repos/topicmodeler/scalability/scalability_models")
mallet_path = Path(
    "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/src/topicmodeling/mallet-2.0.8/bin/mallet")
####################################################

# Fixed params
fixed_params = {
    "activation": "softplus",
    "alpha": 5.0,
    "batch_size": 64,
    "doc_topic_thr": 0.0,
    "dropout": 0.2,
    "dropout_in": 0.2,
    "dropout_out": 0.2,
    "hidden_sizes": (50, 50),
    "labels": "",
    "learn_priors": True,
    "lr": 2e-3,
    "mallet_path": mallet_path,
    "momentum": 0.99,
    "num_data_loader_workers": mp.cpu_count(),
    "num_threads": 4,
    "optimize_interval": 10,
    "reduce_on_plateau": False,
    "sbert_model_to_load": "paraphrase-distilroberta-base-v1",
    "solver": "adam",
    "thetas_thr": 0.003,
    "token_regexp": "[\\p{L}\\p{N}][\\p{L}\\p{N}\\p{P}]*\\p{L}",
    "topic_prior_mean": 0.0,
    "topic_prior_variance": None,
}
# Tuned params
tuned_params = {
    "ctm_model_type": None,
    "model_type": None,
    "ntopics": None,
    "num_epochs": None,
    "num_iterations": None,
    "num_samples": None,
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def mem_use(fname, processes, gpu=False):
    m = Mem(user=getuser(), processes=processes, gpu=gpu)
    m.proc_info(f"{fname}.txt")


def copy_sampled_corpus(
    model_type,
    corpusFile: Path,
    dataset: Path
):
    """Saves a sample of the original origCFile into selected corpusFile"""
    dataset_file = dataset / "corpus.parquet"
    if not dataset_file.exists():
        print(
            f"Error: File '{dataset_file.resolve().as_posix()}' does not exist.")
        return

    if model_type == "mallet":
        df = pd.read_parquet(dataset_file)
        texts = [el[0] for el in df[["bow_text"]].values.tolist()]
        with open(corpusFile, 'w', encoding='utf-8') as fout:
            id = 0
            for el in texts:
                fout.write(str(id) + ' 0 ' + el + '\n')
                id += 1
    else:
        df = pd.read_parquet(dataset_file)
        df.to_parquet(corpusFile)

def get_model_config(trainer, TMparam):
    """Select model configuration based on trainer"""
    if trainer == 'mallet':
        fields = ["ntopics",
                  "thetas_thr",
                  "mallet_path",
                  "alpha",
                  "optimize_interval",
                  "num_threads",
                  "num_iterations",
                  "doc_topic_thr",
                  "token_regexp"]

    elif trainer == 'sparkLDA':
        fields = []

    elif trainer == 'prodLDA':
        fields = ["n_components",
                  "thetas_thr",
                  "model_type",
                  "hidden_sizes",
                  "activation",
                  "dropout",
                  "learn_priors",
                  "lr",
                  "momentum",
                  "solver",
                  "num_epochs",
                  "reduce_on_plateau",
                  "batch_size",
                  "topic_prior_mean",
                  "topic_prior_variance",
                  "num_samples",
                  "num_data_loader_workers", ]

    elif trainer == 'ctm':
        fields = ["n_components",
                  "thetas_thr",
                  "model_type",
                  "ctm_model_type",
                  "hidden_sizes",
                  "activation",
                  "dropout_in",
                  "dropout_out",
                  "learn_priors",
                  "lr",
                  "momentum",
                  "solver",
                  "num_epochs",
                  "reduce_on_plateau",
                  "batch_size",
                  "topic_prior_mean",
                  "topic_prior_variance",
                  "num_samples",
                  "num_data_loader_workers", ]
    elif trainer == "bertopic":
        fields = ["ntopics",
                  "thetas_thr"]

    params = {"trainer": trainer, "TMparam": {t: TMparam[t] for t in fields}}
    return params


def train_model(train_config, corpusFile, embeddingsFile=None):
    """Train a model based on train_config, using corpusFile and embeddingsFile"""
    trainer = train_config["trainer"]
    TMparam = train_config["TMparam"]

    if trainer == 'mallet':
        trainer = MalletTrainer(**TMparam, logger=logger)

    # elif trainer == 'sparkLDA':
    #     sparkLDATr = sparkLDATrainer()

    elif trainer == 'prodLDA':
        trainer = ProdLDATrainer(**TMparam, logger=logger)

    elif trainer == 'ctm':
        trainer = CTMTrainer(**TMparam, logger=logger)

    elif trainer == "bertopic":
        trainer = BERTopicTrainer(**TMparam, logger=logger)

    if trainer == 'ctm' and train_config['hierarchy-level'] == 1:
        t_train = trainer.fit(corpusFile=corpusFile, embeddingsFile=embeddingsFile)
    else:
        t_train = trainer.fit(corpusFile=corpusFile)
    return t_train


def main():
    # Create folder structure
    # TODO:
    models.mkdir(parents=True, exist_ok=True)

    # Iterate configuration
    # Define values for tunable parameters
    topicModelTypes = ["mallet", "prodLDA", "ctm", "bertopic"]  # "sparkLDA"
    coherences = ["c_npmi", "c_v"]  # c_npmi is computed by default
    ########################################
    # TODO:
    number_topics = [5]#[5, 10, 15, 20, 30, 50, 75, 100, 150, 300]
    number_iterations = [1000]
    number_threads = [8]  # 2, 4,
    number_samples = [20]
    number_epochs = [10]
    model_types = ["prodLDA"]  # , "LDA"
    ctm_model_types = ["CombinedTM"]  # "ZeroShotTM"
    ########################################
    
    path_datasets = Path("/export/usuarios_ml4ds/lbartolome/Repos/intelcomp_repos/topicmodeler/scalability/TMmodels")
    datasets = [
        path_datasets / "sample_cordis_1",
        path_datasets / "sample_cordis_3"
        
    ]
    
    for dataset in datasets:

        for trainer in topicModelTypes[:3]:
            if trainer == "mallet":
                param_conf = (
                    {"ntopics": n,
                    "num_iterations": i,
                    "num_threads": t}
                    for n, i, t in product(number_topics, number_iterations, number_threads)
                )
                CFtype = "txt"
                processes = ["python3", "java"]
                use_gpu = False
            elif trainer == "prodLDA":
                param_conf = (
                    {"n_components": n,
                    "num_samples": s,
                    "num_epochs": e,
                    "model_type": m}
                    for n, s, e, m in product(number_topics, number_samples, number_epochs, model_types)
                )
                CFtype = "parquet"
                processes = ["python3"]
                use_gpu = True
            elif trainer == "ctm":
                param_conf = (
                    {"n_components": n,
                    "num_samples": s,
                    "num_epochs": e,
                    "model_type": m,
                    "ctm_model_type": t}
                    for n, s, e, m, t in product(number_topics, number_samples, number_epochs, model_types, ctm_model_types)
                )
                CFtype = "parquet"
                processes = ["python3"]
                use_gpu = True
            elif trainer == "bertopic":
                param_conf = (
                    {"ntopics": n}
                    for n in number_topics
                )
                CFtype = "parquet"
                processes = ["python3"]
                use_gpu = False
            else:
                logger.error("Not valid trainer")
                exit()

            for tuned_params in param_conf:
                
                # Set configuration
                TMparam = {**fixed_params, **tuned_params}
                train_config = get_model_config(trainer, TMparam)

                # Save configuration
                model_path = models.joinpath(
                    f"{trainer}_{'_'.join(f'{i}' for i in tuned_params.values())}_{DT.datetime.now().strftime('%Y%m%d')}")
                model_path.mkdir(parents=True, exist_ok=True)
                model_stats = model_path.joinpath("stats")
                model_stats.mkdir(parents=True, exist_ok=True)
                tuned_params_conf = model_stats.joinpath("tuned_params.json")
                with tuned_params_conf.open("w", encoding="utf-8") as fout:
                    tuned_params = {"trainer": trainer, **tuned_params}
                    json.dump(tuned_params, fout, ensure_ascii=False,
                            indent=2, default=str)
                config_file = model_path.joinpath("config.json")
                with config_file.open("w", encoding="utf-8") as fout:
                    json.dump(train_config, fout, ensure_ascii=False,
                            indent=2, default=str)
                tm = TMmodel(model_path.joinpath("TMmodel"))

                # Copy corpus to model path
                logger.info(f"Copying corpus to {model_path}")
                corpusFile = config_file.parent.joinpath(f"corpus.{CFtype}")
                copy_sampled_corpus(trainer, corpusFile, dataset)

                # Start memory usage script
                logger.info("-- Started measuring memory")
                thrd = mp.Process(target=mem_use, args=(
                    model_stats.joinpath("mem_use"), processes, use_gpu))
                thrd.start()
                time.sleep(1)

                # Start timer
                t_start = time.perf_counter()

                # Train model
                t_train = train_model(train_config, corpusFile)

                # Compute coherences
                coh_stats = []
                for c in coherences:
                    tc_start = time.perf_counter()
                    tm.calculate_topic_coherence(
                        metrics=[c],
                        only_one=True)
                    coh = tm._topic_coherence
                    tc_end = time.perf_counter()
                    coh_stats.append(
                        {"measure": c, "coherences": coh, "time": tc_end-tc_start, "training_time": t_train})
                with model_stats.joinpath(f"topic_coherence.json").open("w") as fout:
                    json.dump(coh_stats, fout, ensure_ascii=False,
                            indent=2, default=str)

                t_end = time.perf_counter()
                thrd.terminate()
                time.sleep(1)
                logger.info("-- Finished measuring memory")

                t_total = t_end - t_start
                logger.info(f"Total time --> {t_total}")
                print("\n")
                print("-" * 100)
                print("\n")


if __name__ == "__main__":
    main()
