"""
Compute some additional measures when creating models such as memory or cpu usage
"""

import argparse
import datetime as DT
import multiprocessing as mp
import logging
import json
# import shutil
import sys
import time
import warnings
from getpass import getuser
from pathlib import Path
from subprocess import check_output

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())

from src.utils.mem_usage import Mem

warnings.filterwarnings(action="ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def mem_use(fname, processes, gpu=False):
    m = Mem(user=getuser(), processes=processes, gpu=gpu)
    m.proc_info(f"{fname}.txt")

def main(nw=0):
    # Create folder structure
    # TODO:
    models = Path("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/scalPreproc_nw" + str(nw))
    models.mkdir(parents=True, exist_ok=True)

    # Iterate configuration
    datasets = ["S2CS_1", "S2CS_3", "S2CS_10", "S2CS_30", "S2CS_100"]

    Preproc = {
        "min_lemas": 15,
        "no_below": 15,
        "no_above": 0.4,
        "keep_n": 100000,
        "stopwords": [
          "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/wordlists/english_generic.json",
          "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/wordlists/S2_stopwords.json",
          "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/wordlists/S2CS_stopwords.json"
        ],
        "equivalences": [
          "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/wordlists/S2_equivalences.json",
          "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/wordlists/S2CS_equivalences.json"
        ]
    }
    
    processes = ["python"]
    use_gpu = False

    for Dtset in datasets:

        # Save configuration
        model_path = models.joinpath(Dtset)
        model_path.mkdir(parents=True, exist_ok=True)
        model_stats = model_path.joinpath("stats")
        model_stats.mkdir(parents=True, exist_ok=True)

        #Save dataset json file
        DtsetConfig = model_path.joinpath(Dtset+'.json')
        parquetFile = "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/fromHDFS_scalability/" + Dtset + ".parquet"
        TrDtset = {
          "name": Dtset,
          "Dtsets": [
            {
              "parquet": parquetFile,
              "source": "Semantic Scholar",
              "idfld": "id",
              "lemmasfld": [
                "lemmas"
              ],
          "filter": ""
            }
          ]
        }
        with DtsetConfig.open('w', encoding='utf-8') as outfile:
            json.dump(TrDtset, outfile,
                      ensure_ascii=False, indent=2, default=str)

        #Save configuration file
        configFile = model_path.joinpath("trainconfig.json")
        train_config = {
            "name": Dtset,
            "description": "",
            "visibility": "Public",
            "trainer": "mallet",
            "TrDtSet": DtsetConfig.resolve().as_posix(),
            "Preproc": Preproc,
            "TMparam": {},
            "creation_date": DT.datetime.now(),
            "hierarchy-level": 0,
            "htm-version": None,
        }
        with configFile.open('w', encoding='utf-8') as outfile:
            json.dump(train_config, outfile,
                      ensure_ascii=False, indent=2, default=str)

        # Start memory usage script
        logger.info("-- Started measuring memory")
        thrd = mp.Process(target=mem_use, args=(model_stats.joinpath("mem_use"), processes, use_gpu))
        thrd.start()
        time.sleep(1)

        # Execute command
        cmd = f"python src/topicmodeling/topicmodeling.py --preproc --config {configFile.resolve().as_posix()} --nw {str(nw)}"
        logger.info(f"Running command '{cmd}'")
        
        t_start = time.perf_counter()
        check_output(args=cmd, shell=True)
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

    parser = argparse.ArgumentParser(description='Study scalability of text preprocessing using Dask')
    parser.add_argument('--nw', type=int, required=False, default=0,
                        help="Number of workers when preprocessing data with Dask. Use 0 to use Dask default")
    args = parser.parse_args()
    main(args.nw)
