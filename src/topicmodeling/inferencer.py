import argparse
import json
import os
import sys
from abc import abstractmethod
from pathlib import Path
from subprocess import check_output

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

from tm_utils import unpickler


class Inferencer(object):
    """
    Wrapper for a Generic Topic Model Inferencer. Implements the
    following functionalities

    """

    def __init__(self, inferConfig, logger=None):
        """
        Initilization Method

        Parameters
        ----------
        inferConfigFile: Json object
            Inference config file
        logger: Logger object
            To log object activity
        """

        self._inferConfig = inferConfig

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Inferencer')

        return

    def transform_inference_output(self, thetas32, multiplier):
        """Saves the topic distribution for each document in text format (topic|weight)

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents
        multiplier: int
            Factor by which the topic weights are multiplied
        """

        self._logger.info(
            '-- Inference: Saving the topic distribution for each document in text format')
        docs_repr = []
        for doc_id in tqdm(np.arange(thetas32.shape[0])):
            doc_repr = ""
            for tpc_id in np.arange(thetas32.shape[1]):
                freq = multiplier*thetas32[doc_id][tpc_id]
                doc_repr += str(tpc_id) + "|" + str(freq) + " "
            docs_repr.append([doc_id, doc_repr])
        df = pd.DataFrame(docs_repr, columns=["DocID", "TpcRepr"]).set_index(
            "DocID", drop=False)

        infer_path = Path(self._inferConfig["infer_path"])
        doc_topics_file_csv = infer_path.joinpath("doc-topics.csv")
        df.to_csv(doc_topics_file_csv, index=False)

        return

    def apply_model_editions(self, thetas32):
        """Load thetas file, apply model edition actions, and save it as a numpy array

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents
        """
        self._logger.info(
            '-- Inference: Applying model edition transformations')

        model_for_infer_path = Path(self._inferConfig["model_for_infer_path"])
        infer_path = Path(self._inferConfig["infer_path"])

        model_edits = model_for_infer_path.joinpath('TMmodel/edits.txt')
        if model_edits.is_file():
            with model_edits.open('r', encoding='utf8') as fin:
                for line in fin:
                    line_els = line.strip().split()
                    if line_els[0] == 's':
                        idx = [int(el) for el in line_els[1:]]
                        thetas32 = thetas32[:, idx]
                    elif line_els[0] == 'd':
                        tpc = int(line_els[1])
                        ntopics = thetas32.shape[1]
                        tpc_keep = [k for k in range(ntopics) if k != tpc]
                        thetas32 = thetas32[:, tpc_keep]
                        thetas32 = normalize(thetas32, axis=1, norm='l1')
                    elif line_els[0] == 'f':
                        tpcs = [int(el) for el in line_els[1:]]
                        thet = np.sum(thetas32[:, tpcs], axis=1)
                        thetas32[:, tpcs[0]] = thet
                        thetas32 = np.delete(thetas32, tpcs[1:], 1)
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        self._logger.info(thetas32.shape)  # nodcs*ntopics
        doc_topics_file_npy = infer_path.joinpath("doc-topics.npy")
        np.save(doc_topics_file_npy, thetas32)

        return

    @abstractmethod
    def predict(self):
        pass


class MalletInferencer(Inferencer):
    def __init__(self, inferConfigFile, logger=None):

        super().__init__(inferConfigFile, logger)

    def predict(self):
        """
        Performs topic inference utilizing a pretrained model according to Mallet
        """

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # A proper corpus should exist with the corresponding ipmortation pipe
        path_pipe = Path(
            self._inferConfig["model_for_infer_path"]).joinpath('modelFiles/import.pipe')
        if not path_pipe.is_file():
            self._logger.error(
                '-- Inference error. Importation pipeline not found')
            return

        # Holdout corpus should exist
        holdout_corpus = Path(
            self._inferConfig['infer_path']).joinpath("corpus.txt")
        if not holdout_corpus.is_file():
            self._logger.error(
                '-- Inference error. File to perform the inference on not found')
            return

        # Get inferencer
        inferencer = Path(
            self._inferConfig['model_for_infer_path']).joinpath("modelFiles/inferencer.mallet")

        # The following files will be generated in the same folder
        corpus_mallet_inf = \
            holdout_corpus.parent.joinpath('corpus_inf.mallet')
        doc_topics_file = holdout_corpus.parent.joinpath('doc-topics-inf.txt')

        # Get Mallet Path
        mallet_path = Path(
            self._inferConfig['TMparam']['mallet_path'])
        if not mallet_path.is_file():
            self._logger.error(
                f'-- -- Provided mallet path is not valid -- Stop')
            return

        # Import data to mallet
        self._logger.info('-- Inference: Mallet Data Import')

        cmd = mallet_path.as_posix() + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_pipe, holdout_corpus, corpus_mallet_inf)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Mallet failed to import data. Revise command')
            return

        # Get topic proportions
        self._logger.info('-- Inference: Inferring Topic Proportions')
        num_iterations = 100
        doc_topic_thr = 0

        cmd = mallet_path.as_posix() + \
            ' infer-topics --inferencer %s --input %s --output-doc-topics %s ' + \
            ' --doc-topics-threshold ' + str(doc_topic_thr) + \
            ' --num-iterations ' + str(num_iterations)
        cmd = cmd % (inferencer, corpus_mallet_inf, doc_topics_file)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- Mallet inference failed. Revise command')
            return

        # Get inferred thetas
        ntopics = \
            self._inferConfig['TMparam']['ntopics']
        cols = [k for k in np.arange(2, ntopics + 2)]
        thetas32 = np.loadtxt(doc_topics_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)

        super().apply_model_editions(thetas32)
        super().transform_inference_output(thetas32, 100)

        return


class SparkLDAInferencer(Inferencer):
    def __init__(self, inferConfigFile, logger=None):

        super().__init__(inferConfigFile, logger)

    def predict(self):
        return


class ProdLDAInferencer(Inferencer):
    def __init__(self, inferConfigFile, logger=None):

        super().__init__(inferConfigFile, logger)

    def predict(self):
        """
        Performs topic inference utilizing a pretrained model according to ProdLDA
        """

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # A proper pickle file containing the avitm model should exist
        path_pickle = Path(
            self._inferConfig["model_for_infer_path"]).joinpath('modelFiles/model.pickle')
        if not path_pickle.is_file():
            self._logger.error(
                '-- Inference error. Pickle with the AVITM model not found')
            return

        # Holdout corpus should exist
        holdout_corpus = Path(
            self._inferConfig['infer_path']).joinpath("corpus.parquet")
        if not os.path.isdir(holdout_corpus):
            self._logger.error(
                '-- Inference error. File to perform the inference on not found')
            return

        # Generating holdout corpus in the input format required by ProdLDA
        self._logger.info(
            '-- -- Inference: BOW Dataset object generation')
        df = pd.read_parquet(holdout_corpus)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]

        # Get avitm object for performing inference
        avitm = unpickler(path_pickle)

        # Prepare holdout corpus in avitm format
        ho_corpus = [el for el in df_lemas]
        ho_data = prepare_hold_out_dataset(
            ho_corpus, avitm.train_data.cv, avitm.train_data.idx2token)

        # Get inferred thetas matrix
        self._logger.info(
            '-- -- Inference: Getting inferred thetas matrix')
        thetas32 = np.asarray(
            avitm.get_doc_topic_distribution(ho_data))

        super().apply_model_editions(thetas32)
        super().transform_inference_output(thetas32, 100)

        return


class CTMInferencer(Inferencer):
    def __init__(self, inferConfigFile, logger=None):

        super().__init__(inferConfigFile, logger)

    def predict(self):
        """
        Performs topic inference utilizing a pretrained model according to CTM
        """

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # A proper pickle file containing the avitm model should exist
        path_pickle = Path(
            self._inferConfig["model_for_infer_path"]).joinpath('modelFiles/model.pickle')
        if not path_pickle.is_file():
            self._logger.error(
                '-- Inference error. Pickle with the CTM model not found')
            return

        # Get avitm object for performing inference
        ctm = unpickler(path_pickle)

        # Holdout corpus should exist
        holdout_corpus = Path(
            self._inferConfig['infer_path']).joinpath("corpus.parquet")
        if not os.path.isdir(holdout_corpus):
            self._logger.error(
                '-- Inference error. File to perform the inference on not found')
            return

        # Generating holdout corpus in the input format required by CTM
        self._logger.info(
            '-- -- Inference: CTM Dataset object generation')
        df = pd.read_parquet(holdout_corpus)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]
        corpus = [el for el in df_lemas]

        if not "embeddings" in list(df.columns.values):
            df_raw = df[["all_rawtext"]].values.tolist()
            df_raw = [doc[0].split() for doc in df_raw]
            unpreprocessed_corpus = [el for el in df_raw]
            embeddings = None
        else:
            embeddings = df.embeddings.values
            unpreprocessed_corpus = None

        ho_data = prepare_hold_out_dataset(
            hold_out_corpus=corpus,
            qt=ctm.train_data.qt, unpreprocessed_ho_corpus=unpreprocessed_corpus, embeddings_ho=embeddings)

        # Get inferred thetas matrix
        self._logger.info(
            '-- -- Inference: Getting inferred thetas matrix')
        thetas32 = np.asarray(
            ctm.get_doc_topic_distribution(ho_data))

        super().apply_model_editions(thetas32)
        super().transform_inference_output(thetas32, 100)

        return


##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference utilities')
    parser.add_argument('--config', type=str, default=None,
                        help="path to inference configuration file")
    parser.add_argument('--infer', action='store_true', default=False,
                        help="Perform inference according to config file")
    args = parser.parse_args()

    if args.infer:
        configFile = Path(args.config)
        if configFile.is_file():
            with configFile.open('r', encoding='utf8') as fin:
                infer_config = json.load(fin)

                if infer_config['trainer'] == 'mallet':
                    inferencer = MalletInferencer(infer_config)

                elif infer_config['trainer'] == 'sparkLDA':
                    inferencer = SparkLDAInferencer(infer_config)

                elif infer_config['trainer'] == 'prodLDA':
                    # Import necessary libraries for prodLDA
                    from neural_models.pytorchavitm.utils.data_preparation import \
                        prepare_hold_out_dataset

                    # Create inferencer object
                    inferencer = ProdLDAInferencer(infer_config)

                elif infer_config['trainer'] == 'ctm':
                    # Import necessary libraries for CTM
                    from neural_models.contextualized_topic_models.utils.data_preparation import \
                        prepare_hold_out_dataset

                    # Create inferencer object
                    inferencer = CTMInferencer(infer_config)

                inferencer.predict()
        else:
            sys.exit('You need to provide a valid configuration file')
