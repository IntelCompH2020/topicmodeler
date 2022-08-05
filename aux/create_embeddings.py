import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from langdetect import detect
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingsManager(object):

    def _check_max_local_length(self, max_seq_length: int, texts: List[str]):
        """
        Returns a dictionary with all wordlists available in the folder 

        Parameters
        ----------
        max_seq_length: int
            Context of the transformer model used for the embeddings generation
        texts: list[str]
            The sentences to embed
        """

        max_local_length = np.max([len(t.split()) for t in texts])
        if max_local_length > max_seq_length:
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                          f"truncates to {max_seq_length} tokens.")
        return

    def bert_embeddings_from_list(self, texts: List[str], sbert_model_to_load: str, batch_size=32, max_seq_length=None) -> np.ndarray:
        """
        Creates SBERT Embeddings from a list

        Parameters
        ----------
        texts : list[str]
            The sentences to embed
        sbert_model_to_load: str
            Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
        batch_size: int (default=32)
            The batch size used for the computation
        max_seq_length: int
            Context of the transformer model used for the embeddings generation

        Returns
        -------
        embeddings: list
            List with the embeddings for each document
        """

        model = SentenceTransformer(sbert_model_to_load)

        if max_seq_length is not None:
            model.max_seq_length = max_seq_length

        self._check_max_local_length(max_seq_length, texts)
        embeddings = model.encode(
            texts, show_progress_bar=True, batch_size=batch_size).tolist()

        return embeddings

    def add_embeddins_to_parquet(self, parquet_file: Path, parquet_new: Path, embeddins_model: str, max_seq_length: int, source: str) -> Path:
        """Generates the embeddings for a set of files given in parquet format, and saves them in a new parquet file that containing the original data plus an additional column named 'embeddings'.

        Parameters
        ----------
        parquet_file : Path
            Path from which the source parquet files are read
        parquet_new: Path
            Path in which the new parquet files will be saved
        embeddins_model: str
            Model to be used for generating the embeddings
        max_seq_length: int
            Context of the transformer model used for the embeddings generation
        source: str
        """

        path_parquet = Path(parquet_file)

        res = []
        for entry in path_parquet.iterdir():
            # check if it is a file
            if entry.as_posix().endswith("parquet"):
                res.append(entry)
        
        print("Number of parquet to process: " + str(len(res)))

        def det(x):
            try:
                lang = detect(x)
            except:
                lang = 'Other'
            return lang

        if source == "scholar":
            raw_text_fld = "paperAbstract"
            title_fld = "title"
        elif source == "cordis":
            raw_text_fld = "rawtext"
            title_fld = "title"
        elif source == "patstat":
            raw_text_fld = "appln_abstract"
            title_fld = "appln_title"
            pass

        for i, f in enumerate(tqdm(res)):
            df = pd.read_parquet(f)#.fillna("")

            # Filter out abstracts with no text
            df = df[df[raw_text_fld] != ""]
            print("Number of papers without empty abstract: " + str(len(df)))

            # Detect abstracts' language and filter out those which are not in English
            df['langue'] = df[raw_text_fld].apply(det)
            df = df[df['langue'] == 'en']

            print("Number of papers in english: " + str(len(df)))

            # Concatenation of title and abstract is used as the text to generated of the embeddings
            df['title_abstract'] = df[[title_fld, raw_text_fld]].apply(
                " ".join, axis=1)
            raw = df['title_abstract'].values.tolist()
            embeddings = self.bert_embeddings_from_list(
                texts=raw,
                sbert_model_to_load=embeddins_model,
                max_seq_length=max_seq_length)

            # Remove unnecessay columns in the df
            df.drop('title_abstract', inplace=True, axis=1)
            df['embeddings'] = embeddings

            # Save new df in parquet file
            new_name = "parquet_embeddings_part_" + str(i) + ".parquet"
            outFile = parquet_new.joinpath(new_name)
            df.to_parquet(outFile)
            print(outFile)

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scripts for Embeddings Service")
    parser.add_argument("--path_parquet", type=str, default=None,
                        required=True, metavar=("path_to_parquet"),
                        help="path to parquet file to caclulate embeddings of")
    parser.add_argument("--source", type=str, default=None,
                        required=True, metavar=("source"),
                        help="Key of the source dataset for which the embeddings are being calculated, 'cordis', 'scholar', 'patents'.")
    parser.add_argument("--path_new", type=str, default=None,
                        required=True, metavar=("path_new"),
                        help="path to parquet folder to locate embeddings")
    parser.add_argument("--embeddings_model", type=str,
                        default="all-mpnet-base-v2", required=False,
                        metavar=("embeddings_model"),
                        help="Model to be used for calculating the embeddings")
    parser.add_argument("--max_sequence_length", type=int, default=384,
                        required=False, metavar=("max_sequence_length"), help="Model's context")

    args = parser.parse_args()

    em = EmbeddingsManager()

    parquet_path = Path(args.path_parquet)
    parquet_new = Path(args.path_new)
    em.add_embeddins_to_parquet(
        parquet_path, parquet_new, args.embeddings_model, args.max_sequence_length, args.source)
