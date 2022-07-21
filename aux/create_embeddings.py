import argparse
import warnings
from pathlib import Path
from typing import List
import pandas as pd
import dask.dataframe as dd
from langdetect import detect
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from time import gmtime, strftime

class EmbeddingsManager(object):

    def _check_max_local_length(self, max_seq_length: int, texts: List[str]):
        """
        Returns a dictionary with all wordlists available in the folder 

        Parameters
        ----------
        max_seq_length : 
        texts:

        Returns
        -------
        allWdLists : Dictionary (path -> dictionary)
            One dictionary entry per wordlist
            key is the absolute path to the wordlist
            value is a dictionary with metadata
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
        texts : 
        sbert_model_to_load:
        batch_size:
        max_seq_length:
        
        Returns
        -------
        embeddings : 
        """

        model = SentenceTransformer(sbert_model_to_load)

        if max_seq_length is not None:
            model.max_seq_length = max_seq_length

        self._check_max_local_length(max_seq_length, texts)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size).tolist()
     
        return embeddings

    def add_embeddins_to_parquet(self, parquet_file: Path, parquet_new: Path, embeddins_model: str, max_seq_length: int) -> Path:

        path_parquet = Path(parquet_file)

        res = []
        for entry in path_parquet.iterdir():
            # check if it a file
            if entry.as_posix().endswith("parquet"):
                res.append(entry)

        def det(x):
            try:
                lang = detect(x)
            except:
                lang = 'Other'
            return lang

        for f in tqdm(res):
            df = pd.read_parquet(f)
            df = df[df.paperAbstract != ""]
            df['langue'] = df['paperAbstract'].apply(det)
            df = df[df.langue == 'en']
            df['title_abstract'] = df[["title", "paperAbstract"]].apply(" ".join, axis=1)
            raw = df.title_abstract.values.tolist()
            embeddings = self.bert_embeddings_from_list(
                    texts=raw, 
                    sbert_model_to_load=embeddins_model, 
                    max_seq_length=max_seq_length)
            df.drop('title_abstract', inplace=True, axis=1)
            df['embeddings'] = embeddings
            time = strftime("_%Y-%m-%d-%-S", gmtime())
            new_name = "parquet_embeddings" + time + ".parquet"
            outFile = parquet_new.joinpath(new_name)
            df.to_parquet(outFile)
            print(outFile)

        return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scripts for Embeddings Service")
    parser.add_argument("--path_parquet", type=str, default=None, required=True, metavar=("path_to_parquet"), help="path to parquet file to caclulate embeddings of")
    parser.add_argument("--path_new", type=str, default=None, required=True, metavar=("path_new"), help="path to parquet folder to locate embeddings")
    parser.add_argument("--embeddings_model", type=str, default="all-mpnet-base-v2", required=False, metavar=("embeddings_model"), help="Model to be used for calculating the embeddings")
    parser.add_argument("--max_sequence_length", type=int, default=384, required=False, metavar=("max_sequence_length"), help="Model's context")

    args = parser.parse_args()

    em = EmbeddingsManager()

    parquet_path = Path(args.path_parquet)
    parquet_new = Path(args.path_new)
    em.add_embeddins_to_parquet(parquet_path, parquet_new, args.embeddings_model, args.max_sequence_length)

