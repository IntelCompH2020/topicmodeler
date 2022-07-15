import argparse
import warnings
from pathlib import Path
from typing import List
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer


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

        embeddings = np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))

        return embeddings

    def add_embeddins_to_parquet(self, parquet_file: Path, embeddins_model: str, max_seq_length: int) -> Path:

        df = pd.read_parquet(parquet_file).fillna("")
        print(df.columns)
        df_raw = df[["rawtext"]].values.tolist()
        df_raw = [doc[0] for doc in df_raw]
        e = self.bert_embeddings_from_list(texts=df_raw, 
                                           sbert_model_to_load=embeddins_model, 
                                           max_seq_length=max_seq_length)
        df['embeddings'] = pd.Series(e)
        print(df.head)
        new_name = parquet_file.as_posix() + "_embeddings"
        df.to_parquet(new_name)

        return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scripts for Embeddings Service")
    parser.add_argument("--path_parquet", type=str, default=None, required=True, metavar=("path_to_parquet"), help="path to parquet file to caclulate embeddings of")
    parser.add_argument("--embeddings_model", type=str, default="all-mpnet-base-v2", required=False, metavar=("embeddings_model"), help="Model to be used for calculating the embeddings")
    parser.add_argument("--max_sequence_length", type=int, default=384, required=False, metavar=("max_sequence_length"), help="Maximum number of tokens that the model can look at a time simultaneously")

    args = parser.parse_args()

    em = EmbeddingsManager()

    parquet_path = Path(args.path_parquet)
    em.add_embeddins_to_parquet(parquet_path, args.embeddings_model, args.max_sequence_length)