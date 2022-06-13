"""
Temporary routine for generation of datasets
for demonstration purposes

It is a "Fake Data Mediator" for use with the first
version of the Interactive Topic Model Trainer
"""

import argparse
import json
import os
from pathlib import Path
from langdetect import detect
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
import argparse

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("PythonSort")\
        .getOrCreate()

    parser = argparse.ArgumentParser(description='Script for importing datasets from HDFS')
    parser.add_argument('-p', help='path to main table that will be imported', type=str, required=True)
    parser.add_argument('-s', help='list of fields that will be imported', type=str, required=True)
    parser.add_argument('-d', help='path to dataset where the data will be stored', type=str, required=True)
    parser.add_argument('-f', help='String with filtering conditions', type=str, required=False)
    args = parser.parse_args()

    parquet_table = args.p
    selectFields = args.s
    path_dataset = args.d
    
    # We read the table with the output of NLP processes and identify id field 
    lemmas_table = parquet_table.split('.parquet')[0] + '_NLP.parquet'
    lemmas_df = spark.sql(f"SELECT * FROM parquet.`{lemmas_table}`")
    # Obtain the name of the column that has to be used as the main identifier for the corpus
    id_fld = [el for el in lemmas_df.columns if el in ['projectID', 'id', 'appln_id']][0]
    
    # We read the main table including selected fields and the identifier
    flds = [el.strip() for el in selectFields.split(',')]
    query = "SELECT " + id_fld + " AS id, " + (",").join(flds) + \
                    " FROM parquet.`" + parquet_table + "`"

    # Add filtering condition to SELECT clause if necessary
    if args.f:
        # This is not very smart. Used for being able to send arguments with
        # "'" to the spark job
        query += " WHERE " + args.f.replace("XxX","'").replace('SsS',' ')
    #with open("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/query.txt", 'w') as fout:
    #    fout.write(query)
    dataset = spark.sql(query)
    
    # Join tables
    lemmas_df = lemmas_df.withColumnRenamed(id_fld,"id")
    dataset = (dataset.join(lemmas_df, dataset.id ==  lemmas_df.id, "left")
                          .drop(lemmas_df.id)
                    )

    # Save dataset
    dataset.write.parquet(f"file://{path_dataset}",
        mode="overwrite",
    )
    
    """
    For testing purposes only
    df2 = spark.read.parquet(f"file://{path_dataset}")
    df2 = spark.sample(fraction=0.1)
    df2.write.parquet(f"file://{path_dataset}",
        mode="overwrite",
    )
    """
