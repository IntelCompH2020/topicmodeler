"""
Created on Mar 05 2022
@author: José Antonio Espinosa Melchor
         Jerónimo Arenas García

Temporary routine for generation of CSV datasets
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
from pyspark.sql.functions import array_contains, concat_ws, col, udf
from pyspark.sql.types import StringType


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("PythonSort")\
        .getOrCreate()

    sc = spark.sparkContext
    
    #The path to the dataset is received as a parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', help='Path to dataset')
    args = parser.parse_args()
    
    #Paths to the config file for this dataset
    #And to the folder where the corpus will be stored
    current_path = Path(os.getcwd())
    path_dataset = current_path.joinpath(args.p)
    path_config = path_dataset.joinpath("config.json")
    path_csv = path_dataset.joinpath("CSV")

    #Read query from config file
    with open(path_config, 'r') as fin:
        datasetMeta = json.load(fin)

    query = datasetMeta["query"]
    print("query is: ", query)

    #################
    # This part needs to be changed after lemmas are available in the data lake
    #################
    # Read subtable from parquet file
    # We should normally use the query, but lemmas are not calculated yet
    # So this part will retrieve title and abstract instead (note this only
    # works for Semantic Scholar)
    S2papers = spark.sql(query.replace("lemmas", "title, paperAbstract"))

    #Concatenate text fields to lemmatize
    S2papers = (
        S2papers.withColumn("rawtext",concat_ws('. ', "title", "paperAbstract"))
        .drop("title")
        .drop("paperAbstract")
    )

    #We detect language, and keep only documents that can be identified as english
    def my_detect(rawt):
        try:
            return detect(rawt)
        except:
            return "na"

    udf_detect = udf(lambda x:my_detect(x), StringType() )
    S2papers = S2papers.withColumn("language",udf_detect(col("rawtext"))).select("id","rawtext","language")
    S2papers = (
        S2papers.filter(col("language")=="en")
        .drop("language")
    )

    #Lemmatize rawtext column
    S2papers = S2papers.withColumnRenamed("rawtext", "lemmas")
    #################
    # END This part needs to be changed after lemmas are available in the data lake
    #################
    
    #Save corpus
    #We leave it distributed in case some WP3 tools can benefit from it?
    #Alternatively the data mediator could return an object to iterate over
    S2papers.repartition(25).write.option("header",True).csv(f"file://{path_dataset.as_posix()}")

