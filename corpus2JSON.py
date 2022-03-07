"""
Created on Mar 06 2022
@author: Jerónimo Arenas García

Temporary routine for generation of JSON files
simulating the creation of files required for the
BI tool

If other datasets want to be ingested, we need to
1) Change the SELECT statement accordingly
2) Change the fields as necessary
3) Change the name of the folder where the JSON will be save
"""

import TMinferencer
import io
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
    #We need to add the Inferncer to the list of .py files, so that
    #workers can use it
    sc.addPyFile("/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/TMinferencer.py")

    # Read subtable from parquet file
    S2papers = spark.sql("SELECT id, year, venue, title, paperAbstract FROM parquet.`/export/ml4ds/IntelComp/Datalake/SemanticScholar/20220201/papers.parquet` where array_contains(fieldsOfStudy, 'Computer Science')")

    #Concatenate text fields to lemmatize
    #This is not needed in general, but we do not have access to lemmas yet
    S2papers = (
        S2papers.withColumn("rawtext",concat_ws('. ', "title", "paperAbstract"))
        .drop("paperAbstract")
    )

    def my_detect(rawt):
        try:
            return detect(rawt)
        except:
            return "na"

    #Keep only papers in English language
    udf_detect = udf(lambda x:my_detect(x), StringType() )
    S2papers = S2papers.withColumn("language",udf_detect(col("rawtext")))
    S2papers = (
        S2papers.filter(col("language")=="en")
        .drop("language")
    )

    #Inference of topics for each document
    def TMinference(rawt):
        thetas = TMinferencer.main(io.StringIO(rawt.replace("\n", " ")))
        return " ".join("t"+str(el[0])+"|"+str(round(1000*el[1])) for el in thetas[0].items())
    udf_TMinference = udf(lambda x:TMinference(x), StringType() )

    S2papers = S2papers.withColumn("TM40",udf_TMinference(col("rawtext")))
    S2papers = S2papers.drop("rawtext")

    #Save final dataframe as JSON files
    S2papers.repartition(100).write.json(f"file:///export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/S2CS.json")




