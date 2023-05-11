# `aux` folder
This folder contains all the code that is not part of the Topic Modeling toolbox, but was developed by the UC3M team at IntelComp to replicate functionality that will come from other components, but are not yet available.

   - Subfolder `lemmatization`: Contains notebooks for the lemmatization of the main data sets that were used to test the functionality of the toolbox. Notebooks are provided to lemmatize Cordis / PATSTAT / NIH / Semantic Scholar. Paths to the HDFS / NFS location of these datasets are hard-coded in the notebooks, and lemmatization of the main text fields of each corpus is carried out using Spark NLP. As a result of the process, new tables are created, where each table consists of rows with three columns, `id`, `rawtext`, `lemmas`. This is the minimum structure that is required by Topic Modelers, though additional columns may be present. Lemmas are necessary to train LDA using mallet or Spark MLLIB, whereas the rawtext is necessary for the Neural Network topic modeling tools.

   - Subfolder `fromHDFS`: Contains a script to download datasets from the UC3M data space to the a local repository (specified when launching the script)

