# topicmodeler
Topic Modeling tools for IntelComp H2020 project

#Pending tasks:
1. Class topicmodeling.MalletTrainer can probably benefit from Spark. Function preproc iterates over the CSV files that constitute the folder, and this preprocessing can be easily parallelized (though in the end the construction of the vocabulary and creation of the final corpus needs to be centralized). -> JAEM
2. 
