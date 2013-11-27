Predicting default for Shopify
==============================


Which factors predict the risk of incurring in financial distress?

The input dataset is taken from [Kaggle](http://www.kaggle.com/c/GiveMeSomeCredit), and is made of 10 variables and 150,000 samples.

We analyze this dataset with Hadoop 1.2.1 on Ubuntu 10.04, taking advantage of Hadoop Streaming.

Description of the files in the repository:
- The list of bash commands to setup HDFS and run MapReduce: runjob
- The script to be run during Map: mapper.py
- The script to be run during Reduce: reducer.py
- The dataset: data.txt
- The report of the analysis: report.pdf

Data processing:
- Map step: each row in data.txt is filtered from unreasonable values (decided a priori) and cleaned from missing values. Then, the -- ok
