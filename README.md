
The goal of this project is to implement the original Isolation Forest algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. (A later version of this work is also available: Isolation-based Anomaly Detection.) There are two general approaches to anomaly detection:

model what normal looks like and then look for nonnormal observations
focus on the anomalies, which are few and different. This is the interesting and relatively-new approach taken by the authors of isolation forests.
The isolation forest algorithm is original and beautiful in its simplicity; and also seems to work very well, with a few known weaknesses. The academic paper is extremely readable so you should start there.


**Data sets**
For this project, we'll use three data sets:

Kaggle credit card fraud competition data set; download, unzip to get creditcard.csv

Get cancer data into cancer.csv by executing savecancer.csv that I provide.

http.zip; download, unzip to get http.csv.
