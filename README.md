
The goal of this project is to implement the original Isolation Forest algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. (A later version of this work is also available: Isolation-based Anomaly Detection.) There are two general approaches to anomaly detection:

model what normal looks like and then look for nonnormal observations
focus on the anomalies, which are few and different. This is the interesting and relatively-new approach taken by the authors of isolation forests.
The isolation forest algorithm is original and beautiful in its simplicity; and also seems to work very well, with a few known weaknesses. The academic paper is extremely readable so you should start there.



**RESULTS**


Running noise=True improved=True
INFO creditcard.csv fit time 0.40s
INFO creditcard.csv 20348 total nodes in 300 trees
INFO creditcard.csv score time 19.60s
SUCCESS creditcard.csv 300 trees at desired TPR 80.0% getting FPR 0.0221%

INFO http.csv fit time 0.30s
INFO http.csv 15014 total nodes in 300 trees
INFO http.csv score time 17.19s
SUCCESS http.csv 300 trees at desired TPR 99.0% getting FPR 0.0165%

INFO cancer.csv fit time 0.13s
INFO cancer.csv 7992 total nodes in 1000 trees
INFO cancer.csv score time 0.79s
SUCCESS cancer.csv 1000 trees at desired TPR 75.0% getting FPR 0.2885%

