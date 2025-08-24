Run this command on bash:

python data.py 
--input ./input.csv 
--output results.csv 
--timestamp Time
--train-start "2004-01-01 00:00" 
--train-end "2004-01-05 23:59"
--analysis-start "2004-01-01 00:00"
--analysis-end "2004-01-19 07:59"


Proposed Solution
The solution is a modular, command-line Python pipeline for automated anomaly detection in multivariate time series data, as specified by your hackathon. It preprocesses raw sensor/process data, robustly handles missing/invalid/constant data, uses unsupervised learning (PCA-based reconstruction error), and surfaces both anomaly scores and root-cause feature attributions. Training occurs only on a user-defined “normal” period; scoring is performed for a larger analysis period, with the output as a fully annotated CSV per requirements.

Key Features:

Self-contained Python module, easy to deploy/run from command line

Input: CSV with timestamp + numeric features (auto-detect/override timestamp column)

Output: CSV with all original columns + 8 new columns (anomaly score + top 7 contributing feature names)

Argument-driven workflow: start/end of training (normal) and scoring/analysis periods are configurable

Thorough error handling and logging
