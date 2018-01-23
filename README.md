# ML_W2017_Ass3

## Requirements
python3

Install/type this (with admin rights):

`pip3 install sklearn`

`pip3 install pandas`

`pip3 install liac-arff`

`pip3 install matplotlib`

## How to run

Generate hyper-parameter search results for each dataset and estimator (output is generated in ./@experiment@_hpsearch/*):
(To customize the search space see util.py)

`python3 hp_search.py <config name>.json`

To run the experiments (results are generated in ./@experiment@/*):

`python3 main.py <config name>.json`

For a detailed description of the system, see Section 2 of the report.
