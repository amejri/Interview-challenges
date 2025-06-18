# Instructions

## How to run the code :
All information and parameters need to be specified in the config file.

You have to specify the following parameters:

- training_set : the path for the training set (Ex: data/drugs_train.csv)
- test_set : the path for the test set (Ex:data/drugs_test.csv)
- fe_set : the path for the feature engineering dataset provided based on the description (Ex : data/drug_label_feature_eng.csv)
- submission_file_path: The file path where to store the prediction (Ex: data/submission.csv)

Please keep all the remaining parameters unchanged.
They are set based on the study that you can find on the notebook in which I explain my approach and my strategy.

### Install environment

```
python3 -m venv manomano_env
source manomano_env/bin/activate
pip install -r requirements.txt
pip install pre-commit &&  pre-commit install
```
### Launch the pipline :

```
python main.py
```


## Explanation:

- The objective here is to predict the price for each drug which is a numerical (real) data and therefore we are facing a **regression supervised problem**.

    - the metrics to be based on are :

        - MSE as a loss
        - r2-score as to mesure the precision of the model

- EDA : The data exploratory allows me to distinguish between the differents types of variables.

The feature engineering set was helpful in order to add more feature. but it caused a problem of missing values espacially when I merge the both dataset. The Thing that I am proud of is how I exploit description  which is a textual data in order to fill missing values. The second thing that made me proud, was the feature engineering
where I tried from to description to add more relevant feature especially drug materials.

Feel free to find in the notebook, the way the best model (i.e XGBoost) was selected between a list of models and how to parameters was tuned.

In my opinion the potential improvement would be a deeper investigation in the description column in order to extract more feature and also to try a better way for feature selection.
