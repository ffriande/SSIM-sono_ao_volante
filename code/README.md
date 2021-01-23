To run the code:
python3 main.py [option]

option can be:
--grid to run with gridSearch for hyperparameter tuning
--folds to perform cross validation
--train to read data an perform feature extraction



--train should be run with dataset, but the dataset is too large to include, but all the other options are working without this step.

In order to change parameters in grid search you should:
- Go to "config.py"
- Array "params" and change to which combination you want to run, in this case only for LightGBM and XGBoost



To try the different feature selection methods change variable "FT_SELECTION" in config.py to true or false. Where true uses the correlation and false the best parameters in random forest.

To change the number of sleep stages to predict change variable "NR_STAGES" in config.py to 3 or 4.
