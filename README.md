# health_data

This code supports both lightgbm and XGBoost predictors.

The dataset directory path should be passed by --database_path. 

The logging path should also be provided for saving log and model. It is passed by --result_path.

Sample code configuration is as follows:

```
python main.py -m XGBoost --database_path ... --result_path ...
```
Please note that after each training, the model is automatically saved inside the directory result_path and the address of the saved model
is provided ath the end of log file.

You can also set a seed by passing --seed.

If you want to load a saved model you should write:

```
python main.py --load_mode --path_model ...
```
where path model is the file containing the saved model.

Finally, in the beggining of each run the configuration of each experiment containing all the argument is saved inside the log.
