# health_data

This code supports both lightgbm and XGBoost predictors.

The dataset directory path should be passed by ``--database_path``. 

The logging path should also be provided for saving log and model. It is passed by ``--result_path``.

Sample code configuration is as follows:

```
python main.py -m XGBoost --database_path ... --result_path ...
```
or if you want to use lightgbm:
```
python main.py -m lightgbm --database_path ... --result_path ...
```
Please note that after each training, the model is automatically saved inside the directory result_path and the address of the saved model
is provided at the end of log file.

You can also set a seed by passing ``--seed``.

If you want to load a saved model you should write:

```
python main.py --load_mode --path_model ...
```
where ``--path_model`` is for passing the path of file containing the saved model. Please note that by default, the program is not on the load mode.

Finally, in the beggining of each run the configuration of the experiment containing all the argument is saved inside the log.
