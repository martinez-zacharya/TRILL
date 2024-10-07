from loguru import logger
from icecream import ic
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import skops.io as sio
import multiprocessing
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_squared_error, make_scorer

def prep_reg_data(df, args):
    if args.train_split is not None:
        if args.key == None:
            logger.error('Training a regressor requires a value key CSV!')
            raise Exception('Training a regressor requires a value key CSV!')
        key_df = pd.read_csv(args.key)
        df = df.merge(key_df, on='Label', how='left')
        df = df.sample(frac=1)
        if not float(args.train_split) == 1 or not float(args.train_split) == 1.0:
            train_df, test_df = train_test_split(df, train_size=float(args.train_split))
            return train_df, test_df
        else:
            return df, df

def train_reg_model(train_df, args):
    # if args.regressor == 'LightGBM':
    #     d_train = lgb.Dataset(train_df.iloc[:, :-2], label=train_df['NewLab'])
    # elif args.regressor == 'Linear':
    #     d_train = xgb.DMatrix(train_df.iloc[:, :-2], label=train_df['NewLab'])
    
    config = {
        'lightgbm': {
            'metric': ['mae', 'rmse'],
            'learning_rate': args.lr,
            'max_depth': args.max_depth,
            'num_leaves': args.num_leaves,
            'num_threads': args.n_workers,
            'seed': args.RNG_seed,
            'verbosity' : -1
        }
    }
    
    # Model training
    if args.regressor == 'LightGBM':
        clf = LGBMRegressor(num_leaves=args.num_leaves, learning_rate=args.lr, max_depth = args.max_depth, num_threads=args.n_workers, seed=args.RNG_seed, metric=['mae', 'rmse']).fit(train_df.iloc[:, :-2], train_df['Score'])
    elif args.regressor == 'Linear':
        clf = LinearRegression().fit(train_df.iloc[:, :-2], train_df['Score'])
    # elif args.classifier == 'XGBoost':
    #     clf = xgb.train(config['xgboost'], d_train, args.n_estimators)
    
    return clf

def load_reg_model(args):
    # Check the model type and load accordingly
    if args.regressor == 'Linear':
        model = sio.load(args.preTrained, trusted=True)
    elif args.regressor == 'LightGBM':
        model = sio.load(args.preTrained, trusted=True)  
    else:
        logger.error("Unsupported model type. Please specify 'Linear' or 'LightGBM'.")
        raise ValueError("Unsupported model type. Please specify 'Linear' or 'LightGBM'.")

    return model


def predict_and_evaluate_reg(model, test_df, args):
    if args.regressor == 'LightGBM':
        test_preds = model.predict(test_df.iloc[:, :-2])
        # test_preds = np.argmax(test_preds, axis=0)
    elif args.regressor == 'Linear':
        test_preds = model.predict(test_df.iloc[:, :-2])
    r2 = r2_score(test_df.iloc[:, -1].values, test_preds)
    rmse = root_mean_squared_error(test_df.iloc[:, -1].values, test_preds)

    return r2, rmse

def custom_model_reg_test(model, test_df, args):
    # Generate probability predictions based on the model type
    model_type = args.regressor
    if model_type == 'Linear':
        test_preds = model.predict(test_df.iloc[:, :-1])
    elif model_type == 'LightGBM':
        test_preds = model.predict(test_df.iloc[:, :-1])
    
    # Prepare and save the predictions to a CSV file
    pred_df = pd.DataFrame(test_preds, columns=['Prediction'])
    pred_df['Label'] = test_df['Label']
    
    pred_file_name = f'{args.name}_{model_type}_predictions.csv'
    pred_df.to_csv(os.path.join(args.outdir, pred_file_name), index=False)

    return

def log_reg_results(out_file, command_str, args, rmse=None, r2=None, best_params=None):
    with open(out_file, 'w+') as out:
        out.write('TRILL command used: ' + command_str + '\n\n')
        
        if best_params:
            out.write('Best model parameters from tuning: \n')
            for param, value in best_params.items():
                out.write(f'\t{param}: {value}\n')
            out.write('\n')
        
        if rmse is not None:
            out.write(f'RMSE (Root Mean Squared Error): {rmse:.4f}\n')
        
        if r2 is not None:
            out.write(f'R² (Coefficient of Determination): {r2:.4f}\n')
        
        # If there are additional arguments to log, you can do so here. For example:
        if hasattr(args, 'other_metrics') and args.other_metrics:
            out.write('\nAdditional Metrics:\n')
            for metric, value in args.other_metrics.items():
                out.write(f'\t{metric}: {value}\n')

def sweep(train_df, args):
    model_type = args.regressor
    logger.info(f"Setting up hyperparameter sweep for {model_type}")
    np.int = np.int64
    if args.n_workers == 1:
        logger.warning("WARNING!")
        logger.warning("You are trying to perform a hyperparameter sweep with only 1 core!")
        logger.warning(f"In your case, you have {multiprocessing.cpu_count()} CPU cores available!")
    logger.info(f"Using {args.n_workers} CPU cores for sweep")
    # Define model and parameter grid based on the specified model_type
    if model_type == 'LightGBM':
        model = lgb.LGBMRegressor(metric=['mae'])
    param_grid = {
        'boosting_type': Categorical(['gbdt', 'dart']),
        'learning_rate': Real(0.01, 0.2),
        'num_leaves': Integer(20, 100),
        'max_depth': Integer(3, 15), 
        'min_split_gain': Real(0.0, 0.1), 
        'min_child_weight': Real(1e-5, 1e-3),  
        'n_estimators': Integer(100, 1000),
        'subsample_for_bin': Integer(50000, 300000),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(0.0, 1.0),
        'reg_lambda': Real(0.0, 1.0),
    }
    
    mse_scorer = make_scorer(mean_squared_error)


    clf = BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=100, n_jobs=int(args.n_workers),scoring=mse_scorer, cv=int(args.sweep_cv), return_train_score=True, verbose=False)
    
    logger.info("Sweeping...")
    clf.fit(train_df.iloc[:, :-2], train_df['Score'])
    
    # Save the best model
    if model_type == 'LightGBM':
        clf.best_estimator_.booster_.save_model(os.path.join(args.outdir, f'{args.name}_LightGBM-Regression_sweeped.json'))

    logger.info("Sweep Complete! Now evaluating...")
    
    return clf