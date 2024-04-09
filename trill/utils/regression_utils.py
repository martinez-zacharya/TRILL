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

def prep_reg_data(df, args):
    if args.train_split is not None:
        if args.key == None:
            logger.error('Training a regressor requires a value key CSV!')
            raise Exception('Training a classifier requires a value key CSV!')
        key_df = pd.read_csv(args.key)
        df = df.merge(key_df, on='Label', how='left')
        df = df.sample(frac=1)
        train_df, test_df = train_test_split(df, train_size=float(args.train_split))
    return train_df, test_df

def train_reg_model(train_df, args):
    # if args.regressor == 'LightGBM':
    #     d_train = lgb.Dataset(train_df.iloc[:, :-2], label=train_df['NewLab'])
    # elif args.regressor == 'Linear':
    #     d_train = xgb.DMatrix(train_df.iloc[:, :-2], label=train_df['NewLab'])
    
    config = {
        'lightgbm': {
            'metric': ['mae, rmse'],
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
