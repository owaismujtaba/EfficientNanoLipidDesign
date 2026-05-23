import pandas as pd
import os

import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
import hashlib
from joblib import Parallel, delayed
from info_gain import info_gain
import logging
import multiprocessing
import pdb

import warnings
warnings.filterwarnings('ignore')

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def drop_duplicate_features(dataset, logger, n_jobs=-1):
    logger.info("*********************************Dropping Duplicate Features *****************************************")
    
    # Step 1: Hash each column to a string representation
    def hash_column(col):
        # Convert series to bytes and hash
        return hashlib.md5(pd.util.hash_pandas_object(col, index=False).values).hexdigest()
    
    # Compute hashes in parallel
    column_hashes = Parallel(n_jobs=n_jobs)(delayed(hash_column)(dataset[col]) for col in dataset.columns)
    
    # Step 2: Identify duplicates using hashes
    seen = {}
    dup_features = []
    for col, h in zip(dataset.columns, column_hashes):
        if h in seen:
            dup_features.append(col)
        else:
            seen[h] = col
    
    # Step 3: Drop duplicates
    for i, f in enumerate(dup_features):
        print(f"{i+1}. {f}")
    
    dataset.drop(dup_features, axis=1, inplace=True)
    
    logger.info("*********************************Dropped Duplicate Features *****************************************")
    logger.info(f"Dataset Shape: {dataset.shape}")
    
    return dataset


def drop_less_information_gain_features(dataset, y, logger, threshold=0.05):
    '''
    Measures the reduction in entropy after the split  
    
    '''
    logger.info("*******************************Deleting less info_gain features*************************")
    less_ig_cols = []

    features = list(set(dataset.columns) - set([' Label']))
    for col in features:
        info_gain_value = info_gain.info_gain(dataset[col], y)
        if info_gain_value < threshold:
            less_ig_cols.append(col)
    i = 1
    

    logger.info("*******************************Less info_gain features Deleted*************************")

    dataset.drop(less_ig_cols, axis=1, inplace=True)
    logger.info(f"Dataset Shape: {dataset.shape}")

    return dataset


def drop_qasi_constant_features(dataset, logger):
    logger.info("************************************ Dropping Qasi Constant Features ****************************")
    dataset1 = dataset.copy()

    qasi_constant_filter = VarianceThreshold(threshold=0.05)

    qasi_constant_filter.fit(dataset1)

    qasi_support = qasi_constant_filter.get_support()

    qasi_constant_features = []
    features = dataset1.columns
    for i in range(len(qasi_support)):
        if qasi_support[i] == True:
            qasi_constant_features.append(features[i])

    
    dataset.drop(qasi_constant_features, axis=1, inplace=True)

    logger.info("************************************ Dropped Qasi Constant Features ****************************")
    logger.info(f"Dataset Shape: {dataset.shape}")
    return dataset



def _calc_desc_process(smile, return_dict):
    """Calculates descriptors in a separate process."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return_dict['res'] = {desc[0]: 0.0 for desc in Descriptors._descList}
        return
        
    res = {}
    blacklist = {'Ipc', 'BertzCT'}
    for name, func in Descriptors._descList:
        if name in blacklist:
            res[name] = 0.0
            continue
        try:
            res[name] = func(mol)
        except Exception:
            res[name] = np.nan
    return_dict['res'] = res

def get_rdkit_descriptors(smile):
    """Calculate RDKit descriptors with a hard timeout using multiprocessing"""
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(target=_calc_desc_process, args=(smile, return_dict))
    p.start()
    p.join(2.0) # 2 second timeout for the entire molecule
    
    if p.is_alive():
        print(f"    Timeout! Skipping some descriptors for smile: {smile[:30]}...")
        p.terminate()
        p.join()
        # Return zeros if it timed out completely
        return {desc[0]: 0.0 for desc in Descriptors._descList}
    
    # If the process finished successfully but returned nothing
    if 'res' not in return_dict:
        return {desc[0]: 0.0 for desc in Descriptors._descList}
        
    return return_dict['res']

def extract_rdkit_features(df):
    print("Generating RDKit descriptors for unique molecules...")
    desc_dfs = []
    
    for col in ['m1', 'm2', 'm3', 'm4']:
        print(f"Processing {col}...")
        unique_smiles = df[col].dropna().unique()
        print(f"  Found {len(unique_smiles)} unique SMILES for {col}")
        
        # Calculate descriptors only for unique smiles
        smiles_to_desc = {}
        t0 = time.time()
        for s in unique_smiles:
            smiles_to_desc[s] = get_rdkit_descriptors(s)
        print(f"  Time taken for {col}: {time.time() - t0:.2f} seconds")
            
        # Map descriptors back to the full dataset
        smiles_list = df[col].tolist()
        descriptors_list = [smiles_to_desc[s] for s in smiles_list]
        desc_df = pd.DataFrame(descriptors_list)
        desc_df.columns = [f"{col}_{c}" for c in desc_df.columns]
        desc_dfs.append(desc_df)
    
    X_desc = pd.concat(desc_dfs, axis=1)
    return X_desc

def extract_molar_ratios(df):
    print("Processing molar ratios...")
    def round_to_nearest(x, base):
        return base * round(x / base)
    
    df_temp = df.copy()
    df_temp['p1_b'] = df_temp['p1'].apply(lambda x: round_to_nearest(x, 5.0))
    df_temp['p2_b'] = df_temp['p2'].apply(lambda x: round_to_nearest(x, 5.0))
    df_temp['p3_b'] = df_temp['p3'].apply(lambda x: round_to_nearest(x, 5.0))
    df_temp['p4_b'] = df_temp['p4'].apply(lambda x: round_to_nearest(x, 0.25))
    
    p13_cats = list(range(0, 105, 5))
    p4_cats = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]
    
    ohe_dfs = []
    for p_col, cats in zip(['p1_b', 'p2_b', 'p3_b', 'p4_b'], [p13_cats, p13_cats, p13_cats, p4_cats]):
        df_temp[p_col] = pd.Categorical(df_temp[p_col], categories=cats)
        # We drop the first category to get exactly 65/66 features representing the composition
        ohe = pd.get_dummies(df_temp[p_col], prefix=p_col, drop_first=True) 
        ohe_dfs.append(ohe)
    
    X_ratios = pd.concat(ohe_dfs, axis=1)
    return X_ratios

def run_feature_generation(input_path, intermediate_path):
    set_seed(42)
    print(f"Loading raw data from {input_path}...")
    #pdb.set_trace()
    
    df = pd.read_csv(input_path)
    
    X_desc = extract_rdkit_features(df)
    X_ratios = extract_molar_ratios(df)
    
    X_all = X_desc.copy()
    print(f"Total initial features: {X_all.shape[1]}")
    
    X_all.fillna(0, inplace=True)
    
    X_all['target'] = df['y2']
    
    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    X_all.to_csv(intermediate_path, index=False)
    X_ratios.to_csv('data/molar_ratios.csv')
    print(f"Saved intermediate generated features to {intermediate_path}")
    

def run_preprocessing(intermediate_path, output_path):
    set_seed(42)
    #pdb.set_trace()
    print(f"Loading intermediate data from {intermediate_path}...")
    df = pd.read_csv(intermediate_path)
    print("Dataset Shape:", df.shape)
    X_ratios = pd.read_csv('data/molar_ratios.csv')
    y = df['target']
    X_all = df.drop(columns=['target'])
    
    # Feature Reduction
    print("Applying feature reduction...")
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    # Algorithm 1: Duplicate Feature Removal
    X_all = drop_duplicate_features(X_all, logger, n_jobs=-1)
    
    # Algorithm 2: Info Gain Removal (Threshold=0.05)
    X_ig = drop_less_information_gain_features(X_all, y, logger, threshold=0.05)

    # Algorithm 3: Quasi-Constant features (variance < 0.01)
    X_final = drop_qasi_constant_features(X_ig, logger)
    
    # Save the processed dataset
    final_df = pd.concat([X_final, X_ratios], axis=1)
    final_df = pd.concat([X_final, y.rename('target')], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to {output_path}")

def main():
    import os
    os.makedirs('data', exist_ok=True)
    
    Step 1: Feature generation (RDKit + Molar Ratios)
    #run_feature_generation('data/all_data.csv', 'data/intermediate_features.csv')
    
    # Step 2: Preprocessing (Feature Reduction)
    run_preprocessing('data/intermediate_features.csv', 'data/processed_data.csv')

if __name__ == '__main__':
    main()
