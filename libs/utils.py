import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import json
import pickle as pkl

from datasets import Dataset
from sklearn.model_selection import train_test_split

from hashlib import sha256
import os
from libs.paths import CONFIG_PATH, RESULTS_PATH
from libs.propensity_score import PropensityScoreMatcher
from datetime import datetime

def mkdir_experiment():
    exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = RESULTS_PATH / exp_name / 'mdl_checkpoints'
    os.makedirs(folder_name, exist_ok=True)
    return exp_name

def read_config(name):
    asd = Path(name).name.split('.')[0] + '.json'
    file_name = CONFIG_PATH / asd
    with open(file_name, 'r') as f:
        config = json.load(f)
    return config

def read_json(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def write_obj(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

def load_data(data_path, cohort_path):
    df = pd.read_csv(data_path, index_col=0)

    cohort = pd.read_csv(cohort_path)
    cohort['label'] = 1
    cohort = cohort.rename(columns={'ID_PAZIENTE':'id_paziente'})

    df_cls = df.merge(cohort.loc[:,['id_paziente','label','DATA_ORA_ESEC']], how='left', on='id_paziente')
    df_cls.label = df_cls.label.fillna(0)
    df_cls['to_drop'] = df_cls.apply(lambda r: 0 if type(r.DATA_ORA_ESEC) is not str else 1 if r.DATA_ORA_ESEC <= r.data_ingresso_ps[:10] else 0, axis=1)
    df_cls = df_cls.query("to_drop==0")
    return df_cls

def format_data(df):
    df.loc[:, 'text'] = df.text.str.replace(r'[\r\n]+', ' ', regex=True)
    df = df.loc[:, ['anon_id', 'text', 'label']].to_dict('records')
    
    ds = Dataset.from_list(df)
    return ds

def anon(uid, sha_salt, sha_len):
    hash = sha256(f"{uid}{sha_salt}".encode()).hexdigest()
    anon = int(hash[:sha_len], 16)
    return anon

def sample_and_split(df, n_control):
    df_pz = df.groupby('id_paziente').label.max().reset_index()
    df_pz = pd.concat(
        [
            df_pz.query("label==0").sample(n_control), 
            df_pz.query("label==1")
            ], 
        ignore_index=True
        )
    
    pz_train, pz_test = train_test_split(df_pz.id_paziente, test_size=0.3, stratify=df_pz.label)
    df_train = df[df.id_paziente.isin(pz_train)]
    df_test = df[df.id_paziente.isin(pz_test)]
    return df_train, df_test

def get_data(df, txt_col, n_control=5, to_anonimize=None, sha_salt=None, sha_len=None):
    df = df.rename(columns={txt_col: "text"})

    df['anon_id'] = df[to_anonimize].apply(lambda x: anon(x, sha_salt, sha_len))
    df.sesso = df.sesso.replace({'M':0, 'F':1})
    df['psm_target'] = df.priorita.apply(lambda x: 1 if x<3 else 0)
    
    # df_train, df_test = sample_and_split(df, n_control)
    psm = PropensityScoreMatcher(df)
    _, matched_df = psm.run(
        ['sesso','eta'], 'psm_target', 'label', n_control=n_control
    )
    df_pz = matched_df.groupby('id_paziente').label.max().reset_index()
    pz_train, pz_test = train_test_split(df_pz.id_paziente, test_size=0.3, stratify=df_pz.label)
    df_train = matched_df[matched_df.id_paziente.isin(pz_train)]
    df_test = matched_df[matched_df.id_paziente.isin(pz_test)]

    ds_train = format_data(df_train)
    ds_test = format_data(df_test)
    return ds_train, ds_test
