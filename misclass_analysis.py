import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from wordcloud import WordCloud
import string
from collections import Counter
from nltk.corpus import stopwords

from libs.utils import read_json, anon, load_data

import os
from argparse import ArgumentParser
from pathlib import Path
import dotenv

def read_merged(exp_config):
    df = pd.read_csv('./data/raw/first.csv', index_col=0)
    df['anon_id'] = df.id_ingresso_ps.apply(lambda x: anon(x, os.getenv('SHA_SALT'), int(os.getenv('SHA_LEN'))))
    train_df = load_dataset(
            'json', 
            data_files=exp_config['train_path']
            )['train'].to_pandas()
    test_df = load_dataset(
            'json', 
            data_files=exp_config['test_path']
            )['train'].to_pandas()
    cohort = pd.concat([train_df, test_df], ignore_index=True)
    merged = df.drop(columns=['priorita','data_ingresso_ps']).merge(cohort.loc[:, ['anon_id', 'label']], how='right', on='anon_id')
    return merged

def plot_ds_volume(merged):
    num_pz = merged.groupby('id_paziente').label.max().value_counts()
    num_encounters = merged.groupby('id_ingresso_ps').label.max().value_counts()

    f, ax = plt.subplots()

    _ = num_encounters.plot(kind='bar', ax=ax)
    _ = ax.set_ylabel('# encounters')
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _ = ax.grid(axis='y', alpha=0.4)
    for i in num_pz.index:
        _ = ax.text(i, num_encounters.loc[i] / 2, f"unique pz:\n{num_pz.loc[i]}", ha='center', va='center')

    return f, ax

def plot_gender_distr(merged):
    df_gender = merged.groupby(['id_paziente','label']).sesso.max().reset_index()

    w = 0.3
    x = np.array([0,1])
    f, ax = plt.subplots()

    normalizer = df_gender.label.value_counts().sort_index()
    m_values = df_gender.query("sesso=='M'").label.value_counts().sort_index()
    m_height = (m_values / normalizer) * 100
    f_values = df_gender.query("sesso=='F'").label.value_counts().sort_index()
    f_height = (f_values / normalizer) * 100
    
    _ = ax.bar(x - w/2, height=m_height, width=w, label='Male')
    for i,(h,v) in enumerate(zip(m_height, m_values)):
        _ = ax.text(i - w/2, h/2, s=f"# patients:\n{v}", ha='center', va='center')
    
    _ = ax.bar(x + w/2, height=f_height, width=w, label='Female')
    for i,(h,v) in enumerate(zip(f_height, f_values)):
        _ = ax.text(i + w/2, h/2, s=f"# patients:\n{v}", ha='center', va='center')
    
    _ = ax.legend()
    _ = ax.set_xticks(x)
    _ = ax.set_xlabel('label')
    _ = ax.set_ylabel('[%]')
    _ = ax.grid(axis='y', alpha=0.4)
    _ = ax.set_title('Gender distribution')

    return f, ax

def plot_age_distr(merged):
    df_age = merged.groupby(['id_paziente','label']).eta.min().reset_index()

    f, ax = plt.subplots()
    x = np.linspace(0, 100, 50)

    _ = ax.hist(df_age.query("label==0").eta, bins=x, label='Class 0', alpha=0.6, density=True)
    _ = ax.hist(df_age.query("label==1").eta, bins=x, label='Class 1', alpha=0.6, density=True)

    _ = ax.legend()
    _ = ax.grid(alpha=0.4)
    _ = ax.set_xlabel('Age [years]')
    _ = ax.set_ylabel('[density]')
    _ = ax.set_title('Age distribution')

    return f, ax

def plot_missing_values(merged):
    cols = ['diagnosi_1_descr','ps_01_patologia_descrittiva','ps_03_anamnesi','ps_03_esame_obiettivo']
    df_missing = (merged.loc[:, cols].isna().sum() / merged.shape[0]) * 100

    f, ax = plt.subplots()
    _ = df_missing.plot(kind='bar', ax=ax)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    _ = ax.grid(axis='y', alpha=0.4)
    _ = ax.set_ylabel('[%]')
    _ = ax.set_title('Missing values percentage')

    return f, ax

def read_df(results_path, exp_config):
    if Path(exp_config['train_path']).parts[-1].split('_')[1] == 'reu':
        df_path = './data/raw/artr_reuma_drugs_grouped.csv'
    elif Path(exp_config['train_path']).parts[-1].split('_')[1] == 'psor':
        df_path = './data/raw/artr_psor_drugs_grouped.csv'
    df = load_data('./data/raw/first.csv', df_path)
    df['anon_id'] = df.id_ingresso_ps.apply(lambda x: anon(x, os.getenv('SHA_SALT'), int(os.getenv('SHA_LEN'))))
    df_preds = pd.read_csv(results_path / 'test_predictions.csv', index_col=0)
    # df_preds = pd.read_csv('results/20240719_144050/test_predictions.csv', index_col=0)
    df = df_preds.merge(df, how='left', on='anon_id')

    df = df.query("y_true==1")

    df.data_ingresso_ps = pd.to_datetime(df.data_ingresso_ps, yearfirst=True).dt.date
    df.DATA_ORA_ESEC = pd.to_datetime(df.DATA_ORA_ESEC, yearfirst=True).dt.date
    df['delta'] = df.apply(lambda r: (r.DATA_ORA_ESEC - r.data_ingresso_ps).days, axis=1)
    return df

def plot_temp_distr(df):
    f, axs = plt.subplots(1,2,figsize=(10,4))

    x = np.linspace(0,1500,20)
    _ = axs[0].hist(df.query("y_pred==0").delta, bins=x, label='False negative', alpha=0.5)
    _ = axs[0].hist(df.query("y_pred==1").delta, bins=x, label='True positive', alpha=0.5)
    _ = axs[0].grid(alpha=0.4)
    _ = axs[0].set_xlabel('# of days before the diagnosis')
    _ = axs[0].set_ylabel('# of encounters')
    _ = axs[0].legend()

    x = np.linspace(0,1,50)
    _ = axs[1].hist(df.query("y_pred==0").score, bins=x, label='False negative', alpha=0.5)
    _ = axs[1].hist(df.query("y_pred==1").score, bins=x, label='True positive', alpha=0.5)
    _ = axs[1].grid(alpha=0.4)
    _ = axs[1].set_xlabel('Probability score')
    _ = axs[1].legend()
    return f, axs

it_stopwords = stopwords.words('italian')
to_add = ['giunge','nega','riferisce','apr','dx','sx','fa']
_ = [it_stopwords.append(w) for w in to_add]

def get_counter(txt_series):
    c = Counter()
    table = str.maketrans(dict.fromkeys(string.punctuation))
    _ = txt_series.fillna('empty').str.lower().str.replace(r'[\r\n]+', ' ', regex=True).apply(lambda x: x.translate(table)).str.findall(r'\b\w+\b').apply(lambda x: c.update(x))
    _ = [c.pop(sw, None) for sw in it_stopwords]
    return c

def plot_wordclouds(df):
    f, axs = plt.subplots(1,2,figsize=(10,5))

    w = WordCloud(width=800, height=500, margin=10, random_state=10)
    c = get_counter(df.query("y_pred==0").ps_01_patologia_descrittiva)
    wc = w.fit_words(c, )
    _ = axs[0].imshow(wc)
    _ = axs[0].axis("off")
    _ = axs[0].set_title('False Negative wordcloud')

    w = WordCloud(width=800, height=500, margin=10, random_state=10)
    c = get_counter(df.query("y_pred==1").ps_01_patologia_descrittiva)
    wc = w.fit_words(c, )
    _ = axs[1].imshow(wc)
    _ = axs[1].axis("off")
    _ = axs[1].set_title('True Positive wordcloud')
    return f, axs

def main(results_path: Path):

    dotenv.load_dotenv()

    exp_config = read_json(results_path / 'exp_config.json')
    
    merged = read_merged(exp_config)
    f, ax = plot_ds_volume(merged)
    f.savefig(results_path / 'fig_df_volume.png')
    f, ax = plot_gender_distr(merged)
    f.savefig(results_path / 'fig_gender_distr.png')
    f, ax = plot_age_distr(merged)
    f.savefig(results_path / 'fig_age_distr.png')
    f, ax = plot_missing_values(merged)
    f.savefig(results_path / 'fig_missing_values.png')

    df = read_df(results_path, exp_config)
    f, axs = plot_temp_distr(df)
    f.savefig(results_path / 'fig_temp_distr.png')
    f, axs = plot_wordclouds(df)
    f.savefig(results_path / 'fig_wordclouds.png')
    

if __name__ == '__main__':
    argpars = ArgumentParser()
    argpars.add_argument('--result_path', '--p', default='./results/20240904_103950')
    args = argpars.parse_args()
    main(Path(args.result_path))
    