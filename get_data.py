import pandas as pd

import argparse
from libs.paths import DATA_PATH

# Proprietary library to access the database
from db_connect import KeyMaster

def read_query(path):
    with open(path, 'r') as f:
        q = f.read()
    return q

def main(query_path, filename):
    q = read_query(query_path)

    km = KeyMaster()
    con = km.get_connection('DWHPRO')

    df = pd.read_sql(q, con)
    df.to_csv(DATA_PATH / filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-f", "--filename", required=True)
    args = parser.parse_args()

    main(
        args.query,
        args.filename,
    )