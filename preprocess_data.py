from libs.utils import load_data, get_data, read_config
from libs.processing  import prepare_model, preprocess_function
from libs.paths import DATA_PATH

import dotenv
import os
import argparse

def main(filename):

    dotenv.load_dotenv()
    config = read_config(__file__)

    df_cls = load_data(filename, config['source_df']) 
    train_data, test_data = get_data(
        df_cls, config['txt_column'], 
        n_control=config['n_control'], to_anonimize=config['anonim_column'], 
        sha_salt=os.getenv('SHA_SALT'),
        sha_len=int(os.getenv('SHA_LEN')))
    
    _, tokenizer, _ = prepare_model(freeze_base=False)

    train_tokenized = train_data.map(lambda x: preprocess_function(x, tokenizer))
    test_tokenized = test_data.map(lambda x: preprocess_function(x, tokenizer))

    train_tokenized = train_tokenized.remove_columns('text')
    test_tokenized = test_tokenized.remove_columns('text')

    train_tokenized.to_json(config['train_save_path'])
    test_tokenized.to_json(config['test_save_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename",required=True)
    args = parser.parse_args()
    main(args.filename)