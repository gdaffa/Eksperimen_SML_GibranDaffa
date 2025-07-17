from pipeline import extract_main, preproces_main, load_main
import argparse
import json

config = {}

METADATA_PATH = 'metadata.json'

def get_config() -> dict:
   arg_parser = argparse.ArgumentParser()

   arg_parser.add_argument('dataset')
   arg_parser.add_argument('--target-col', default='price')
   arg_parser.add_argument('--outlier-treshold', default=1.5, type=float)

   args   = arg_parser.parse_args()
   config = dict(args._get_kwargs())

   with open(METADATA_PATH, 'r') as file:
      config['metadata'] = json.load(file)

   return config

def main():
   config = get_config()

   df = extract_main(config)
   splitted_data = preproces_main(config, df)
   load_main(config, splitted_data)

if __name__ == '__main__':
   main()