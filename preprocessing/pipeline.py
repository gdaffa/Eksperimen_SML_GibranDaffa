from pipeline import extract_main, preproces_main, load_main
import json

CONFIG_PATH = 'config.json'

def get_config() -> dict:
   with open(CONFIG_PATH, 'r') as file:
      config = json.load(file)

   return config

def main():
   config = get_config()

   df = extract_main(config)
   splitted_data = preproces_main(config, df)
   load_main(config, splitted_data)

if __name__ == '__main__':
   main()