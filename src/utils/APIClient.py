import os
import requests
import json
import yaml
from urllib.parse import urlencode


##############################################
# Reuseable class for generic api to get data#
##############################################

class APIClient:
    def __init__(self, base_url, api_key_name):
        self.api_url = base_url
        self.api_key_name = api_key_name
        self.api_key_value = ''
        
        # Get current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go one folder up
        parent_dir = os.path.dirname(current_dir)
        self.config_file_path = os.path.join(parent_dir, 'config')
        self.config_file_path = os.path.join(self.config_file_path, 'applicationProperties.yml')
        print(self.config_file_path)
        

    def LoadAPIKey(self):
        try:
            with open(self.config_file_path, 'r') as file:
                data = yaml.safe_load(file)

            self.api_key_value = data[self.api_key_name]
            print(self.api_key_value)
        except FileNotFoundError:
            print(f"Error: The file '{self.config_file_path}' was not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        except KeyError as e:
            print(f"Error: Key '{e}' not found in YAML data.")

    def RequestBuilder(self, parameters):
        parameters['apikey'] = self.api_key_value
        self.api_url = f"{self.api_url}{urlencode(parameters)}"
        print("Full URL:", self.api_url)
        '''response = requests.get(self.api_url)
        print(response.json())'''

    
    def get(self):
        try:
            self.response = requests.get(self.api_url)
            self.response.raise_for_status()
            return self.response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None


APIClientObj = APIClient ('https://api.tomorrow.io/v4/weather/forecast?', 'TomorrowWeather.io')
APIClientObj.LoadAPIKey()
APIClientObj.RequestBuilder({'location' : '43.65107,-79.347015', 'timesteps' :'1h'})
APIResponse = APIClientObj.get()
file_path = r"F:\OneEnergy\OneEnergyPredictor\data\weathedata.json"
with open(file_path, 'w') as json_file:
    json.dump(APIResponse, json_file, indent=4)
#print(APIResponse)
