import argparse
import pathlib
import requests
import json
from urllib import parse
import ast


def get_labels(
    chems: list
) -> str:
    
    data = {'chemical_description': chems}
    query_string = parse.urlencode(data)
    url_labeller = 'http://kumo01:8080/topiclabeller/getLabels/'
    url_ = '{}?{}'.format(url_labeller, query_string)
  
    try:
        print("Getting labels...")
        resp = requests.post(
                url=url_,
                timeout=120,
                params={'Content-Type': 'application/json'}
            )
    except Exception as e:
        print(f"Exception when getting label: {e}")
        return ""
    return ast.literal_eval(resp.text)
    
    
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_tm', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/ewb_models/root_model_30_tpcs_20231028",
                        help="Path to the topic model to be labelled.")
    
    args = parser.parse_args()
    
    # Get chemical description from topic model
    #pathlib.Path(args.path_tm).joinpath("TMmodel/tpc_descriptions.txt"), "r")
    with open(path_tm, "r") as file:
        chems = [line.strip() for line in file]
        
    labels = get_labels(chems)
    
    
    
    