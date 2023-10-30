"""This is a temporary script to label the topics of a topic model trained via the topicmodeler. Given the path to a folder allocating a TMmodel, it will read the chemical descriptions from the 'tpc_descriptions.txt' file and will send them to the topic labeller service (kumo01:8080). The labels obtained will be saved to the 'tpc_labels.txt' file."""

import argparse
import pathlib
import requests
from urllib import parse
import ast


def get_labels(
    chems: list
) -> list:
    """Get labels from topic labeller service.

    Parameters
    ----------
    chems : list
        List of chemical descriptions.

    Returns
    -------
    list
        List of labels.
    """

    data = {'chemical_description': chems}
    query_string = parse.urlencode(data)
    url_labeller = 'http://kumo01:8080/topiclabeller/getLabels/'
    url_ = '{}?{}'.format(url_labeller, query_string)

    try:
        print("-- -- Getting labels...")
        resp = requests.get(
            url=url_,
            timeout=None,
            params={'Content-Type': 'application/json'}
        )
        print(f"Labels obtained: {resp.text}")
    except Exception as e:
        print(f"-- -- Exception when getting label: {e}")
        return ""
    return ast.literal_eval(resp.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_tm', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/ewb_models/root_model_30_tpcs_20231028",
                        help="Path to the topic model to be labelled.")

    args = parser.parse_args()

    # Get chemical description from topic model
    with open(
            pathlib.Path(args.path_tm).joinpath("TMmodel/tpc_descriptions.txt"), "r") as file:
        chems = [line.strip() for line in file]

    # Get labels from topic labeller
    labels = get_labels(chems)

    # Save labels to file
    with open(
            pathlib.Path(args.path_tm).joinpath("TMmodel/tpc_labels.txt"), "w") as file:
        file.writelines(label + "\n" for label in labels)

    print("-- -- Labels saved to file")
