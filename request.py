import argparse
import glob
import json
import requests
from pathlib import Path
import re


URL_TEMPLATE = 'http://127.0.0.1:{port}/inference'

REQUEST_PATH = './requests/'

BUILD_PATH = Path('./build')
BUILD_PATH.mkdir(exist_ok=True)


def request(port):
    # get url with given port
    url = URL_TEMPLATE.format(port=port)

    # search for predefined requests
    files = glob.glob(REQUEST_PATH + '*.json')
    print('Requests found:')
    print('\n'.join(files))

    # execute each request and save response in build folder
    for file in files:
        # build and send request
        name = file.split('/')[-1]
        match = re.match(r'request([0-9]+)\.json', name)
        number = match.group(1)
        print(f'\n##### {name} #####')
        with open(file, 'r') as f:
            req_json = json.load(f)
        print('Send request...')
        r = requests.post(url, json=req_json)

        # validate response
        if r.status_code != 200:
            print('Error: Server responded with ', r.status_code)
            print(r.text)
            continue

        # save predictions to build folder
        res_json = r.json()
        out = BUILD_PATH / f'response{number}.json'
        print('Save JSON response to ', out)
        with open(out, 'w') as f_out:
            json.dump(res_json, f_out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start question answering REST API')
    parser.add_argument('-p', metavar='PORT', default=5000, help='the port flask is using')
    args = parser.parse_args()
    request(args.p)
