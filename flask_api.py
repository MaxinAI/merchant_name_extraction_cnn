"""Flask Api Service"""

import argparse
import json

from flask import Flask, request
from waitress import serve

from merchant_extractor import MerchantExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="directory containing the dataset")
parser.add_argument('--host', default='0.0.0.0', help="server host")
parser.add_argument('--port', default='5000', help="server port")
parser.add_argument('--query_key', default='text', help="default query key name for GET request")
parser.add_argument('--model_dir', default='experiments/base_model', help="directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--batch_size', default=32, type=int,
                    help='size of the batch for processing several transaction strings together with nn model')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def info():
    if request.method == 'GET':
        text = request.args.get(args.query_key)
        results = string_cleaner.get_merchant(text)
        response = json.dumps(results)

    elif request.method == 'POST':
        data = json.loads(request.data.decode())
        results = string_cleaner.get_merchant(data)
        response = json.dumps(results)

    return response


if __name__ == '__main__':
    args = parser.parse_args()

    string_cleaner = MerchantExtractor(args.data_dir, args.model_dir, args.restore_file, args.batch_size)

    serve(app.wsgi_app, host=args.host, port=args.port, threads=True)

    # this will be used for development purposes instead of serve
    # app.run(host=args.port, port=args.port, debug=True)
