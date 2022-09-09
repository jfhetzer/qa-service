import argparse
import os

from flask import Flask, jsonify, request
from dacite import from_dict
from dacite.exceptions import DaciteError
from dataclasses import dataclass
from typing import Optional, List

from inference import Inference


@dataclass
class Example:
    questions: List[str]
    context: str

    def __iter__(self):
        for question in self.questions:
            yield question, self.context


@dataclass
class Input:
    data: List[Example]
    impossible: Optional[bool] = True
    top_k: Optional[int] = 1
    max_ans_len: Optional[int] = 15

    def __iter__(self):
        for example in self.data:
            yield from example


# create flask service and initialize inference model
app = Flask(__name__)
infer = Inference()


@app.route('/inference', methods=['POST'])
def inference():
    if request.method == 'POST':
        # parse and validate request data
        input = from_dict(
            data_class=Input,
            data=request.json
        )

        # validate top_k and max_ans_len
        if input.top_k < 1 or input.max_ans_len < 1:
            raise DaciteError('The parameters top_k and max_ans_len have to be at least 1')

        # flatten questions and contexts and pass them into the inference model
        questions, contexts = zip(*input)
        answers = infer(questions, contexts, input.impossible, input.top_k, input.max_ans_len)

        # return flat answers as json
        return jsonify(answers)


@app.errorhandler(500)
def error_500(exception):
    o_exception = exception.original_exception
    if isinstance(o_exception, DaciteError):
        return jsonify({'error': 'Input Error', 'description': str(o_exception)}), 400
    return jsonify({'error': 'Unknown Server Error', 'description': str(exception)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start question answering REST API')
    parser.add_argument('-p', metavar='PORT', default=5000, help='the port flask is using')
    args = parser.parse_args()

    host = os.getenv('FLASK_HOST', '127.0.0.1')
    app.run(host=host, port=args.p)
