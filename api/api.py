#!/home/t/anaconda3/bin/python

from flask import Flask, request, jsonify, abort
from punctuation import PunctuationRestore

punctuation = PunctuationRestore()

app = Flask(__name__)


@app.route('/punctuation', methods=['POST'])
def punctuation_restoration():
    print(request.json)
    if not request.json or not 'clear_text' in request.json:
        abort(400)

    # output_text = "Hello " + request.json['clear_text']

    try:
        output_text = punctuation.restore(request.json['clear_text'])
    except expression as identifier:
        output_text = 'Can not restore punctuation. Model error!. Please check'

    return jsonify({'output_text': output_text}), 200


if __name__ == '__main__':
    app.run(port=2203, debug=True)
