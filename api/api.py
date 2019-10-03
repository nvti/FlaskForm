#!/home/t/anaconda3/bin/python

from flask import Flask, request, jsonify, abort

app = Flask(__name__)


@app.route('/punctuation', methods=['POST'])
def punctuation_restoration():
    print(request.json)
    if not request.json or not 'clear_text' in request.json:
        abort(400)

    output_text = "Hello " + request.json['clear_text']

    return jsonify({'output_text': output_text}), 200


if __name__ == '__main__':
    app.run(port=2203, debug=True)
