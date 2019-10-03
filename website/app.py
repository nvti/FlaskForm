#!/home/t/anaconda3/bin/python

from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Required
import csv
import requests

app = Flask(__name__)
app.config['DEBUG'] = True

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'

# Flask-Bootstrap requires this line
Bootstrap(app)

PUNC_API_URL = "http://localhost:2203/punctuation"


class SubmitForm(FlaskForm):
    clear_text = StringField(
        'Enter a text without punctuation', validators=[Required()])
    submit = SubmitField('Submit')


# all Flask routes below

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SubmitForm()
    message = ""

    if form.validate_on_submit():
        clear_text = form.clear_text.data

        headers = {
            'Content-Type': 'application/json'
        }

        json = {
            'clear_text': clear_text
        }
        response = requests.post(
            PUNC_API_URL, headers=headers, json=json)

        try:
            message = response.json()['output_text']
        except:
            message = 'Can not connect to Punctuation restoration service: ' + PUNC_API_URL

    return render_template('index.html', form=form, message=message)


# keep this as is
if __name__ == '__main__':
    app.run()
