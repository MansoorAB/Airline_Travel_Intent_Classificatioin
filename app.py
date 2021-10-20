from flask import Flask, render_template, request, jsonify
import os
import numpy as np

from prediction_service import prediction

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    '''
        For rendering results on HTML GUI
    '''

    print('1. Mansoor: ', request.method, request.form, request.json)
    try:
        if request.form:
            print('2. in request.form')
            query = request.form['query']
            response = prediction.form_response(query)
            print('6. ', response)
            return render_template("index.html", prediction_text=response)
        elif request.json:
            print('2. in request.json')
            query = request.json['query']
            response = prediction.api_response(query)
            print('6. ', response)
            return jsonify(response)

    except Exception as e:
        print(e)
        # error = {"error": "Something went wrong!! Try again later!"}
        error = {"error": e}
        return render_template("404.html", error=error)

if __name__ == "__main__":

    app.run(host='127.0.0.1', port=5000, debug=True)
    # app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(debug=True)