# app/app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('models/lstm_model.h5') 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        likes_count_input = request.form['likesCount']
        try:
            # Ensure the input is a float
            likes_count = float(likes_count_input)
            prediction = model.predict(np.array([[likes_count]]))[0]
            return render_template('index.html', prediction=prediction, likes_count=likes_count)
        except ValueError:
            return render_template('index.html', prediction="Invalid input. Please enter a numeric value.", likes_count=None)
    return render_template('index.html', prediction=None, likes_count=None)

if __name__ == '__main__':
    app.run(debug=True)
