from flask import Flask, request, render_template, redirect, url_for
import pickle
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Pickle model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Receive input from user (could be a video file, text, or some feature data)
        input_data = request.form['input']

        # Convert input into the format expected by the model (e.g., list of features)
        prediction = model.predict([[float(x) for x in input_data.split(',')]])

        # Render result page with the prediction
        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)