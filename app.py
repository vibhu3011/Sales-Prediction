import os
from flask import Flask, request, render_template
import pickle
import numpy as np

# Add the directory containing xgboost.dll
os.add_dll_directory(r"C:\sales prediction\.venv\Lib\site-packages\xgboost\lib")

app = Flask(__name__)

# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract valid input fields (matching model's expected 8 features)
        prev_sales = float(request.form['prev_sales'])
        comp_dist = float(request.form['comp_dist'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        item_mrp = float(request.form['item_mrp'])
        item_visibility = float(request.form['item_visibility'])
        outlet_type = int(request.form['outlet_type'])
        outlet_size = int(request.form['outlet_size'])

        # Create input array with only 8 features
        input_array = np.array([
            prev_sales, comp_dist, month, year,
            item_mrp, item_visibility, outlet_type, outlet_size
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]
        output_message = f"The predicted sales value is ₹{prediction:,.2f}"

    except Exception as e:
        output_message = f"Error: {str(e)}"

    return render_template('result.html', message=output_message)

if __name__ == '__main__':
    app.run(debug=True)