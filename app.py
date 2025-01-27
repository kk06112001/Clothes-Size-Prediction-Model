from flask import Flask, render_template, request, redirect, url_for
from etl_pipeline.model_trainer import predict_size
import logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    success = request.args.get('success', False, type=bool)
    error_message = request.args.get('error_message', '')
    predicted_size = request.args.get('predicted_size', '')
    return render_template('index.html', success=success, error_message=error_message, predicted_size=predicted_size)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        weight = float(request.form.get('weight'))
        age = float(request.form.get('age'))
        height = float(request.form.get('height'))
        predicted_size = predict_size(weight, age, height)
        logging.info(f"Prediction successful: {predicted_size}")
        return redirect(url_for('index', success=True, predicted_size=predicted_size))
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return redirect(url_for('index', success=False, error_message=str(e)))

if __name__ == '__main__':
    app.run(debug=True)
