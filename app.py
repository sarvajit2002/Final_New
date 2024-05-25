from flask import Flask, request, jsonify

app = Flask(__name__)
import joblib
import Untitled


# Load your model
model = joblib.load('./Untitled.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input_data'])  # Assuming input_data is a list
    input_data = input_data.reshape(1, input_data.shape[0], 1)  # Reshape input for LSTM
    prediction = model.predict(input_data)
    


    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)