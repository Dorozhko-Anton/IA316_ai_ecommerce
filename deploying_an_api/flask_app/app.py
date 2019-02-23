from flask import Flask, request, jsonify
from model import RecommenderModel

app  = Flask(__name__)
agent = None

@app.route("/train", methods=['POST'])
def train():
    global agent
    history = request.json
    print(history)
    # history['nb_users']
    # history['nb_items']
    # history['user_history']
    # history['item_history']
    # history['rating_history']
    agent = RecommenderModel(history)
    return jsonify('OK')

@app.route("/predict", methods=['POST'])
def predict():
    global agent
    print(request.json)
    input_data = request.json

    #input_data['user']
    #input_data['item']
    output_data = agent.predict(input_data)
    response = jsonify(
                    rating = float(output_data)
                    )
    return response

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)
