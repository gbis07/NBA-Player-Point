from flask import Flask, request, jsonify, redirect, url_for
import pandas as pd

app = Flask(__name__)

try:
    data = pd.read_csv("predictions.csv")
except Exception as e:
    print("Error reading CSV file:", e)
    data = None

@app.route("/")
def home():
    return redirect(url_for("get_predictions"))

@app.route("/predictions", methods=["GET"])
def get_predictions():
    player = request.args.get("player")
    if player and data is not None:
        filtered_data = data[data["player_name"].str.lower() == player.lower()]
        if filtered_data.empty:
            return jsonify({"error": f"No predictions found for player: {player}"}), 404
        return jsonify(filtered_data.to_dict(orient="records"))
    elif data is not None:
        return jsonify(data.to_dict(orient="records"))
    else:
        return jsonify({"error": "Data not loaded"}), 500

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
