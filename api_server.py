from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route("/")
def home():
    return "Gesture Volume Control API Running"

@app.route("/status", methods=["GET"])
def status():

    # Example values (later you can connect real values)
    data = {
        "system": "Gesture Volume Control",
        "camera": "active",
        "hands_detected": 1,
        "volume_level": random.randint(0,100),
        "detection_quality": "Good"
    }

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)