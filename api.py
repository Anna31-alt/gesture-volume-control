from flask import Flask, request, jsonify

app = Flask(__name__)

# Home route (optional)
@app.route('/')
def home():
    return "Gesture Volume API Running"

# Volume API
@app.route('/volume', methods=['POST'])
def calculate_volume():
    data = request.json

    distance = data.get("distance", 0)

    # Same logic as your project
    volume = int((distance - 20) * 100 / (200 - 20))
    volume = max(0, min(100, volume))

    return jsonify({
        "input_distance": distance,
        "calculated_volume": volume,
        "status": "success"
    })

if __name__ == "__main__":
    app.run(debug=True)