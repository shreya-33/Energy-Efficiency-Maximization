from flask import Flask, request, jsonify
from flask_cors import cross_origin
from smartenergybackend import generate_energy_recommendations  # Import the function from your script

app = Flask(__name__)

@app.route('/recommendations', methods=['POST'])
@cross_origin()
def get_recommendations():
    data = request
    results = generate_energy_recommendations(data)
    return results

if __name__ == '__main__':
    app.run(debug=True)
    
