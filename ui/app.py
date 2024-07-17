from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/hrv', methods=['POST'])
def analyze_hrv():
    data = request.json
    # Perform HRV analysis here
    result = {"status": "success", "data": data}  # Placeholder for analysis result
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
