from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='AI Network Optimizer')

@app.route('/optimize', methods=['POST'])
def optimize():
    # Get the metric and value from the form data
    metric = request.json.get('metric')
    value = request.json.get('value')

    # Validate if the metric is valid
    if metric not in ['latency', 'packet_loss', 'throughput']:
        return jsonify({"error": "Invalid metric selected!"}), 400

    # Simulate network data for now (replace with actual AI model)
    if metric == 'latency':
        optimized_value = round(random.uniform(20, 100), 2)
    elif metric == 'packet_loss':
        optimized_value = round(random.uniform(0, 2), 2)
    else:  # throughput
        optimized_value = round(random.uniform(500, 1000), 2)

    # Set status to 'Network Normal' or 'Fault Detected' based on values
    status = "Network Normal" if optimized_value < 50 else "Fault Detected"
    
    # Return JSON data with the optimized values
    return jsonify({
        "status": status,
        "latency": f"{optimized_value} ms" if metric == 'latency' else "N/A",
        "packet_loss": f"{optimized_value}%" if metric == 'packet_loss' else "N/A",
        "throughput": f"{optimized_value} Mbps" if metric == 'throughput' else "N/A"
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Simulating live data for testing purposes
    metrics = {
        "latency": f"{random.uniform(20, 100):.2f} ms",
        "packet_loss": f"{random.uniform(0, 2):.2f}%",
        "throughput": f"{random.uniform(500, 1000):.2f} Mbps"
    }
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)
