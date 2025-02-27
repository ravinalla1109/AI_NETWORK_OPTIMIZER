# AI-Driven Autonomous Network Optimizer

## Overview
The **AI-Driven Autonomous Network Optimizer** leverages Artificial Intelligence (AI) and real-time network metrics to optimize network traffic, monitor performance, and evaluate AI-driven improvements in real-time. This project allows you to simulate and optimize network traffic (latency, packet loss, throughput), visualize network performance, and explore world map-based network traffic data.

## Features
- **Network Traffic Optimization**: Users can select various network metrics (latency, packet loss, throughput) and enter corresponding values for optimization.
- **Real-time Network Metrics**: Fetch live network metrics such as latency, packet loss, and throughput.
- **Network Performance Visualization**: Display network metrics in a colorful and interactive line/bar chart.
- **World Map Visualization**: Visual representation of network traffic hotspots on a world map with multiple points of interest.
- **AI-Powered Optimizations**: Uses AI techniques (simulated for now) to optimize network performance based on real-time input.

## Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Charting & Visualization**: Chart.js for network performance graphs, Leaflet.js for world map visualization.
- **AI Optimization**: Simulated with random data (future work: integrate AI models for real optimization).
- **Database**: None (current data is simulated).
- **Monitoring & Visualization**: Prometheus for real-time metrics collection, Grafana for dashboard visualization.
- **Network Simulation & Automation**: Docker for network simulation, Terraform for infrastructure automation.
- **Packet Analysis**: Wireshark for network traffic capture.

## Installation & Setup

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-network-optimizer.git
cd ai-network-optimizer
```

### 2. Set Up a Virtual Environment
For Python dependency management, it's highly recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
Install all required dependencies using `pip`.

```bash
pip install -r requirements.txt
```

### 4. Docker & Terraform Setup
To simulate the network environment and automate deployment, use Docker and Terraform:

1. **Docker**: To simulate network nodes, use Docker containers.
2. **Terraform**: Use Terraform to automate the infrastructure deployment.

To start Docker containers and set up network simulation:
```bash
docker-compose up -d
```

To automate deployment using Terraform:
```bash
terraform init
terraform apply
```

### 5. Run the Application
To start the Flask server, use the following command:

```bash
python app.py
```

Once the server is running, open your browser and go to `http://127.0.0.1:5000/` to view the application.

### 6. Set Up Monitoring with Prometheus and Grafana
- **Prometheus**: Used for collecting network metrics in real-time.
  - Run the Prometheus container:
  ```bash
  docker run -p 9090:9090 prom/prometheus
  ```
  - Configure Prometheus to collect metrics from Docker containers.
  
- **Grafana**: Use Grafana to visualize the data from Prometheus.
  - Run the Grafana container:
  ```bash
  docker run -d -p 3000:3000 grafana/grafana
  ```
  - Add Prometheus as the data source in Grafana and create dashboards to visualize network metrics.

### 7. Monitor Traffic with Wireshark
You can use **Wireshark** to capture network packets in the simulated environment. Install Wireshark and start capturing packets on the network interfaces used by the Docker containers.

## Usage
1. **Optimize Network Traffic**:
   - Select the network metric (Latency, Packet Loss, Throughput) and enter the corresponding value (e.g., 500ms for Latency).
   - Click "Optimize" to get the optimized network performance, including status (Network Normal or Fault Detected) along with the adjusted values.

2. **Fetch Live Data**:
   - Click on "Fetch Live Data" to retrieve current network metrics (Latency, Packet Loss, Throughput).
   
3. **Network Performance Visualization**:
   - The "Network Performance Visualization" section shows a line/bar chart for real-time comparison of network metrics.

4. **World Map Visualization**:
   - View the network traffic on a world map with multiple points representing hotspots for optimized network usage.

## Example Screenshots

### Network Optimization:
<img width="572" alt="{116E6FFA-922E-4908-8550-F9ED624E0737}" src="https://github.com/user-attachments/assets/e351fd43-efdb-465e-aa20-887437c978c9" />

### Network Performance Metrics:
<img width="720" alt="{B98F1AD4-5A49-400F-843C-869AF3510459}" src="https://github.com/user-attachments/assets/524b9744-d5ab-4754-93d8-cc3dd84234d3" />

### Network Traffic on World Map:
<img width="720" alt="{362966BF-AD75-454A-AF09-79CFB44A6B0C}" src="https://github.com/user-attachments/assets/44c047a0-1d8c-4eb2-adf1-e9240b62415b" />


## Roadmap

### Planned Enhancements:
- **Real-Time AI Integration**: Implement actual AI models for dynamic network optimization.
- **Enhanced UI**: Improve the frontend for a more interactive user experience.
- **Advanced World Map Visualization**: Show network traffic data on the world map with more granular details.
- **Live Data Stream**: Integrate actual network monitoring tools to fetch live data (e.g., Wireshark, Scapy).
<img width="440" alt="{55DB38E1-569F-44EF-A3BD-2D3992972A57}" src="https://github.com/user-attachments/assets/61467eb5-c580-48da-ae42-3da15051f5d4" />

- **AI-based Fault Detection**: Implement machine learning to predict and detect faults in the network in real-time.

### Future Considerations:
- **Cloud Integration**: Expand the project for cloud-based network optimization.
- **Scalability**: Improve the project for scalability to handle more complex network environments.

## Contribution

We welcome contributions to this project! If you'd like to help out, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request with a description of your changes.

## Acknowledgements

- **Flask**: Lightweight Python web framework used for backend development.
- **Chart.js**: Used for creating visually appealing network performance graphs.
- **Leaflet.js**: Powerful open-source library used for interactive maps.
- **OpenStreetMap**: Provides free map data for visualizing network traffic locations.
- **Prometheus**: For real-time network metric collection.
- **Grafana**: Used for monitoring and visualizing network data.
- **Docker**: For simulating network environments.
- **Wireshark**: For network traffic analysis.

```
