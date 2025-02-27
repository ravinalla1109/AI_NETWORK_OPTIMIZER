$(document).ready(function() {
    $('#optimizeForm').submit(function(event) {
        event.preventDefault();

        var metric = $('#metric').val();
        var value = $('#value').val();

        // Send the selected metric and value to the backend
        $.ajax({
            url: '/optimize',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                metric: metric,
                value: value
            }),
            success: function(response) {
                // Display the result of optimization
                $('#optimizationResult').html(`
                    <h5>Status: ${response.status}</h5>
                    <p>Latency: ${response.latency}</p>
                    <p>Packet Loss: ${response.packet_loss}</p>
                    <p>Throughput: ${response.throughput}</p>
                `);
                updateNetworkChart(response);
            },
            error: function(response) {
                alert('Error: ' + response.responseJSON.error);
            }
        });
    });
});

function getNetworkMetrics() {
    $.get('/metrics', function(data) {
        $('#networkMetrics').html(`
            <p>Latency: ${data.latency}</p>
            <p>Packet Loss: ${data.packet_loss}</p>
            <p>Throughput: ${data.throughput}</p>
        `);
        updateNetworkChart(data);
    });
}

// Update the chart with the latest metrics
function updateNetworkChart(data) {
    var ctx = document.getElementById('networkChart').getContext('2d');
    var networkChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Latency', 'Packet Loss', 'Throughput'],
            datasets: [{
                label: 'Network Metrics',
                data: [
                    parseFloat(data.latency), 
                    parseFloat(data.packet_loss), 
                    parseFloat(data.throughput)
                ],
                backgroundColor: ['rgba(75, 192, 192, 0.7)', 'rgba(153, 102, 255, 0.7)', 'rgba(255, 159, 64, 0.7)'],
                borderColor: ['rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        font: {
                            size: 14
                        }
                    },
                    title: {
                        display: true,
                        text: 'Metric Value',
                        font: {
                            size: 16
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Network Metrics',
                        font: {
                            size: 16
                        }
                    },
                    ticks: {
                        font: {
                            size: 14
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 16
                        }
                    }
                }
            }
        }
    });
}

// Initialize the map using Leaflet.js
function initializeMap() {
    var map = L.map('worldMap').setView([51.505, -0.09], 2); // Starting at London

    // Add a tile layer (this is a free layer provided by OpenStreetMap)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Simulate multiple markers representing network hotspots
    var points = [
        { lat: 37.7749, lon: -122.4194, label: 'San Francisco' }, // US
        { lat: 51.5074, lon: -0.1278, label: 'London' }, // UK
        { lat: 48.8566, lon: 2.3522, label: 'Paris' }, // France
        { lat: 35.6762, lon: 139.6503, label: 'Tokyo' }, // Japan
        { lat: -33.8688, lon: 151.2093, label: 'Sydney' } // Australia
    ];

    points.forEach(function(point) {
        L.marker([point.lat, point.lon]).addTo(map)
            .bindPopup(`<b>Network Traffic Location</b><br>${point.label}<br>Real-time data can be monitored here.`)
            .openPopup();
    });
}

initializeMap();
