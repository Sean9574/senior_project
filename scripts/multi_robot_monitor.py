#!/usr/bin/env python3
"""
Multi-Robot Training Monitor

Web dashboard showing training progress for all robots.
http://localhost:5000
"""

import json
import threading
from collections import defaultdict

import rclpy
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from rclpy.node import Node
from std_msgs.msg import String

app = Flask(__name__)
app.config['SECRET_KEY'] = 'multi_robot_monitor'
socketio = SocketIO(app, cors_allowed_origins="*")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Robot Training</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg: #0f1419;
            --card: #1a1f2e;
            --border: #2d3748;
            --text: #e2e8f0;
            --muted: #718096;
            --accent: #3b82f6;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 20px;
        }
        h1 { margin-bottom: 20px; font-size: 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
        .card {
            background: var(--card);
            border-radius: 8px;
            padding: 16px;
        }
        .card-title {
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .robot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .robot-card {
            background: var(--card);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .robot-id { font-size: 14px; color: var(--accent); margin-bottom: 8px; }
        .robot-stat { font-size: 24px; font-weight: bold; }
        .robot-label { font-size: 10px; color: var(--muted); }
        .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); }
        .stat-label { color: var(--muted); }
        .stat-value { font-family: monospace; }
        .success { color: var(--green); }
        .collision { color: var(--red); }
        .timeout { color: var(--yellow); }
        .chart-container { height: 200px; }
        #log { max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 11px; }
        .log-entry { padding: 4px 8px; border-bottom: 1px solid var(--border); }
        .log-entry.success { background: rgba(16, 185, 129, 0.1); }
        .log-entry.collision { background: rgba(239, 68, 68, 0.1); }
        .log-entry.timeout { background: rgba(245, 158, 11, 0.1); }
    </style>
</head>
<body>
    <h1>🤖 Multi-Robot Training Monitor</h1>
    
    <div class="robot-grid" id="robotGrid"></div>
    
    <div class="grid">
        <div class="card">
            <div class="card-title">Episode Returns (All Robots)</div>
            <div class="chart-container">
                <canvas id="returnChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">Training Statistics</div>
            <div class="stat-row">
                <span class="stat-label">Total Steps</span>
                <span class="stat-value" id="totalSteps">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Episodes</span>
                <span class="stat-value" id="totalEpisodes">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Success Rate</span>
                <span class="stat-value success" id="successRate">0%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Collision Rate</span>
                <span class="stat-value collision" id="collisionRate">0%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Return (last 10)</span>
                <span class="stat-value" id="avgReturn">--</span>
            </div>
        </div>
        
        <div class="card" style="grid-column: 1 / -1;">
            <div class="card-title">Episode Log</div>
            <div id="log"></div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        let numRobots = 4;
        let robotStats = {};
        let episodes = [];
        let outcomes = { success: 0, collision: 0, timeout: 0 };
        
        // Initialize robot cards
        function initRobotCards(n) {
            numRobots = n;
            const grid = document.getElementById('robotGrid');
            grid.innerHTML = '';
            for (let i = 0; i < n; i++) {
                robotStats[i] = { episodes: 0, returns: [], lastReturn: 0 };
                const card = document.createElement('div');
                card.className = 'robot-card';
                card.id = `robot${i}`;
                card.innerHTML = `
                    <div class="robot-id">Robot ${i}</div>
                    <div class="robot-stat" id="robot${i}Return">--</div>
                    <div class="robot-label">Last Return</div>
                    <div style="margin-top: 8px; font-size: 11px; color: var(--muted);">
                        Episodes: <span id="robot${i}Ep">0</span>
                    </div>
                `;
                grid.appendChild(card);
            }
        }
        initRobotCards(4);
        
        // Chart
        const ctx = document.getElementById('returnChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Return',
                    data: [],
                    borderColor: '#3b82f6',
                    tension: 0.3,
                    pointRadius: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#1a1f2e' }, ticks: { color: '#718096' } },
                    y: { grid: { color: '#1a1f2e' }, ticks: { color: '#718096' } }
                }
            }
        });
        
        socket.on('training_update', (data) => {
            const robotId = data.robot_id;
            
            // Update robot stats
            if (robotId >= numRobots) {
                initRobotCards(robotId + 1);
            }
            
            robotStats[robotId].episodes++;
            robotStats[robotId].lastReturn = data.return;
            robotStats[robotId].returns.push(data.return);
            
            document.getElementById(`robot${robotId}Return`).textContent = data.return.toFixed(0);
            document.getElementById(`robot${robotId}Ep`).textContent = robotStats[robotId].episodes;
            
            // Update outcomes
            if (data.outcome) outcomes[data.outcome]++;
            
            // Update chart
            episodes.push(data);
            if (chart.data.labels.length > 100) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            chart.data.labels.push(episodes.length);
            chart.data.datasets[0].data.push(data.return);
            chart.update('none');
            
            // Update stats
            document.getElementById('totalSteps').textContent = data.total_steps.toLocaleString();
            document.getElementById('totalEpisodes').textContent = episodes.length;
            
            const total = outcomes.success + outcomes.collision + outcomes.timeout;
            if (total > 0) {
                document.getElementById('successRate').textContent = (outcomes.success / total * 100).toFixed(1) + '%';
                document.getElementById('collisionRate').textContent = (outcomes.collision / total * 100).toFixed(1) + '%';
            }
            
            const recent = episodes.slice(-10);
            const avgRet = recent.reduce((a, b) => a + b.return, 0) / recent.length;
            document.getElementById('avgReturn').textContent = avgRet.toFixed(1);
            
            // Update log
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + (data.outcome || '');
            entry.textContent = `Robot ${robotId} | EP ${data.episode} | ${data.steps} steps | Return: ${data.return.toFixed(1)} | ${data.outcome || 'running'}`;
            log.insertBefore(entry, log.firstChild);
            while (log.children.length > 50) log.removeChild(log.lastChild);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

class MonitorNode(Node):
    def __init__(self):
        super().__init__('multi_robot_monitor')
        self.sub = self.create_subscription(
            String, '/training_stats',
            self.callback, 10
        )
        self.get_logger().info("Multi-Robot Monitor: http://localhost:5000")
    
    def callback(self, msg):
        try:
            data = json.loads(msg.data)
            socketio.emit('training_update', data)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def run_ros():
    rclpy.init()
    node = MonitorNode()
    rclpy.spin(node)
    rclpy.shutdown()

def main():
    ros_thread = threading.Thread(target=run_ros, daemon=True)
    ros_thread.start()
    
    print("\n" + "=" * 50)
    print("  Multi-Robot Training Monitor")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()
