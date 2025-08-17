#!/usr/bin/env python3
"""
ArthChain API Monitoring Dashboard Server
Serves the comprehensive monitoring dashboard at xyz.arthachain.in
"""

from flask import Flask, send_file, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    """Serve the main monitoring dashboard"""
    return send_file('arthchain_monitor_dashboard.html')

@app.route('/api/dashboard/health')
def dashboard_health():
    """Dashboard health check"""
    return jsonify({
        "status": "healthy",
        "dashboard": "ArthChain API Monitor",
        "version": "1.0.0",
        "serving": "xyz.arthachain.in"
    })

if __name__ == '__main__':
    print("🚀 Starting ArthChain API Monitoring Dashboard...")
    print("📊 Dashboard: http://localhost:8081")
    print("🌐 Live URL: https://xyz.arthachain.in")
    print("📈 Monitoring 100+ ArthChain APIs with real-time data")
    print("🔗 No mock data - all real blockchain metrics")
    
    app.run(host='0.0.0.0', port=8081, debug=False)
