#!/usr/bin/env python3
import socket
import threading
import json
import time
from datetime import datetime

class SimpleP2PNode:
    def __init__(self, port=30303, api_port=8080):
        self.port = port
        self.api_port = api_port
        self.peers = []
        self.running = True
        self.bootstrap_ip = "223.228.101.153"
        
    def start_p2p_listener(self):
        """Start P2P listener on port 30303"""
        try:
            p2p_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            p2p_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            p2p_socket.bind(('0.0.0.0', self.port))
            p2p_socket.listen(50)
            print(f"🔗 P2P listening on port {self.port}")
            
            while self.running:
                try:
                    client, addr = p2p_socket.accept()
                    print(f"🤝 New peer connected: {addr}")
                    self.peers.append(addr[0])
                    threading.Thread(target=self.handle_peer, args=(client, addr)).start()
                except:
                    break
        except Exception as e:
            print(f"❌ P2P listener error: {e}")
            
    def handle_peer(self, client, addr):
        """Handle peer connection"""
        try:
            while self.running:
                data = client.recv(1024)
                if not data:
                    break
                # Echo back peer discovery response
                response = {"status": "connected", "node_type": "arthachain", "peers": len(self.peers)}
                client.send(json.dumps(response).encode())
        except:
            pass
        finally:
            client.close()
            if addr[0] in self.peers:
                self.peers.remove(addr[0])
                
    def start_api_server(self):
        """Start simple API server"""
        try:
            api_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            api_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            api_socket.bind(('0.0.0.0', self.api_port))
            api_socket.listen(10)
            print(f"🌐 API listening on port {self.api_port}")
            
            while self.running:
                try:
                    client, addr = api_socket.accept()
                    threading.Thread(target=self.handle_api, args=(client,)).start()
                except:
                    break
        except Exception as e:
            print(f"❌ API server error: {e}")
            
    def handle_api(self, client):
        """Handle API requests"""
        try:
            request = client.recv(1024).decode()
            
            if "/api/health" in request:
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK"
            elif "/api/stats" in request:
                stats = {
                    "latest_block": int(time.time() % 1000),
                    "peers_connected": len(self.peers),
                    "p2p_port": self.port,
                    "status": "P2P_ENABLED"
                }
                body = json.dumps(stats)
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{body}"
            else:
                response = "HTTP/1.1 404 Not Found\r\n\r\n404 Not Found"
                
            client.send(response.encode())
        except:
            pass
        finally:
            client.close()
            
    def run(self):
        """Start both P2P and API servers"""
        print("🚀 Starting ArthaChain P2P Node")
        print(f"📡 P2P Port: {self.port}")
        print(f"🌐 API Port: {self.api_port}")
        print(f"🔗 Bootstrap: {self.bootstrap_ip}")
        
        # Start P2P listener
        threading.Thread(target=self.start_p2p_listener, daemon=True).start()
        
        # Start API server
        threading.Thread(target=self.start_api_server, daemon=True).start()
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                if len(self.peers) > 0:
                    print(f"👥 Connected peers: {len(self.peers)}")
        except KeyboardInterrupt:
            print("🛑 Stopping node...")
            self.running = False

if __name__ == "__main__":
    node = SimpleP2PNode()
    node.run()
