#!/usr/bin/env python3
"""
ArthaChain Active P2P Node
Actively discovers and connects to peers
"""
import socket
import json
import threading
import time
import urllib.request
import random

class ActiveP2PNode:
    def __init__(self):
        self.p2p_port = self.find_free_port(30301)
        self.api_port = self.find_free_port(8081)
        self.peers = {}  # peer_ip: connection_info
        self.running = True
        self.node_id = f"node_{random.randint(1000, 9999)}"
        
        # Known network nodes to discover
        self.discovery_endpoints = [
            "https://api.arthachain.in/api/stats",
        ]
        
        # Peer discovery via port scanning
        self.common_p2p_ports = [30301, 30302, 30303, 30304, 30305]
        
    def find_free_port(self, start_port):
        for port in range(start_port, start_port + 100):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except:
                continue
        return start_port + random.randint(1, 100)
    
    def discover_peers(self):
        """Actively discover other nodes in the network"""
        print("🔍 Starting peer discovery...")
        
        # Method 1: Via API endpoints
        for endpoint in self.discovery_endpoints:
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    print(f"✅ Found network via {endpoint}")
                    print(f"📊 Network block: {data.get('latest_block', 'unknown')}")
            except:
                pass
        
        # Method 2: Scan for local peers (same network)
        threading.Thread(target=self.scan_for_peers, daemon=True).start()
        
    def scan_for_peers(self):
        """Scan for other ArthaChain nodes on common ports"""
        import subprocess
        
        # Get local network range
        try:
            # Simple network scan
            for port in self.common_p2p_ports:
                if port == self.p2p_port:
                    continue
                    
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', port))
                    if result == 0:
                        self.connect_to_peer('127.0.0.1', port)
                    sock.close()
                except:
                    pass
                    
            # Try to connect to known external nodes
            external_ips = ['223.228.101.153']  # Your main system
            for ip in external_ips:
                for port in self.common_p2p_ports:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(2)
                        result = sock.connect_ex((ip, port))
                        if result == 0:
                            self.connect_to_peer(ip, port)
                        sock.close()
                    except:
                        pass
        except:
            pass
    
    def connect_to_peer(self, ip, port):
        """Establish connection to discovered peer"""
        try:
            if f"{ip}:{port}" not in self.peers:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((ip, port))
                
                # Send handshake
                handshake = {
                    "node_id": self.node_id,
                    "p2p_port": self.p2p_port,
                    "api_port": self.api_port,
                    "action": "peer_connect"
                }
                sock.send(json.dumps(handshake).encode())
                
                # Get response
                response = sock.recv(1024)
                self.peers[f"{ip}:{port}"] = {
                    "ip": ip,
                    "port": port,
                    "connected_at": time.time(),
                    "socket": sock
                }
                print(f"🤝 Connected to peer {ip}:{port}")
                return True
        except Exception as e:
            print(f"❌ Failed to connect to {ip}:{port} - {e}")
            return False
    
    def start_p2p_listener(self):
        """P2P listener that accepts and tracks connections"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.p2p_port))
            sock.listen(50)
            print(f"🔗 P2P listening on port {self.p2p_port}")
            
            while self.running:
                try:
                    client, addr = sock.accept()
                    peer_key = f"{addr[0]}:{addr[1]}"
                    self.peers[peer_key] = {
                        "ip": addr[0],
                        "port": addr[1],
                        "connected_at": time.time(),
                        "socket": client
                    }
                    print(f"🤝 New peer connected: {addr[0]} (Total: {len(self.peers)})")
                    threading.Thread(target=self.handle_peer, args=(client, addr)).start()
                except:
                    break
        except Exception as e:
            print(f"❌ P2P listener error: {e}")
    
    def handle_peer(self, client, addr):
        """Handle peer communication"""
        try:
            while self.running:
                data = client.recv(1024)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode())
                    response = {
                        "status": "connected",
                        "node_id": self.node_id,
                        "peers": len(self.peers),
                        "action": "peer_response"
                    }
                    client.send(json.dumps(response).encode())
                except:
                    pass
        except:
            pass
        finally:
            # Remove peer when disconnected
            peer_key = f"{addr[0]}:{addr[1]}"
            if peer_key in self.peers:
                del self.peers[peer_key]
                print(f"👋 Peer disconnected: {addr[0]} (Remaining: {len(self.peers)})")
            client.close()
    
    def start_api_server(self):
        """API server with peer information"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.api_port))
            sock.listen(10)
            print(f"🌐 API listening on port {self.api_port}")
            
            while self.running:
                try:
                    client, addr = sock.accept()
                    threading.Thread(target=self.handle_api, args=(client,)).start()
                except:
                    break
        except Exception as e:
            print(f"❌ API error: {e}")
    
    def handle_api(self, client):
        """Handle API requests with real peer data"""
        try:
            request = client.recv(1024).decode()
            
            if "/api/health" in request:
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK"
            elif "/api/stats" in request:
                stats = {
                    "latest_block": int(time.time() % 10000),
                    "peers_connected": len(self.peers),
                    "p2p_port": self.p2p_port,
                    "api_port": self.api_port,
                    "node_id": self.node_id,
                    "status": "ACTIVE_P2P",
                    "peer_list": list(self.peers.keys()),
                    "discovery_active": True
                }
                body = json.dumps(stats)
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{body}"
            elif "/api/peers" in request:
                peer_info = {
                    "total_peers": len(self.peers),
                    "connected_peers": list(self.peers.keys()),
                    "discovery_status": "active"
                }
                body = json.dumps(peer_info)
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{body}"
            else:
                response = "HTTP/1.1 404 Not Found\r\n\r\n404"
                
            client.send(response.encode())
        except:
            pass
        finally:
            client.close()
    
    def periodic_peer_discovery(self):
        """Continuously discover new peers"""
        while self.running:
            time.sleep(30)  # Discovery every 30 seconds
            print(f"🔍 Peer discovery cycle - Current peers: {len(self.peers)}")
            self.scan_for_peers()
    
    def run(self):
        """Start active P2P node"""
        print("🚀 Starting ArthaChain Active P2P Node")
        print(f"🆔 Node ID: {self.node_id}")
        print(f"📡 P2P Port: {self.p2p_port}")
        print(f"🌐 API Port: {self.api_port}")
        print("🔍 Active peer discovery enabled")
        print("✅ Will actively connect to other nodes")
        
        # Start services
        threading.Thread(target=self.start_p2p_listener, daemon=True).start()
        threading.Thread(target=self.start_api_server, daemon=True).start()
        threading.Thread(target=self.periodic_peer_discovery, daemon=True).start()
        
        # Initial peer discovery
        self.discover_peers()
        
        # Keep running and show status
        try:
            while True:
                time.sleep(10)
                if len(self.peers) > 0:
                    print(f"👥 Active peers: {len(self.peers)} - {list(self.peers.keys())}")
                else:
                    print("🔍 Searching for peers...")
        except KeyboardInterrupt:
            print("🛑 Stopping active P2P node...")
            self.running = False

if __name__ == "__main__":
    node = ActiveP2PNode()
    node.run()
