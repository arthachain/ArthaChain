#!/usr/bin/env python3
"""
Global WebSocket Test Script for ArthaChain API
Tests WebSocket connection through Cloudflare tunnel
"""

import asyncio
import websockets
import json
import sys

async def test_global_websocket():
    """Test WebSocket connection through Cloudflare tunnel"""
    # Test both local and global endpoints
    endpoints = [
        ("Local", "ws://localhost:8080/api/v1/ws"),
        ("Global via Cloudflare", "wss://api.arthachain.in/api/v1/ws"),
        ("Testnet via Cloudflare", "wss://testnet.arthachain.in/api/v1/ws"),
        ("RPC via Cloudflare", "wss://rpc.arthachain.in/api/v1/ws")
    ]
    
    for name, uri in endpoints:
        print(f"\n{'='*50}")
        print(f"🔌 Testing {name}: {uri}")
        print(f"{'='*50}")
        
        try:
            print(f"📡 Attempting connection...")
            async with websockets.connect(uri) as websocket:
                print(f"✅ {name} WebSocket connection established!")
                
                # Wait for welcome message
                print(f"📨 Waiting for welcome message...")
                welcome_msg = await websocket.recv()
                print(f"📨 Welcome: {welcome_msg}")
                
                # Subscribe to events
                subscribe_msg = {
                    "action": "subscribe",
                    "events": ["new_block", "new_transaction"]
                }
                print(f"📋 Subscribing to events...")
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for subscription confirmation
                sub_confirmation = await websocket.recv()
                print(f"✅ Subscription confirmed: {sub_confirmation}")
                
                # Test real-time communication
                print(f"🎧 Testing real-time communication...")
                
                # Send a ping message
                ping_msg = {"action": "ping", "timestamp": "test"}
                await websocket.send(json.dumps(ping_msg))
                print(f"📤 Sent ping message")
                
                # Wait for any response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📥 Response received: {response}")
                except asyncio.TimeoutError:
                    print(f"⏱️  No response in 5 seconds (expected)")
                
                # Unsubscribe
                unsubscribe_msg = {
                    "action": "unsubscribe",
                    "events": ["new_block", "new_transaction"]
                }
                await websocket.send(json.dumps(unsubscribe_msg))
                
                # Wait for unsubscription confirmation
                unsub_confirmation = await websocket.recv()
                print(f"✅ Unsubscription confirmed: {unsub_confirmation}")
                
                print(f"✅ {name} WebSocket test PASSED!")
                
        except websockets.exceptions.InvalidURI:
            print(f"❌ {name}: Invalid WebSocket URI")
        except ConnectionRefusedError:
            print(f"❌ {name}: Connection refused - server not accessible")
        except websockets.exceptions.WebSocketException as e:
            print(f"❌ {name}: WebSocket error: {e}")
        except asyncio.TimeoutError:
            print(f"❌ {name}: Connection timeout - endpoint not responding")
        except Exception as e:
            print(f"❌ {name}: Unexpected error: {e}")
        
        print(f"⏳ Waiting 2 seconds before next test...")
        await asyncio.sleep(2)

if __name__ == "__main__":
    print("🌍 ArthaChain Global WebSocket Test")
    print("Testing WebSocket endpoints through Cloudflare tunnel")
    print("=" * 60)
    
    try:
        asyncio.run(test_global_websocket())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    
    print("\n✅ Global WebSocket test completed!")
