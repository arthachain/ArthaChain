#!/usr/bin/env python3
"""
WebSocket Test Script for ArthaChain API
Tests real-time blockchain data streaming
"""

import asyncio
import websockets
import json
import sys

async def test_websocket():
    """Test WebSocket connection to ArthaChain API"""
    uri = "ws://localhost:8080/api/v1/ws"
    
    try:
        print(f"🔌 Connecting to WebSocket: {uri}")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established!")
            
            # Wait for welcome message
            print("📡 Waiting for welcome message...")
            welcome_msg = await websocket.recv()
            print(f"📨 Welcome: {welcome_msg}")
            
            # Subscribe to new block events
            subscribe_msg = {
                "action": "subscribe",
                "events": ["new_block", "new_transaction", "consensus_update"]
            }
            print(f"📋 Subscribing to events: {subscribe_msg}")
            await websocket.send(json.dumps(subscribe_msg))
            
            # Wait for subscription confirmation
            sub_confirmation = await websocket.recv()
            print(f"✅ Subscription confirmed: {sub_confirmation}")
            
            # Listen for real-time events
            print("🎧 Listening for real-time blockchain events...")
            print("⏳ Waiting 30 seconds for events...")
            
            try:
                for i in range(30):
                    try:
                        # Wait for events with timeout
                        event = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        print(f"🚀 Event received: {event}")
                    except asyncio.TimeoutError:
                        print(f"⏱️  No events in second {i+1}/30...")
                        continue
            except websockets.exceptions.ConnectionClosed:
                print("🔌 WebSocket connection closed")
            
            # Unsubscribe from all events
            unsubscribe_msg = {
                "action": "unsubscribe",
                "events": ["new_block", "new_transaction", "consensus_update"]
            }
            print(f"📋 Unsubscribing: {unsubscribe_msg}")
            await websocket.send(json.dumps(unsubscribe_msg))
            
            # Wait for unsubscription confirmation
            unsub_confirmation = await websocket.recv()
            print(f"✅ Unsubscription confirmed: {unsub_confirmation}")
            
    except websockets.exceptions.InvalidURI:
        print("❌ Invalid WebSocket URI")
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused - WebSocket server not running")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 ArthaChain WebSocket Test")
    print("=" * 40)
    
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    
    print("\n✅ WebSocket test completed!")
