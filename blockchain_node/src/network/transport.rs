use anyhow::Result;
use futures::{SinkExt, StreamExt};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::mpsc::{self, Receiver, Sender},
    time,
};
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use crate::network::{
    error::NetworkError,
    message::{NetworkMessage, NodeInfo},
};

#[allow(dead_code)]
const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024; // 10MB
#[allow(dead_code)]
const PING_INTERVAL: Duration = Duration::from_secs(30);
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Transport {
    local_addr: SocketAddr,
    #[allow(dead_code)]
    node_info: NodeInfo,
    connections: Arc<Mutex<HashMap<SocketAddr, Connection>>>,
    #[allow(dead_code)]
    message_tx: Sender<(SocketAddr, NetworkMessage)>,
    message_rx: Receiver<(SocketAddr, NetworkMessage)>,
}

struct Connection {
    addr: SocketAddr,
    tx: Sender<NetworkMessage>,
}

impl Transport {
    pub fn new(local_addr: SocketAddr, node_info: NodeInfo) -> Self {
        let (message_tx, message_rx) = mpsc::channel(1000);

        Self {
            local_addr,
            node_info,
            connections: Arc::new(Mutex::new(HashMap::new())),
            message_tx,
            message_rx,
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        let listener = TcpListener::bind(self.local_addr).await?;
        println!("Transport listening on {}", self.local_addr);

        let (broadcast_tx, mut broadcast_rx) = mpsc::channel(1000);
        let connections = Arc::clone(&self.connections);

        // Handle incoming connections
        tokio::spawn(async move {
            while let Ok((stream, addr)) = listener.accept().await {
                println!("New connection from {addr}");
                if let Err(e) = Self::handle_connection(
                    stream,
                    addr,
                    broadcast_tx.clone(),
                    Arc::clone(&connections),
                )
                .await
                {
                    eprintln!("Error handling connection from {addr}: {e}");
                }
            }
        });

        // Handle outgoing messages
        loop {
            tokio::select! {
                Some((addr, msg)) = self.message_rx.recv() => {
                    if let Err(e) = self.send_message(addr, msg).await {
                        eprintln!("Error sending message to {addr}: {e}");
                    }
                }
                Some(msg) = broadcast_rx.recv() => {
                    if let Err(e) = self.broadcast_message(msg).await {
                        eprintln!("Error broadcasting message: {e}");
                    }
                }
            }
        }
    }

    async fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        broadcast_tx: Sender<NetworkMessage>,
        connections: Arc<Mutex<HashMap<SocketAddr, Connection>>>,
    ) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(100);

        // Store connection
        connections.lock().unwrap().insert(
            addr,
            Connection {
                addr,
                tx: tx.clone(),
            },
        );

        // Split stream into read/write parts
        let (mut write, mut read) = Framed::new(stream, LengthDelimitedCodec::new()).split();

        // Handle incoming messages
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(bytes) => {
                        match bincode::deserialize::<NetworkMessage>(&bytes) {
                            Ok(msg) => {
                                // Forward message to broadcast channel
                                if let Err(e) = broadcast_tx.send(msg).await {
                                    eprintln!("Error forwarding message: {e}");
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("Error deserializing message: {e}");
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error reading from stream: {e}");
                        break;
                    }
                }
            }

            // Remove connection on error/disconnect
            connections.lock().unwrap().remove(&addr);
        });

        // Handle outgoing messages
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match bincode::serialize(&msg) {
                    Ok(bytes) => {
                        if let Err(e) = write.send(bytes.into()).await {
                            eprintln!("Error sending message: {e}");
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Error serializing message: {e}");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    pub async fn connect(&mut self, addr: SocketAddr) -> Result<()> {
        // Check if already connected
        if self.connections.lock().unwrap().contains_key(&addr) {
            return Ok(());
        }

        // Connect with timeout
        let stream = time::timeout(CONNECTION_TIMEOUT, TcpStream::connect(addr)).await??;

        // Create broadcast channel for this connection
        let (broadcast_tx, _) = mpsc::channel(1000);

        // Handle connection
        Self::handle_connection(stream, addr, broadcast_tx, Arc::clone(&self.connections)).await?;

        Ok(())
    }

    pub async fn disconnect(&mut self, addr: SocketAddr) -> Result<()> {
        self.connections.lock().unwrap().remove(&addr);
        Ok(())
    }

    pub async fn send_message(&self, addr: SocketAddr, msg: NetworkMessage) -> Result<()> {
        let tx = {
            let connections = self.connections.lock().unwrap();
            connections.get(&addr).map(|conn| conn.tx.clone())
        };

        if let Some(tx) = tx {
            tx.send(msg)
                .await
                .map_err(|e| NetworkError::MessageError(e.to_string()))?;
            Ok(())
        } else {
            Err(NetworkError::PeerError(format!("Peer not found: {addr}")).into())
        }
    }

    pub async fn broadcast_message(&self, msg: NetworkMessage) -> Result<()> {
        let senders: Vec<_> = {
            let connections = self.connections.lock().unwrap();
            connections
                .values()
                .map(|conn| (conn.addr, conn.tx.clone()))
                .collect()
        };

        for (addr, tx) in senders {
            if let Err(e) = tx.send(msg.clone()).await {
                eprintln!("Error broadcasting to {addr}: {e}");
            }
        }
        Ok(())
    }

    pub fn get_connected_peers(&self) -> Vec<SocketAddr> {
        self.connections.lock().unwrap().keys().cloned().collect()
    }
}
