use axum::{
    extract::{Extension, WebSocketUpgrade},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_stream::wrappers::BroadcastStream;

use crate::ledger::block::Block;
use crate::ledger::block::BlockExt;
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;

/// Event types that can be sent to WebSocket clients
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketEvent {
    /// New block event
    #[serde(rename = "new_block")]
    NewBlock(BlockEvent),

    /// New transaction event
    #[serde(rename = "new_transaction")]
    NewTransaction(TransactionEvent),

    /// Consensus update event
    #[serde(rename = "consensus_update")]
    ConsensusUpdate(ConsensusEvent),

    /// Subscription confirmation
    #[serde(rename = "subscription")]
    Subscription(SubscriptionEvent),
}

/// Data for a new block event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockEvent {
    /// Block hash
    pub hash: String,
    /// Block height
    pub height: u64,
    /// Number of transactions
    pub tx_count: usize,
    /// Timestamp
    pub timestamp: u64,
}

/// Data for a new transaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionEvent {
    /// Transaction hash
    pub hash: String,
    /// Sender address
    pub sender: String,
    /// Recipient address (if applicable)
    pub recipient: Option<String>,
    /// Transaction amount
    pub amount: u64,
}

/// Data for a consensus update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusEvent {
    /// View number
    pub view: u64,
    /// Phase
    pub phase: String,
    /// Leader
    pub leader: String,
}

/// Data for a subscription confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionEvent {
    /// Event types subscribed to
    pub events: Vec<String>,
    /// Success status
    pub success: bool,
}

/// Client message to subscribe/unsubscribe
#[derive(Debug, Deserialize)]
pub struct ClientMessage {
    /// Action to perform
    pub action: String,
    /// Event types to subscribe to
    pub events: Option<Vec<String>>,
}

/// The WebSocket handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Event manager for WebSocket events
#[derive(Clone)]
pub struct EventManager {
    /// Sender for new block events
    pub block_tx: broadcast::Sender<BlockEvent>,
    /// Sender for new transaction events
    pub transaction_tx: broadcast::Sender<TransactionEvent>,
    /// Sender for consensus update events
    pub consensus_tx: broadcast::Sender<ConsensusEvent>,
}

impl Default for EventManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EventManager {
    /// Create a new event manager
    pub fn new() -> Self {
        let (block_tx, _) = broadcast::channel(100);
        let (transaction_tx, _) = broadcast::channel(100);
        let (consensus_tx, _) = broadcast::channel(100);

        Self {
            block_tx,
            transaction_tx,
            consensus_tx,
        }
    }

    /// Publish a new block event
    pub fn publish_new_block(&self, block: &Block) {
        let event = BlockEvent {
            hash: hex::encode(block.hash_bytes()),
            height: block.header.height,
            tx_count: block.body.transactions.len(),
            timestamp: block.header.timestamp,
        };

        let _ = self.block_tx.send(event);
    }

    /// Publish a new transaction event
    pub fn publish_new_transaction(&self, tx: &Transaction) {
        let event = TransactionEvent {
            hash: hex::encode(tx.hash().as_bytes()),
            sender: tx.sender.clone(),
            recipient: tx.recipient.clone().into(),
            amount: tx.amount,
        };

        let _ = self.transaction_tx.send(event);
    }

    /// Publish a consensus update event
    pub fn publish_consensus_update(&self, view: u64, phase: &str, leader: &str) {
        let event = ConsensusEvent {
            view,
            phase: phase.to_string(),
            leader: leader.to_string(),
        };

        let _ = self.consensus_tx.send(event);
    }
}

/// Handle a WebSocket connection
async fn handle_socket(socket: axum::extract::ws::WebSocket, _state: Arc<RwLock<State>>) {
    // Split the socket into sender and receiver
    let (mut sender, mut receiver) = socket.split();

    // Create a channel for sending messages to the client
    let (tx, mut rx) = mpsc::channel::<WebSocketEvent>(100);

    // Create event subscriptions
    let event_manager = EventManager::new();
    let mut block_rx = None;
    let mut transaction_rx = None;
    let mut consensus_rx = None;

    // Send welcome message
    let welcome = WebSocketEvent::Subscription(SubscriptionEvent {
        events: vec![], // No subscriptions yet
        success: true,
    });
    let _ = tx.send(welcome).await;

    // Task to forward messages from the channel to the WebSocket
    let mut send_task = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let msg = serde_json::to_string(&event).unwrap();
            if sender
                .send(axum::extract::ws::Message::Text(msg))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    // Process messages from the client
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let axum::extract::ws::Message::Text(text) = msg {
                // Parse client message
                if let Ok(client_msg) = serde_json::from_str::<ClientMessage>(&text) {
                    match client_msg.action.as_str() {
                        "subscribe" => {
                            if let Some(events) = client_msg.events {
                                // Create subscriptions
                                let mut subscribed_events = Vec::new();

                                for event_type in events {
                                    match event_type.as_str() {
                                        "new_block" => {
                                            if block_rx.is_none() {
                                                block_rx = Some(BroadcastStream::new(
                                                    event_manager.block_tx.subscribe(),
                                                ));
                                                subscribed_events.push("new_block".to_string());
                                            }
                                        }
                                        "new_transaction" => {
                                            if transaction_rx.is_none() {
                                                transaction_rx = Some(BroadcastStream::new(
                                                    event_manager.transaction_tx.subscribe(),
                                                ));
                                                subscribed_events
                                                    .push("new_transaction".to_string());
                                            }
                                        }
                                        "consensus_update" => {
                                            if consensus_rx.is_none() {
                                                consensus_rx = Some(BroadcastStream::new(
                                                    event_manager.consensus_tx.subscribe(),
                                                ));
                                                subscribed_events
                                                    .push("consensus_update".to_string());
                                            }
                                        }
                                        _ => {}
                                    }
                                }

                                // Send subscription confirmation
                                let confirmation =
                                    WebSocketEvent::Subscription(SubscriptionEvent {
                                        events: subscribed_events,
                                        success: true,
                                    });
                                let _ = tx.send(confirmation).await;
                            }
                        }
                        "unsubscribe" => {
                            if let Some(events) = client_msg.events {
                                for event_type in events {
                                    match event_type.as_str() {
                                        "new_block" => {
                                            block_rx = None;
                                        }
                                        "new_transaction" => {
                                            transaction_rx = None;
                                        }
                                        "consensus_update" => {
                                            consensus_rx = None;
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                // Unsubscribe from all
                                block_rx = None;
                                transaction_rx = None;
                                consensus_rx = None;
                            }

                            // Send confirmation
                            let confirmation = WebSocketEvent::Subscription(SubscriptionEvent {
                                events: Vec::new(),
                                success: true,
                            });
                            let _ = tx.send(confirmation).await;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Client disconnected
        tx.closed().await;
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => {
            recv_task.abort();
        },
        _ = &mut recv_task => {
            send_task.abort();
        }
    };
}
