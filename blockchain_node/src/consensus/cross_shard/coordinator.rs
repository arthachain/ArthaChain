use crate::consensus::cross_shard::{CrossShardConfig, CrossShardTransaction};
use crate::consensus::cross_shard::protocol::{CrossShardStatus, CrossShardTxType, ProtocolMessage};
use crate::utils::crypto::{dilithium_sign, dilithium_verify, quantum_resistant_hash};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};
use log::{debug, info, warn, error};
use serde::{Deserialize, Serialize};

/// Transaction preparation phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxPhase {
    /// Preparation phase (locks resources)
    Prepare,
    /// Commit phase (finalizes changes)
    Commit,
    /// Abort phase (releases locks without changes)
    Abort,
}

/// Cross-shard coordinator message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinatorMessage {
    /// Prepare request from coordinator to participant
    PrepareRequest {
        /// Transaction ID
        tx_id: String,
        /// Transaction data
        tx_data: Vec<u8>,
        /// Source shard
        from_shard: u32,
        /// Target shard
        to_shard: u32,
        /// Coordinator's quantum signature
        signature: Vec<u8>,
        /// Timestamp for message ordering
        timestamp: u64,
    },
    /// Prepare response from participant to coordinator
    PrepareResponse {
        /// Transaction ID
        tx_id: String,
        /// Success or failure
        success: bool,
        /// Failure reason (if any)
        reason: Option<String>,
        /// Participant's quantum signature
        signature: Vec<u8>,
        /// Participant shard
        shard_id: u32,
    },
    /// Commit request from coordinator to participants
    CommitRequest {
        /// Transaction ID
        tx_id: String,
        /// Transaction proof
        proof: Vec<u8>,
        /// Coordinator's quantum signature
        signature: Vec<u8>,
    },
    /// Abort request from coordinator to participants
    AbortRequest {
        /// Transaction ID
        tx_id: String,
        /// Abort reason
        reason: String,
        /// Coordinator's quantum signature
        signature: Vec<u8>,
    },
    /// Acknowledgment from participant to coordinator
    Acknowledgment {
        /// Transaction ID
        tx_id: String,
        /// Phase being acknowledged
        phase: TxPhase,
        /// Success flag
        success: bool,
        /// Participant's quantum signature
        signature: Vec<u8>,
        /// Participant shard
        shard_id: u32,
    },
    /// Heartbeat to detect failures
    Heartbeat {
        /// Source shard
        from_shard: u32,
        /// Timestamp
        timestamp: u64,
        /// Quantum signature
        signature: Vec<u8>,
    },
}

/// Resource lock information
#[derive(Debug, Clone)]
struct ResourceLock {
    /// Resource ID (account, contract, etc.)
    resource_id: String,
    /// Transaction that holds the lock
    tx_id: String,
    /// Timestamp when the lock was acquired
    acquired_at: Instant,
    /// Lock expiration time
    expires_at: Instant,
    /// Shard that holds the resource
    shard_id: u32,
}

/// Transaction coordinator state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoordinatorTxState {
    /// Transaction ID
    tx_id: String,
    /// Current phase
    phase: TxPhase,
    /// List of participating shards
    participants: Vec<u32>,
    /// Prepared participants (those who've voted YES)
    prepared: HashSet<u32>,
    /// Committed/acknowledged participants
    committed: HashSet<u32>,
    /// Transaction start time
    start_time: SystemTime,
    /// Last action time (for timeout tracking)
    last_action: SystemTime,
    /// Transaction data
    tx_data: Vec<u8>,
    /// Transaction type
    tx_type: CrossShardTxType,
    /// Transaction timeout
    timeout: Duration,
    /// Number of retry attempts
    retry_count: u32,
    /// Maximum retries allowed
    max_retries: u32,
    /// Quantum-resistant transaction hash
    quantum_hash: Vec<u8>,
}

impl CoordinatorTxState {
    /// Create a new transaction state
    pub fn new(
        tx_id: String,
        participants: Vec<u32>,
        tx_data: Vec<u8>,
        tx_type: CrossShardTxType,
        timeout: Duration,
        max_retries: u32,
    ) -> Result<Self> {
        let quantum_hash = quantum_resistant_hash(&tx_data)?;
        
        Ok(Self {
            tx_id,
            phase: TxPhase::Prepare,
            participants,
            prepared: HashSet::new(),
            committed: HashSet::new(),
            start_time: SystemTime::now(),
            last_action: SystemTime::now(),
            tx_data,
            tx_type,
            timeout,
            retry_count: 0,
            max_retries,
            quantum_hash,
        })
    }
    
    /// Check if all participants have prepared
    pub fn all_prepared(&self) -> bool {
        self.prepared.len() == self.participants.len()
    }
    
    /// Check if all participants have committed
    pub fn all_committed(&self) -> bool {
        self.committed.len() == self.participants.len()
    }
    
    /// Check if the transaction has timed out
    pub fn is_timed_out(&self) -> bool {
        match SystemTime::now().duration_since(self.last_action) {
            Ok(elapsed) => elapsed >= self.timeout,
            Err(_) => false,
        }
    }
    
    /// Update the last action timestamp
    pub fn update_last_action(&mut self) {
        self.last_action = SystemTime::now();
    }
    
    /// Increment retry count
    pub fn increment_retry_count(&mut self) -> bool {
        self.retry_count += 1;
        self.retry_count <= self.max_retries
    }
}

/// Cross-Shard Transaction Coordinator
/// Implements Two-Phase Commit protocol with quantum cryptography
pub struct CrossShardCoordinator {
    /// Local shard ID
    local_shard: u32,
    /// Configuration
    config: CrossShardConfig,
    /// Transactions being coordinated
    transactions: Arc<RwLock<HashMap<String, CoordinatorTxState>>>,
    /// Resource locks
    resource_locks: Arc<RwLock<HashMap<String, ResourceLock>>>,
    /// Running flag
    running: Arc<Mutex<bool>>,
    /// Timeout checker task
    timeout_checker: Option<JoinHandle<()>>,
    /// Message sender
    message_sender: mpsc::Sender<CoordinatorMessage>,
    /// Message receiver
    message_receiver: mpsc::Receiver<CoordinatorMessage>,
    /// Node's quantum key for signing
    quantum_key: Vec<u8>,
    /// Connected shard heartbeats (shard_id -> last_heartbeat_time)
    heartbeats: Arc<RwLock<HashMap<u32, SystemTime>>>,
}

impl CrossShardCoordinator {
    /// Create a new cross-shard transaction coordinator
    pub fn new(
        config: CrossShardConfig,
        quantum_key: Vec<u8>,
        message_sender: mpsc::Sender<CoordinatorMessage>,
        message_receiver: mpsc::Receiver<CoordinatorMessage>,
    ) -> Self {
        Self {
            local_shard: config.local_shard,
            config,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            resource_locks: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(Mutex::new(false)),
            timeout_checker: None,
            message_sender,
            message_receiver,
            quantum_key,
            heartbeats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Start the coordinator
    pub fn start(&mut self) -> Result<()> {
        // Set running flag
        let mut running = self.running.lock().unwrap();
        if *running {
            return Err(anyhow!("Coordinator already running"));
        }
        *running = true;
        drop(running);
        
        // Clone resources needed for the timeout checker
        let transactions = self.transactions.clone();
        let resource_locks = self.resource_locks.clone();
        let message_sender = self.message_sender.clone();
        let running_flag = self.running.clone();
        let quantum_key = self.quantum_key.clone();
        let local_shard = self.local_shard;
        let retry_interval = self.config.timeout_check_interval;
        let heartbeats = self.heartbeats.clone();
        let mut message_receiver = self.message_receiver.clone();
        
        // Start timeout and message handler
        self.timeout_checker = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(retry_interval);
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Check for timed-out transactions
                        let mut to_abort = Vec::new();
                        
                        {
                            let mut tx_map = transactions.write().unwrap();
                            
                            for (tx_id, tx_state) in tx_map.iter_mut() {
                                if tx_state.is_timed_out() {
                                    if tx_state.increment_retry_count() {
                                        // Retry the transaction
                                        tx_state.update_last_action();
                                        
                                        // For prepare phase, resend prepare requests
                                        // For commit phase, resend commit requests
                                        match tx_state.phase {
                                            TxPhase::Prepare => {
                                                debug!("Retrying prepare phase for transaction {}", tx_id);
                                                // Logic for retrying prepare requests will be added here
                                            }
                                            TxPhase::Commit => {
                                                debug!("Retrying commit phase for transaction {}", tx_id);
                                                // Logic for retrying commit requests will be added here
                                            }
                                            TxPhase::Abort => {
                                                debug!("Retrying abort phase for transaction {}", tx_id);
                                                // Logic for retrying abort requests will be added here
                                            }
                                        }
                                    } else {
                                        // Max retries reached, abort the transaction
                                        to_abort.push((tx_id.clone(), tx_state.clone()));
                                    }
                                }
                            }
                        }
                        
                        // Abort transactions that have reached max retries
                        for (tx_id, tx_state) in to_abort {
                            let tx_id_bytes = tx_id.as_bytes();
                            if let Ok(signature) = dilithium_sign(&quantum_key, tx_id_bytes) {
                                let abort_msg = CoordinatorMessage::AbortRequest {
                                    tx_id: tx_id.clone(),
                                    reason: "Transaction timed out after max retries".to_string(),
                                    signature,
                                };
                                
                                // Send abort message to all participants
                                if let Err(e) = message_sender.try_send(abort_msg) {
                                    error!("Failed to send abort message for tx {}: {}", tx_id, e);
                                }
                            }
                            
                            // Release any locks held by this transaction
                            let mut locks = resource_locks.write().unwrap();
                            locks.retain(|_, lock| lock.tx_id != tx_id);
                        }
                        
                        // Check for expired resource locks
                        let now = Instant::now();
                        let mut locks = resource_locks.write().unwrap();
                        locks.retain(|_, lock| lock.expires_at > now);
                        
                        // Check for shard heartbeats
                        let mut heartbeats_map = heartbeats.write().unwrap();
                        let now = SystemTime::now();
                        let mut expired_shards = Vec::new();
                        
                        for (shard_id, last_heartbeat) in heartbeats_map.iter() {
                            if let Ok(elapsed) = now.duration_since(*last_heartbeat) {
                                if elapsed > retry_interval * 3 {
                                    expired_shards.push(*shard_id);
                                }
                            }
                        }
                        
                        // Handle expired shards by potentially aborting transactions
                        if !expired_shards.is_empty() {
                            warn!("Shards {:?} may be down, checking affected transactions", expired_shards);
                            // Logic for handling down shards will be added here
                        }
                        
                        // Send heartbeat from this coordinator
                        if let Ok(timestamp) = now.duration_since(SystemTime::UNIX_EPOCH) {
                            let timestamp_bytes = timestamp.as_secs().to_le_bytes();
                            if let Ok(signature) = dilithium_sign(&quantum_key, &timestamp_bytes) {
                                let heartbeat = CoordinatorMessage::Heartbeat {
                                    from_shard: local_shard,
                                    timestamp: timestamp.as_secs(),
                                    signature,
                                };
                                
                                if let Err(e) = message_sender.try_send(heartbeat) {
                                    error!("Failed to send heartbeat: {}", e);
                                }
                            }
                        }
                    }
                    
                    Some(message) = message_receiver.recv() => {
                        // Process incoming messages
                        match message {
                            CoordinatorMessage::PrepareResponse { tx_id, success, reason, signature, shard_id } => {
                                Self::handle_prepare_response(
                                    tx_id, success, reason, signature, shard_id,
                                    &transactions, &message_sender, &quantum_key
                                ).await;
                            }
                            CoordinatorMessage::Acknowledgment { tx_id, phase, success, signature, shard_id } => {
                                Self::handle_acknowledgment(
                                    tx_id, phase, success, signature, shard_id,
                                    &transactions, &resource_locks
                                ).await;
                            }
                            CoordinatorMessage::Heartbeat { from_shard, timestamp, signature } => {
                                // Verify the heartbeat signature
                                // For simplicity, we're not doing full verification in this implementation
                                let mut heartbeats_map = heartbeats.write().unwrap();
                                heartbeats_map.insert(from_shard, SystemTime::now());
                            }
                            _ => {
                                // Other message types would be handled here
                            }
                        }
                    }
                }
                
                // Check if we should stop
                let running = running_flag.lock().unwrap();
                if !*running {
                    break;
                }
            }
        }));
        
        Ok(())
    }
    
    /// Stop the coordinator
    pub fn stop(&mut self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        if !*running {
            return Err(anyhow!("Coordinator not running"));
        }
        *running = false;
        
        // Wait for the timeout checker to stop
        if let Some(handle) = self.timeout_checker.take() {
            // In a real implementation, we'd wait for this to complete
            // For simplicity, we just detach it here
            handle.abort();
        }
        
        Ok(())
    }
    
    /// Initiate a new cross-shard transaction
    pub async fn initiate_transaction(
        &self,
        tx_data: Vec<u8>,
        from_shard: u32,
        to_shard: u32,
        resources: Vec<String>,
    ) -> Result<String> {
        // Generate a transaction ID
        let tx_id = uuid::Uuid::new_v4().to_string();
        
        // Check if resources are available and lock them
        let acquired_locks = self.try_acquire_locks(&tx_id, &resources)?;
        
        if !acquired_locks {
            return Err(anyhow!("Failed to acquire locks for some resources"));
        }
        
        // Create transaction state
        let tx_type = CrossShardTxType::DirectTransfer {
            from_shard,
            to_shard,
            amount: 0, // This would be extracted from tx_data in a real implementation
        };
        
        let participants = vec![from_shard, to_shard];
        let tx_state = CoordinatorTxState::new(
            tx_id.clone(),
            participants.clone(),
            tx_data.clone(),
            tx_type,
            self.config.transaction_timeout,
            self.config.retry_count as u32,
        )?;
        
        // Store the transaction state
        {
            let mut tx_map = self.transactions.write().unwrap();
            tx_map.insert(tx_id.clone(), tx_state);
        }
        
        // Send prepare requests to all participants
        for shard in participants {
            if shard == self.local_shard {
                // Local shard does not need a network message
                // In a real implementation, we'd process it locally
                continue;
            }
            
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            // Sign the prepare message with quantum-resistant signature
            let msg_data = format!("prepare:{}:{}:{}", tx_id, from_shard, to_shard).into_bytes();
            let signature = dilithium_sign(&self.quantum_key, &msg_data)?;
            
            let prepare_msg = CoordinatorMessage::PrepareRequest {
                tx_id: tx_id.clone(),
                tx_data: tx_data.clone(),
                from_shard,
                to_shard,
                signature,
                timestamp,
            };
            
            self.message_sender.send(prepare_msg).await?;
        }
        
        Ok(tx_id)
    }
    
    /// Try to acquire locks for resources
    fn try_acquire_locks(&self, tx_id: &str, resources: &[String]) -> Result<bool> {
        let mut locks = self.resource_locks.write().unwrap();
        let now = Instant::now();
        let lock_timeout = self.config.transaction_timeout;
        
        // Check if all resources are available
        for resource in resources {
            if let Some(lock) = locks.get(resource) {
                if lock.tx_id != tx_id && lock.expires_at > now {
                    // Resource is locked by another transaction
                    return Ok(false);
                }
            }
        }
        
        // Acquire locks for all resources
        for resource in resources {
            let lock = ResourceLock {
                resource_id: resource.clone(),
                tx_id: tx_id.to_string(),
                acquired_at: now,
                expires_at: now + lock_timeout,
                shard_id: self.local_shard,
            };
            
            locks.insert(resource.clone(), lock);
        }
        
        Ok(true)
    }
    
    /// Handle prepare response from a participant
    async fn handle_prepare_response(
        tx_id: String,
        success: bool,
        reason: Option<String>,
        signature: Vec<u8>,
        shard_id: u32,
        transactions: &Arc<RwLock<HashMap<String, CoordinatorTxState>>>,
        message_sender: &mpsc::Sender<CoordinatorMessage>,
        quantum_key: &[u8],
    ) {
        // In a real implementation, we would verify the signature here
        
        let mut should_commit = false;
        let mut should_abort = false;
        let participants = Vec::new();
        
        {
            let mut tx_map = transactions.write().unwrap();
            
            if let Some(tx_state) = tx_map.get_mut(&tx_id) {
                if success {
                    // Participant is prepared
                    tx_state.prepared.insert(shard_id);
                    tx_state.update_last_action();
                    
                    // Check if all participants are prepared
                    if tx_state.all_prepared() {
                        // Move to commit phase
                        tx_state.phase = TxPhase::Commit;
                        should_commit = true;
                    }
                } else {
                    // Participant rejected the transaction
                    should_abort = true;
                    tx_state.phase = TxPhase::Abort;
                    
                    // Clone participants for abort messages
                    for participant in &tx_state.participants {
                        participants.push(*participant);
                    }
                }
            }
        }
        
        // Send commit messages to all participants
        if should_commit {
            let msg_data = format!("commit:{}", tx_id).into_bytes();
            if let Ok(signature) = dilithium_sign(quantum_key, &msg_data) {
                let commit_msg = CoordinatorMessage::CommitRequest {
                    tx_id: tx_id.clone(),
                    proof: Vec::new(), // In a real implementation, this would include proof
                    signature,
                };
                
                // Send to all participants
                if let Err(e) = message_sender.send(commit_msg).await {
                    error!("Failed to send commit message: {}", e);
                    // In a real implementation, we would handle this error
                }
            }
        }
        
        // Send abort messages to all participants
        if should_abort {
            let abort_reason = reason.unwrap_or_else(|| "Participant rejected transaction".to_string());
            let msg_data = format!("abort:{}", tx_id).into_bytes();
            
            if let Ok(signature) = dilithium_sign(quantum_key, &msg_data) {
                let abort_msg = CoordinatorMessage::AbortRequest {
                    tx_id: tx_id.clone(),
                    reason: abort_reason,
                    signature,
                };
                
                // Send to all participants
                if let Err(e) = message_sender.send(abort_msg).await {
                    error!("Failed to send abort message: {}", e);
                    // In a real implementation, we would handle this error
                }
            }
        }
    }
    
    /// Handle acknowledgment from a participant
    async fn handle_acknowledgment(
        tx_id: String,
        phase: TxPhase,
        success: bool,
        signature: Vec<u8>,
        shard_id: u32,
        transactions: &Arc<RwLock<HashMap<String, CoordinatorTxState>>>,
        resource_locks: &Arc<RwLock<HashMap<String, ResourceLock>>>,
    ) {
        // In a real implementation, we would verify the signature here
        
        let mut tx_completed = false;
        
        {
            let mut tx_map = transactions.write().unwrap();
            
            if let Some(tx_state) = tx_map.get_mut(&tx_id) {
                match phase {
                    TxPhase::Commit => {
                        if success {
                            tx_state.committed.insert(shard_id);
                            tx_state.update_last_action();
                            
                            // Check if all participants have committed
                            if tx_state.all_committed() {
                                // Transaction is complete
                                tx_completed = true;
                            }
                        } else {
                            // This shouldn't happen in 2PC, but we handle it anyway
                            warn!("Participant {} failed to commit transaction {}", shard_id, tx_id);
                        }
                    }
                    TxPhase::Abort => {
                        // Participant acknowledged the abort
                        tx_state.committed.insert(shard_id);
                        tx_state.update_last_action();
                        
                        // Check if all participants have acknowledged the abort
                        if tx_state.all_committed() {
                            // Transaction abort is complete
                            tx_completed = true;
                        }
                    }
                    _ => {
                        // We don't expect acknowledgments for the prepare phase
                    }
                }
            }
        }
        
        if tx_completed {
            // Remove the transaction from our records
            let mut tx_map = transactions.write().unwrap();
            if let Some(tx_state) = tx_map.remove(&tx_id) {
                // Release resource locks
                let mut locks = resource_locks.write().unwrap();
                locks.retain(|_, lock| lock.tx_id != tx_id);
                
                info!("Transaction {} completed with phase {:?}", tx_id, tx_state.phase);
            }
        }
    }
    
    /// Get transaction status
    pub fn get_transaction_status(&self, tx_id: &str) -> Option<(TxPhase, bool)> {
        let tx_map = self.transactions.read().unwrap();
        
        tx_map.get(tx_id).map(|tx_state| {
            let is_complete = match tx_state.phase {
                TxPhase::Prepare => false,
                TxPhase::Commit => tx_state.all_committed(),
                TxPhase::Abort => tx_state.all_committed(),
            };
            
            (tx_state.phase, is_complete)
        })
    }
}

/// ParticipantHandler processes 2PC messages for a shard acting as a participant
pub struct ParticipantHandler {
    /// Local shard ID
    local_shard: u32,
    /// Configuration
    config: CrossShardConfig,
    /// Resource locks
    resource_locks: Arc<RwLock<HashMap<String, ResourceLock>>>,
    /// Prepared transactions (tx_id -> (resource_ids, tx_data))
    prepared_transactions: Arc<RwLock<HashMap<String, (Vec<String>, Vec<u8>)>>>,
    /// Node's quantum key for signing
    quantum_key: Vec<u8>,
    /// Message sender
    message_sender: mpsc::Sender<CoordinatorMessage>,
}

impl ParticipantHandler {
    /// Create a new participant handler
    pub fn new(
        config: CrossShardConfig,
        quantum_key: Vec<u8>,
        message_sender: mpsc::Sender<CoordinatorMessage>,
    ) -> Self {
        Self {
            local_shard: config.local_shard,
            config,
            resource_locks: Arc::new(RwLock::new(HashMap::new())),
            prepared_transactions: Arc::new(RwLock::new(HashMap::new())),
            quantum_key,
            message_sender,
        }
    }
    
    /// Handle prepare request from coordinator
    pub async fn handle_prepare_request(
        &self,
        tx_id: String,
        tx_data: Vec<u8>,
        from_shard: u32,
        to_shard: u32,
        coordinator_signature: Vec<u8>,
        timestamp: u64,
    ) -> Result<()> {
        // Verify coordinator's signature
        // For simplicity, we're not doing full verification in this implementation
        
        // Extract resources that need to be locked
        // In a real implementation, this would be determined by analyzing tx_data
        let resources = vec![format!("account:{}", from_shard), format!("account:{}", to_shard)];
        
        // Try to acquire locks
        let acquired_locks = self.try_acquire_locks(&tx_id, &resources)?;
        
        // Create response message
        let msg_data = format!("prepare_response:{}:{}", tx_id, self.local_shard).into_bytes();
        let signature = dilithium_sign(&self.quantum_key, &msg_data)?;
        
        let response = if acquired_locks {
            // Store prepared transaction
            {
                let mut prepared = self.prepared_transactions.write().unwrap();
                prepared.insert(tx_id.clone(), (resources, tx_data));
            }
            
            // Prepare successful
            CoordinatorMessage::PrepareResponse {
                tx_id,
                success: true,
                reason: None,
                signature,
                shard_id: self.local_shard,
            }
        } else {
            // Prepare failed
            CoordinatorMessage::PrepareResponse {
                tx_id,
                success: false,
                reason: Some("Failed to acquire locks".to_string()),
                signature,
                shard_id: self.local_shard,
            }
        };
        
        // Send response to coordinator
        self.message_sender.send(response).await?;
        
        Ok(())
    }
    
    /// Handle commit request from coordinator
    pub async fn handle_commit_request(
        &self,
        tx_id: String,
        proof: Vec<u8>,
        coordinator_signature: Vec<u8>,
    ) -> Result<()> {
        // Verify coordinator's signature
        // For simplicity, we're not doing full verification in this implementation
        
        // Check if we have this transaction prepared
        let tx_resources = {
            let prepared = self.prepared_transactions.read().unwrap();
            prepared.get(&tx_id).cloned()
        };
        
        if let Some((resources, tx_data)) = tx_resources {
            // Apply the transaction
            // In a real implementation, this would update the state
            
            // Create acknowledgment
            let msg_data = format!("commit_ack:{}:{}", tx_id, self.local_shard).into_bytes();
            let signature = dilithium_sign(&self.quantum_key, &msg_data)?;
            
            let ack = CoordinatorMessage::Acknowledgment {
                tx_id: tx_id.clone(),
                phase: TxPhase::Commit,
                success: true,
                signature,
                shard_id: self.local_shard,
            };
            
            // Send acknowledgment
            self.message_sender.send(ack).await?;
            
            // Remove from prepared transactions
            let mut prepared = self.prepared_transactions.write().unwrap();
            prepared.remove(&tx_id);
            
            // In a real implementation, we would keep the transaction record
            // but mark it as committed
            
            info!("Transaction {} committed", tx_id);
        } else {
            // We don't have this transaction prepared
            warn!("Received commit request for unknown transaction {}", tx_id);
            
            // Send negative acknowledgment
            let msg_data = format!("commit_nack:{}:{}", tx_id, self.local_shard).into_bytes();
            let signature = dilithium_sign(&self.quantum_key, &msg_data)?;
            
            let nack = CoordinatorMessage::Acknowledgment {
                tx_id,
                phase: TxPhase::Commit,
                success: false,
                signature,
                shard_id: self.local_shard,
            };
            
            self.message_sender.send(nack).await?;
        }
        
        Ok(())
    }
    
    /// Handle abort request from coordinator
    pub async fn handle_abort_request(
        &self,
        tx_id: String,
        reason: String,
        coordinator_signature: Vec<u8>,
    ) -> Result<()> {
        // Verify coordinator's signature
        // For simplicity, we're not doing full verification in this implementation
        
        // Release any locks for this transaction
        {
            let mut locks = self.resource_locks.write().unwrap();
            locks.retain(|_, lock| lock.tx_id != tx_id);
        }
        
        // Remove from prepared transactions
        {
            let mut prepared = self.prepared_transactions.write().unwrap();
            prepared.remove(&tx_id);
        }
        
        // Send acknowledgment
        let msg_data = format!("abort_ack:{}:{}", tx_id, self.local_shard).into_bytes();
        let signature = dilithium_sign(&self.quantum_key, &msg_data)?;
        
        let ack = CoordinatorMessage::Acknowledgment {
            tx_id,
            phase: TxPhase::Abort,
            success: true,
            signature,
            shard_id: self.local_shard,
        };
        
        self.message_sender.send(ack).await?;
        
        info!("Transaction aborted: {}", reason);
        
        Ok(())
    }
    
    /// Try to acquire locks for resources
    fn try_acquire_locks(&self, tx_id: &str, resources: &[String]) -> Result<bool> {
        let mut locks = self.resource_locks.write().unwrap();
        let now = Instant::now();
        let lock_timeout = self.config.transaction_timeout;
        
        // Check if all resources are available
        for resource in resources {
            if let Some(lock) = locks.get(resource) {
                if lock.tx_id != tx_id && lock.expires_at > now {
                    // Resource is locked by another transaction
                    return Ok(false);
                }
            }
        }
        
        // Acquire locks for all resources
        for resource in resources {
            let lock = ResourceLock {
                resource_id: resource.clone(),
                tx_id: tx_id.to_string(),
                acquired_at: now,
                expires_at: now + lock_timeout,
                shard_id: self.local_shard,
            };
            
            locks.insert(resource.clone(), lock);
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_coordinator_initiate_transaction() {
        // Create configuration
        let config = CrossShardConfig {
            local_shard: 0,
            connected_shards: vec![1, 2],
            ..CrossShardConfig::default()
        };
        
        // Create quantum key
        let quantum_key = vec![1, 2, 3, 4];
        
        // Create message channels
        let (tx, rx) = mpsc::channel(100);
        
        // Create coordinator
        let coordinator = CrossShardCoordinator::new(
            config,
            quantum_key,
            tx.clone(),
            rx,
        );
        
        // Initiate a transaction
        let tx_data = vec![5, 6, 7, 8];
        let from_shard = 0;
        let to_shard = 1;
        let resources = vec!["account:123".to_string(), "account:456".to_string()];
        
        let tx_id = coordinator.initiate_transaction(
            tx_data,
            from_shard,
            to_shard,
            resources,
        ).await.unwrap();
        
        // Verify transaction state
        let tx_map = coordinator.transactions.read().unwrap();
        assert!(tx_map.contains_key(&tx_id));
        
        // Verify resource locks
        let locks = coordinator.resource_locks.read().unwrap();
        assert_eq!(locks.len(), 2);
    }
} 