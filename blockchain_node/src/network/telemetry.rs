use super::types::SerializableInstant;
use anyhow::Result;
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Registry};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Network telemetry metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub total_peers: usize,
    pub active_peers: usize,
    pub total_messages: usize,
    pub total_bytes: usize,
    pub average_latency: Duration,
    pub message_rate: f64,
    pub bandwidth_usage: f64,
    pub error_rate: f64,
    pub sync_status: SyncStatus,
    pub shard_metrics: HashMap<u64, ShardMetrics>,
    pub peer_metrics: HashMap<String, PeerMetrics>,
    pub connected_peers: Gauge,
    pub messages_sent: Counter,
    pub messages_received: Counter,
    pub bytes_sent: Counter,
    pub bytes_received: Counter,
    pub block_propagation_time: Histogram,
    pub transaction_propagation_time: Histogram,
}

/// Sync status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    pub is_syncing: bool,
    pub current_height: u64,
    pub target_height: u64,
    pub sync_progress: f64,
    pub sync_speed: f64,
    pub estimated_time_remaining: Duration,
}

/// Shard metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetrics {
    pub total_transactions: usize,
    pub total_blocks: usize,
    pub average_block_time: Duration,
    pub transaction_rate: f64,
    pub block_rate: f64,
    pub shard_size: usize,
    pub cross_shard_messages: usize,
}

/// Peer metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerMetrics {
    pub node_id: String,
    pub connected_since: SerializableInstant,
    pub last_seen: SerializableInstant,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub average_latency: f64,
}

/// Telemetry configuration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub update_interval: Duration,
    pub history_size: usize,
    pub metrics_retention: Duration,
    pub enable_prometheus: bool,
    pub prometheus_port: u16,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(1),
            history_size: 1000,
            metrics_retention: Duration::from_secs(3600),
            enable_prometheus: true,
            prometheus_port: 9090,
        }
    }
}

/// Network telemetry manager
pub struct NetworkTelemetry {
    config: TelemetryConfig,
    metrics: Arc<RwLock<NetworkMetrics>>,
    history: Arc<RwLock<VecDeque<(Instant, NetworkMetrics)>>>,
    _registry: Option<Registry>,
    prometheus_metrics: Option<RwLock<PrometheusMetrics>>,
}

/// Prometheus metrics
struct PrometheusMetrics {
    total_peers: Gauge,
    active_peers: Gauge,
    total_messages: Counter,
    total_bytes: Counter,
    average_latency: Histogram,
    message_rate: Gauge,
    bandwidth_usage: Gauge,
    error_rate: Gauge,
    sync_progress: Gauge,
    shard_metrics: HashMap<u64, ShardPrometheusMetrics>,
}

/// Shard Prometheus metrics
struct ShardPrometheusMetrics {
    total_transactions: Counter,
    total_blocks: Counter,
    average_block_time: Histogram,
    transaction_rate: Gauge,
    block_rate: Gauge,
    shard_size: Gauge,
    cross_shard_messages: Counter,
}

impl NetworkTelemetry {
    pub fn new(config: TelemetryConfig) -> Result<Self> {
        let registry = if config.enable_prometheus {
            Some(Registry::new())
        } else {
            None
        };

        let prometheus_metrics = if let Some(_registry) = &registry {
            Some(RwLock::new(PrometheusMetrics {
                total_peers: Gauge::new("total_peers", "Total number of peers")?,
                active_peers: Gauge::new("active_peers", "Number of active peers")?,
                total_messages: Counter::new("total_messages", "Total number of messages")?,
                total_bytes: Counter::new("total_bytes", "Total bytes transferred")?,
                average_latency: Histogram::with_opts(
                    HistogramOpts::new("average_latency", "Average network latency")
                        .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
                )?,
                message_rate: Gauge::new("message_rate", "Messages per second")?,
                bandwidth_usage: Gauge::new(
                    "bandwidth_usage",
                    "Bandwidth usage in bytes per second",
                )?,
                error_rate: Gauge::new("error_rate", "Error rate")?,
                sync_progress: Gauge::new("sync_progress", "Blockchain sync progress")?,
                shard_metrics: HashMap::new(),
            }))
        } else {
            None
        };

        Ok(Self {
            config,
            metrics: Arc::new(RwLock::new(NetworkMetrics {
                total_peers: 0,
                active_peers: 0,
                total_messages: 0,
                total_bytes: 0,
                average_latency: Duration::from_millis(0),
                message_rate: 0.0,
                bandwidth_usage: 0.0,
                error_rate: 0.0,
                sync_status: SyncStatus {
                    is_syncing: false,
                    current_height: 0,
                    target_height: 0,
                    sync_progress: 0.0,
                    sync_speed: 0.0,
                    estimated_time_remaining: Duration::from_secs(0),
                },
                shard_metrics: HashMap::new(),
                peer_metrics: HashMap::new(),
                connected_peers: Gauge::new("connected_peers", "Number of connected peers")?,
                messages_sent: Counter::new("messages_sent", "Total messages sent")?,
                messages_received: Counter::new("messages_received", "Total messages received")?,
                bytes_sent: Counter::new("bytes_sent", "Total bytes sent")?,
                bytes_received: Counter::new("bytes_received", "Total bytes received")?,
                block_propagation_time: Histogram::with_opts(
                    HistogramOpts::new("block_propagation_time", "Block propagation time")
                        .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
                )?,
                transaction_propagation_time: Histogram::with_opts(
                    HistogramOpts::new(
                        "transaction_propagation_time",
                        "Transaction propagation time",
                    )
                    .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
                )?,
            })),
            history: Arc::new(RwLock::new(VecDeque::new())),
            _registry: registry,
            prometheus_metrics,
        })
    }

    /// Update network metrics
    pub async fn update_metrics(&self, metrics: NetworkMetrics) -> Result<()> {
        let mut current_metrics = self.metrics.write().await;
        *current_metrics = metrics.clone();

        // Update history
        let mut history = self.history.write().await;
        history.push_back((Instant::now(), metrics.clone()));

        // Trim history if needed
        while history.len() > self.config.history_size {
            history.pop_front();
        }

        // Update Prometheus metrics if enabled
        if let Some(prometheus_lock) = &self.prometheus_metrics {
            let mut prometheus = prometheus_lock.write().await;
            prometheus.total_peers.set(metrics.total_peers as f64);
            prometheus.active_peers.set(metrics.active_peers as f64);
            prometheus
                .total_messages
                .inc_by(metrics.total_messages as f64);
            prometheus.total_bytes.inc_by(metrics.total_bytes as f64);
            prometheus
                .average_latency
                .observe(metrics.average_latency.as_secs_f64());
            prometheus.message_rate.set(metrics.message_rate);
            prometheus.bandwidth_usage.set(metrics.bandwidth_usage);
            prometheus.error_rate.set(metrics.error_rate);
            prometheus
                .sync_progress
                .set(metrics.sync_status.sync_progress);

            // Update shard metrics
            for (shard_id, shard_metrics) in &metrics.shard_metrics {
                let prometheus_shard =
                    prometheus
                        .shard_metrics
                        .entry(*shard_id)
                        .or_insert_with(|| ShardPrometheusMetrics {
                            total_transactions: Counter::new(
                                format!("shard_{}_total_transactions", shard_id),
                                "Total transactions in shard",
                            )
                            .unwrap(),
                            total_blocks: Counter::new(
                                format!("shard_{}_total_blocks", shard_id),
                                "Total blocks in shard",
                            )
                            .unwrap(),
                            average_block_time: Histogram::with_opts(HistogramOpts::new(
                                format!("shard_{}_average_block_time", shard_id),
                                "Average block time in shard",
                            ))
                            .unwrap(),
                            transaction_rate: Gauge::new(
                                format!("shard_{}_transaction_rate", shard_id),
                                "Transaction rate in shard",
                            )
                            .unwrap(),
                            block_rate: Gauge::new(
                                format!("shard_{}_block_rate", shard_id),
                                "Block rate in shard",
                            )
                            .unwrap(),
                            shard_size: Gauge::new(
                                format!("shard_{}_size", shard_id),
                                "Shard size",
                            )
                            .unwrap(),
                            cross_shard_messages: Counter::new(
                                format!("shard_{}_cross_shard_messages", shard_id),
                                "Cross-shard messages in shard",
                            )
                            .unwrap(),
                        });

                prometheus_shard
                    .total_transactions
                    .inc_by(shard_metrics.total_transactions as f64);
                prometheus_shard
                    .total_blocks
                    .inc_by(shard_metrics.total_blocks as f64);
                prometheus_shard
                    .average_block_time
                    .observe(shard_metrics.average_block_time.as_secs_f64());
                prometheus_shard
                    .transaction_rate
                    .set(shard_metrics.transaction_rate);
                prometheus_shard.block_rate.set(shard_metrics.block_rate);
                prometheus_shard
                    .shard_size
                    .set(shard_metrics.shard_size as f64);
                prometheus_shard
                    .cross_shard_messages
                    .inc_by(shard_metrics.cross_shard_messages as f64);
            }
        }

        Ok(())
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> NetworkMetrics {
        self.metrics.read().await.clone()
    }

    /// Get metrics history
    pub async fn get_history(&self) -> Vec<(Instant, NetworkMetrics)> {
        self.history.read().await.iter().cloned().collect()
    }

    /// Get metrics for a specific time range
    pub async fn get_metrics_range(
        &self,
        start: Instant,
        end: Instant,
    ) -> Vec<(Instant, NetworkMetrics)> {
        self.history
            .read()
            .await
            .iter()
            .filter(|(time, _)| *time >= start && *time <= end)
            .cloned()
            .collect()
    }

    /// Get shard metrics
    pub async fn get_shard_metrics(&self, shard_id: u64) -> Option<ShardMetrics> {
        self.metrics
            .read()
            .await
            .shard_metrics
            .get(&shard_id)
            .cloned()
    }

    /// Get peer metrics
    pub async fn get_peer_metrics(&self, peer_id: &str) -> Option<PeerMetrics> {
        self.metrics.read().await.peer_metrics.get(peer_id).cloned()
    }

    /// Get sync status
    pub async fn get_sync_status(&self) -> SyncStatus {
        self.metrics.read().await.sync_status.clone()
    }

    /// Clean up old metrics
    pub async fn cleanup_old_metrics(&self) -> Result<()> {
        let mut history = self.history.write().await;
        let now = Instant::now();

        while let Some((time, _)) = history.front() {
            if now.duration_since(*time) > self.config.metrics_retention {
                history.pop_front();
            } else {
                break;
            }
        }

        Ok(())
    }
}

impl PeerMetrics {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            connected_since: SerializableInstant {
                instant: Instant::now(),
            },
            last_seen: SerializableInstant {
                instant: Instant::now(),
            },
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            average_latency: 0.0,
        }
    }

    pub fn update_latency(&mut self, latency: Duration) {
        let latency_ms = latency.as_secs_f64() * 1000.0;
        self.average_latency = (self.average_latency + latency_ms) / 2.0;
        self.last_seen = SerializableInstant {
            instant: Instant::now(),
        };
    }

    pub fn record_message_sent(&mut self, bytes: usize) {
        self.messages_sent += 1;
        self.bytes_sent += bytes as u64;
        self.last_seen = SerializableInstant {
            instant: Instant::now(),
        };
    }

    pub fn record_message_received(&mut self, bytes: usize) {
        self.messages_received += 1;
        self.bytes_received += bytes as u64;
        self.last_seen = SerializableInstant {
            instant: Instant::now(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_network_metrics() -> Result<(), prometheus::Error> {
        let metrics = NetworkMetrics {
            total_peers: 10,
            active_peers: 5,
            total_messages: 100,
            total_bytes: 1000,
            average_latency: Duration::from_millis(100),
            message_rate: 10.0,
            bandwidth_usage: 1000.0,
            error_rate: 0.01,
            sync_status: SyncStatus {
                is_syncing: true,
                current_height: 100,
                target_height: 200,
                sync_progress: 0.5,
                sync_speed: 10.0,
                estimated_time_remaining: Duration::from_secs(10),
            },
            shard_metrics: HashMap::new(),
            peer_metrics: HashMap::new(),
            connected_peers: Gauge::new("connected_peers", "Number of connected peers")?,
            messages_sent: Counter::new("messages_sent", "Total messages sent")?,
            messages_received: Counter::new("messages_received", "Total messages received")?,
            bytes_sent: Counter::new("bytes_sent", "Total bytes sent")?,
            bytes_received: Counter::new("bytes_received", "Total bytes received")?,
            block_propagation_time: Histogram::with_opts(
                HistogramOpts::new("block_propagation_time", "Block propagation time")
                    .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
            )?,
            transaction_propagation_time: Histogram::with_opts(
                HistogramOpts::new(
                    "transaction_propagation_time",
                    "Transaction propagation time",
                )
                .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
            )?,
        };

        assert_eq!(metrics.total_peers, 10);
        assert_eq!(metrics.active_peers, 5);
        assert_eq!(metrics.total_messages, 100);
        assert_eq!(metrics.total_bytes, 1000);
        assert_eq!(metrics.average_latency, Duration::from_millis(100));
        assert_eq!(metrics.message_rate, 10.0);
        assert_eq!(metrics.bandwidth_usage, 1000.0);
        assert_eq!(metrics.error_rate, 0.01);
        assert_eq!(metrics.sync_status.is_syncing, true);
        assert_eq!(metrics.sync_status.current_height, 100);
        assert_eq!(metrics.sync_status.target_height, 200);
        assert_eq!(metrics.sync_status.sync_progress, 0.5);
        assert_eq!(metrics.sync_status.sync_speed, 10.0);
        assert_eq!(
            metrics.sync_status.estimated_time_remaining,
            Duration::from_secs(10)
        );

        Ok(())
    }

    #[test]
    fn test_peer_metrics() {
        let mut metrics = PeerMetrics::new("test_node".to_string());

        // Test latency update
        metrics.update_latency(Duration::from_millis(100));
        assert!(metrics.average_latency > 0.0);

        // Test message recording
        metrics.record_message_sent(100);
        metrics.record_message_received(200);
        assert_eq!(metrics.messages_sent, 1);
        assert_eq!(metrics.messages_received, 1);
        assert_eq!(metrics.bytes_sent, 100);
        assert_eq!(metrics.bytes_received, 200);
    }
}
