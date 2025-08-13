use anyhow::Result;
use arthachain_node::config::NodeConfig;
use arthachain_node::node::Node;
use log::{error, info};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("🚀 Starting ArthaChain Node");
    info!("   ⚡ Production-grade blockchain with AI-native features");
    info!("   🛡️ Quantum-resistant cryptography");
    info!("   🧠 Real neural networks and self-learning");
    info!("   📊 Real-time performance monitoring");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let config_path = args
        .get(1)
        .unwrap_or(&"node_config.toml".to_string())
        .clone();

    info!("📋 Loading configuration from: {}", config_path);

    // Load configuration
    let config = match NodeConfig::load_from_file(&config_path).await {
        Ok(config) => {
            info!("✅ Configuration loaded successfully");
            config
        }
        Err(e) => {
            error!("❌ Failed to load configuration: {}", e);
            info!("🔧 Creating default configuration...");
            NodeConfig::default()
        }
    };

    // Create and initialize node
    let mut node = Node::new(config).await?;

    info!("🔧 Initializing node components...");
    node.init_node().await?;

    info!("🌐 Starting network layer...");
    node.start_network().await?;

    info!("💾 Initializing storage...");
    node.init_storage().await?;

    info!("🧠 Starting AI engine...");
    node.start_ai_engine().await?;

    info!("⚖️  Starting consensus...");
    node.start_consensus().await?;

    info!("📊 Starting monitoring...");
    node.start_monitoring().await?;

    info!("🎉 ArthaChain Node started successfully!");
    info!("   📡 Network: Active");
    info!("   💾 Storage: Ready");
    info!("   🧠 AI Engine: Learning");
    info!("   ⚖️  Consensus: Participating");
    info!("   📊 Monitoring: Collecting metrics");

    // Keep the node running
    tokio::signal::ctrl_c().await?;

    info!("🛑 Shutdown signal received");
    info!("🔄 Gracefully shutting down node...");

    node.shutdown()
        .await
        .map_err(|e| anyhow::anyhow!("Shutdown failed: {}", e))?;

    info!("✅ ArthaChain Node shutdown complete");
    Ok(())
}
