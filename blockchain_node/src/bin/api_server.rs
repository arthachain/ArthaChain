use anyhow::Result;
use arthachain_node::api::start_api_server;
use clap::Parser;

#[derive(Parser)]
#[command(name = "arthachain-api")]
#[command(about = "ArthaChain Blockchain API Server")]
struct Args {
    /// Port to run the API server on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Network mode (mainnet, testnet)
    #[arg(short, long, default_value = "mainnet")]
    network: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("🌐 ArthaChain API Server");
    println!("Network: {}", args.network);
    println!("Port: {}", args.port);

    start_api_server(args.port).await?;

    Ok(())
}
