use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use console::style;
use std::fs;

// Command line argument parser
#[derive(Parser)]
#[command(name = "contract-cli")]
#[command(about = "CLI tool for deploying and interacting with WASM contracts", long_about = None)]
struct Cli {
    /// Optional RPC endpoint URL (defaults to http://localhost:8545)
    #[arg(short, long, default_value = "http://localhost:8545")]
    endpoint: String,

    /// Specify the private key or keyfile path
    #[arg(short, long)]
    key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a WAT (WebAssembly Text) file to WASM binary
    Compile {
        /// Path to the WAT file
        #[arg(required = true)]
        input_file: PathBuf,
        
        /// Output WASM file path [default: input_file with .wasm extension]
        #[arg(short, long)]
        output_file: Option<PathBuf>,
    },
    
    /// Deploy a WASM contract to the blockchain
    Deploy {
        /// Path to the WASM file
        #[arg(required = true)]
        wasm_file: PathBuf,
        
        /// Optional constructor arguments (JSON format)
        #[arg(short, long)]
        args: Option<String>,
        
        /// Gas limit for deployment
        #[arg(short, long, default_value = "10000000")]
        gas_limit: u64,
    },
    
    /// Call a contract method
    Call {
        /// Contract address
        #[arg(required = true)]
        address: String,
        
        /// Method name to call
        #[arg(required = true)]
        method: String,
        
        /// Function arguments (JSON format)
        #[arg(short, long)]
        args: Option<String>,
        
        /// Value to send with transaction (in smallest denomination)
        #[arg(short, long, default_value = "0")]
        value: u64,
        
        /// Gas limit for the call
        #[arg(short, long, default_value = "1000000")]
        gas_limit: u64,
    },
    
    /// Get contract metadata
    GetMetadata {
        /// Contract address
        #[arg(required = true)]
        address: String,
    },
}

/// Response from the RPC server
#[derive(Debug, Serialize, Deserialize)]
struct RpcResponse<T> {
    jsonrpc: String,
    id: u64,
    result: Option<T>,
    error: Option<RpcError>,
}

/// RPC error structure
#[derive(Debug, Serialize, Deserialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Contract metadata structure
#[derive(Debug, Serialize, Deserialize)]
struct ContractMetadata {
    name: String,
    version: String,
    author: String,
    functions: Vec<FunctionMetadata>,
}

/// Function metadata structure
#[derive(Debug, Serialize, Deserialize)]
struct FunctionMetadata {
    name: String,
    inputs: Vec<ParameterMetadata>,
    outputs: Vec<ParameterMetadata>,
    is_view: bool,
    is_payable: bool,
}

/// Parameter metadata structure
#[derive(Debug, Serialize, Deserialize)]
struct ParameterMetadata {
    name: String,
    type_name: String,
}

/// Transaction response
#[derive(Debug, Serialize, Deserialize)]
struct TransactionResponse {
    tx_hash: String,
    contract_address: Option<String>,
    gas_used: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Compile { input_file, output_file } => {
            compile_contract(input_file, output_file)?;
        }
        Commands::Deploy { wasm_file, args, gas_limit } => {
            deploy_contract(&cli.endpoint, cli.key.as_deref(), wasm_file, args.as_deref(), *gas_limit).await?;
        }
        Commands::Call { address, method, args, value, gas_limit } => {
            call_contract(&cli.endpoint, cli.key.as_deref(), address, method, args.as_deref(), *value, *gas_limit).await?;
        }
        Commands::GetMetadata { address } => {
            get_contract_metadata(&cli.endpoint, address).await?;
        }
    }
    
    Ok(())
}

/// Compile WAT to WASM
fn compile_contract(input_file: &PathBuf, output_file: &Option<PathBuf>) -> Result<()> {
    println!("Compiling WAT to WASM: {}", input_file.display());
    
    // Read WAT file
    let wat = fs::read_to_string(input_file)
        .context(format!("Failed to read WAT file: {}", input_file.display()))?;
    
    // Compile WAT to WASM
    let wasm = wat::parse_str(&wat)
        .context("Failed to compile WAT to WASM")?;
    
    // Determine output path
    let output_path = match output_file {
        Some(path) => path.clone(),
        None => {
            let mut path = input_file.clone();
            path.set_extension("wasm");
            path
        }
    };
    
    // Write WASM file
    fs::write(&output_path, wasm)
        .context(format!("Failed to write WASM file: {}", output_path.display()))?;
    
    println!("{} Compiled successfully to {}", style("[SUCCESS]").green(), output_path.display());
    Ok(())
}

/// Deploy a contract to the blockchain
async fn deploy_contract(
    endpoint: &str, 
    key: Option<&str>, 
    wasm_file: &PathBuf, 
    args: Option<&str>, 
    gas_limit: u64
) -> Result<()> {
    println!("Deploying contract: {}", wasm_file.display());
    
    // Read WASM file
    let wasm_bytes = fs::read(wasm_file)
        .context(format!("Failed to read WASM file: {}", wasm_file.display()))?;
    
    // Convert to hex
    let wasm_hex = format!("0x{}", hex::encode(&wasm_bytes));
    
    // Parse constructor arguments if provided
    let args_hex = match args {
        Some(args_json) => {
            // In a real implementation, we would serialize the JSON args according to ABI
            // For now, we just convert the raw JSON to hex
            format!("0x{}", hex::encode(args_json.as_bytes()))
        }
        None => "0x".to_string(),
    };
    
    // Prepare private key
    let private_key = get_private_key(key)?;
    
    // Prepare JSON-RPC request
    let client = reqwest::Client::new();
    let params = serde_json::json!({
        "bytecode": wasm_hex,
        "args": args_hex,
        "gas_limit": gas_limit,
        "private_key": private_key,
    });
    
    let response = client.post(endpoint)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "wasm_deployContract",
            "params": [params]
        }))
        .send()
        .await
        .context("Failed to send deployment request")?;
    
    let rpc_response: RpcResponse<TransactionResponse> = response.json().await
        .context("Failed to parse deployment response")?;
    
    if let Some(error) = rpc_response.error {
        println!("{} {}", style("[ERROR]").red(), error.message);
        return Err(anyhow::anyhow!("Deployment failed: {}", error.message));
    }
    
    if let Some(result) = rpc_response.result {
        println!("{} Contract deployed", style("[SUCCESS]").green());
        println!("  Transaction hash: {}", result.tx_hash);
        if let Some(contract_address) = result.contract_address {
            println!("  Contract address: {}", contract_address);
        }
        println!("  Gas used: {}", result.gas_used);
    }
    
    Ok(())
}

/// Call a contract method
async fn call_contract(
    endpoint: &str,
    key: Option<&str>,
    address: &str,
    method: &str,
    args: Option<&str>,
    value: u64,
    gas_limit: u64
) -> Result<()> {
    println!("Calling contract method: {}.{}", address, method);
    
    // Parse arguments if provided
    let args_hex = match args {
        Some(args_json) => {
            // In a real implementation, we would serialize the JSON args according to ABI
            // For now, we just convert the raw JSON to hex
            format!("0x{}", hex::encode(args_json.as_bytes()))
        }
        None => "0x".to_string(),
    };
    
    // Prepare private key
    let private_key = get_private_key(key)?;
    
    // Prepare JSON-RPC request
    let client = reqwest::Client::new();
    let params = serde_json::json!({
        "to": address,
        "method": method,
        "args": args_hex,
        "value": value,
        "gas_limit": gas_limit,
        "private_key": private_key,
    });
    
    let response = client.post(endpoint)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "wasm_callContract",
            "params": [params]
        }))
        .send()
        .await
        .context("Failed to send contract call request")?;
    
    let rpc_response: RpcResponse<serde_json::Value> = response.json().await
        .context("Failed to parse contract call response")?;
    
    if let Some(error) = rpc_response.error {
        println!("{} {}", style("[ERROR]").red(), error.message);
        return Err(anyhow::anyhow!("Contract call failed: {}", error.message));
    }
    
    if let Some(result) = rpc_response.result {
        println!("{} Method call succeeded", style("[SUCCESS]").green());
        println!("  Result: {}", result);
    }
    
    Ok(())
}

/// Get contract metadata
async fn get_contract_metadata(endpoint: &str, address: &str) -> Result<()> {
    println!("Fetching contract metadata for: {}", address);
    
    // Prepare JSON-RPC request
    let client = reqwest::Client::new();
    let response = client.post(endpoint)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "wasm_getContractMetadata",
            "params": [address]
        }))
        .send()
        .await
        .context("Failed to send metadata request")?;
    
    let rpc_response: RpcResponse<ContractMetadata> = response.json().await
        .context("Failed to parse metadata response")?;
    
    if let Some(error) = rpc_response.error {
        println!("{} {}", style("[ERROR]").red(), error.message);
        return Err(anyhow::anyhow!("Failed to get metadata: {}", error.message));
    }
    
    if let Some(metadata) = rpc_response.result {
        println!("{} Contract: {} (v{})", style("[INFO]").blue(), metadata.name, metadata.version);
        println!("  Author: {}", metadata.author);
        println!("\n  Functions:");
        
        for function in metadata.functions {
            let view_tag = if function.is_view { "view" } else { "" };
            let payable_tag = if function.is_payable { "payable" } else { "" };
            let tags = [view_tag, payable_tag].iter()
                .filter(|t| !t.is_empty())
                .map(|t| format!("[{}]", t))
                .collect::<Vec<_>>()
                .join(" ");
            
            println!("    {}{}:", function.name, if !tags.is_empty() { format!(" {}", tags) } else { "".to_string() });
            
            // Print inputs
            if !function.inputs.is_empty() {
                println!("      Inputs:");
                for input in &function.inputs {
                    println!("        {}: {}", input.name, input.type_name);
                }
            }
            
            // Print outputs
            if !function.outputs.is_empty() {
                println!("      Outputs:");
                for output in &function.outputs {
                    println!("        {}: {}", output.name, output.type_name);
                }
            }
            
            println!();
        }
    }
    
    Ok(())
}

/// Get the private key from various sources
fn get_private_key(key: Option<&str>) -> Result<String> {
    match key {
        Some(k) if k.starts_with("0x") => Ok(k.to_string()),
        Some(path) => {
            // Try to read as a keyfile
            let keyfile = PathBuf::from(path);
            if keyfile.exists() {
                let content = fs::read_to_string(keyfile)
                    .context(format!("Failed to read keyfile: {}", path))?;
                // In a real implementation, this would decrypt the keyfile
                // For now, we just return the content
                Ok(content.trim().to_string())
            } else {
                // Treat as a raw private key
                Ok(format!("0x{}", path))
            }
        }
        None => {
            // Try to get from environment variable
            if let Ok(key) = std::env::var("PRIVATE_KEY") {
                Ok(key)
            } else {
                // Prompt the user for a private key
                let key = dialoguer::Password::new()
                    .with_prompt("Enter private key")
                    .interact()?;
                Ok(key)
            }
        }
    }
} 