use anyhow::Result;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use teloxide::{prelude::*, utils::command::BotCommands};

mod faucet_client;
mod rate_limiter;
mod wallet_validator;

use faucet_client::FaucetClient;
use rate_limiter::RateLimiter;
use wallet_validator::WalletValidator;

/// ArthaChain Faucet Bot Commands
#[derive(BotCommands, Clone, Debug)]
#[command(rename_rule = "lowercase", description = "ArthaChain Faucet Bot Commands:")]
enum Command {
    #[command(description = "Start using the faucet bot")]
    Start,
    #[command(description = "Request ARTHA tokens for your wallet")]
    Faucet(String),
    #[command(description = "Check your wallet balance")]
    Balance(String),
    #[command(description = "Get faucet status and information")]
    Status,
    #[command(description = "Show help and available commands")]
    Help,
    #[command(description = "Show bot statistics")]
    Stats,
    #[command(description = "Show network information")]
    Network,
}

/// User request tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserRequest {
    user_id: UserId,
    username: Option<String>,
    wallet_address: String,
    requested_at: DateTime<Utc>,
    transaction_hash: Option<String>,
    amount: f64,
}

/// Bot state
#[derive(Clone)]
struct BotState {
    faucet_client: Arc<FaucetClient>,
    rate_limiter: Arc<RateLimiter>,
    wallet_validator: Arc<WalletValidator>,
    user_requests: Arc<DashMap<UserId, Vec<UserRequest>>>,
    bot_stats: Arc<DashMap<String, u64>>,
}

impl BotState {
    fn new(faucet_api_url: String) -> Self {
        Self {
            faucet_client: Arc::new(FaucetClient::new(faucet_api_url)),
            rate_limiter: Arc::new(RateLimiter::new()),
            wallet_validator: Arc::new(WalletValidator::new()),
            user_requests: Arc::new(DashMap::new()),
            bot_stats: Arc::new(DashMap::new()),
        }
    }

    fn increment_stat(&self, key: &str) {
        self.bot_stats.entry(key.to_string()).and_modify(|v| *v += 1).or_insert(1);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    info!("🚀 Starting ArthaChain Telegram Faucet Bot...");

    let bot_token = std::env::var("TELEGRAM_BOT_TOKEN")
        .expect("TELEGRAM_BOT_TOKEN environment variable not set");
    
    let faucet_api_url = std::env::var("FAUCET_API_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());

    let bot = Bot::new(bot_token);
    let state = BotState::new(faucet_api_url);

    info!("✅ Bot initialized. Faucet API: {}", state.faucet_client.api_url());
    
    Dispatcher::builder(bot, Update::filter_message().branch(
        dptree::entry()
            .filter_command::<Command>()
            .endpoint(handle_command)
    ))
    .dependencies(dptree::deps![state])
    .enable_ctrlc_handler()
    .build()
    .dispatch()
    .await;

    Ok(())
}

async fn handle_command(bot: Bot, msg: Message, cmd: Command, state: BotState) -> ResponseResult<()> {
    let user_id = msg.from().map(|user| user.id).unwrap_or(UserId(0));
    let username = msg.from().and_then(|user| user.username.clone());
    
    info!("Command received: {:?} from user {}", cmd, user_id);
    state.increment_stat("total_commands");

    match cmd {
        Command::Start => handle_start(bot, msg, state).await,
        Command::Faucet(wallet_address) => handle_faucet(bot, msg, wallet_address, state).await,
        Command::Balance(wallet_address) => handle_balance(bot, msg, wallet_address, state).await,
        Command::Status => handle_status(bot, msg, state).await,
        Command::Help => handle_help(bot, msg, state).await,
        Command::Stats => handle_stats(bot, msg, state).await,
        Command::Network => handle_network(bot, msg, state).await,
    }
}

async fn handle_start(bot: Bot, msg: Message, state: BotState) -> ResponseResult<()> {
    state.increment_stat("start_commands");
    
    let welcome_text = format!(
        "🎉 **Welcome to ArthaChain Faucet Bot!**\n\n\
        🪙 Get **2.0 ARTHA** tokens for testing on the ArthaChain testnet!\n\n\
        **Available Commands:**\n\
        🚿 `/faucet YOUR_WALLET_ADDRESS` - Request ARTHA tokens\n\
        💰 `/balance YOUR_WALLET_ADDRESS` - Check wallet balance\n\
        📊 `/status` - Check faucet status\n\
        🌐 `/network` - View network information\n\
        📈 `/stats` - View bot statistics\n\
        ❓ `/help` - Show this help message\n\n\
        **Example:**\n\
        `/faucet 0x742d35Cc6634C0532925a3b844Bc454e4438f44e`\n\n\
        **Features:**\n\
        ⚡ Ultra-fast transactions (20M+ TPS)\n\
        🛡️ Quantum-resistant security\n\
        🔄 Dual VM support (EVM + WASM)\n\
        💎 50x cheaper gas fees\n\n\
        Start by sending `/faucet` with your wallet address! 🚀"
    );

    bot.send_message(msg.chat.id, welcome_text)
        .parse_mode(teloxide::types::ParseMode::MarkdownV2)
        .await?;
    
    Ok(())
}

async fn handle_faucet(bot: Bot, msg: Message, wallet_address: String, state: BotState) -> ResponseResult<()> {
    let user_id = msg.from().map(|user| user.id).unwrap_or(UserId(0));
    let username = msg.from().and_then(|user| user.username.clone());
    
    state.increment_stat("faucet_requests");

    // Validate wallet address
    if !state.wallet_validator.is_valid_address(&wallet_address) {
        state.increment_stat("invalid_addresses");
        bot.send_message(
            msg.chat.id,
            "❌ **Invalid wallet address!**\n\n\
            Please provide a valid Ethereum-compatible address (42 characters starting with 0x)\n\n\
            **Example:** `0x742d35Cc6634C0532925a3b844Bc454e4438f44e`"
        ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        return Ok(());
    }

    // Check rate limiting
    if !state.rate_limiter.allow_request(user_id).await {
        state.increment_stat("rate_limited");
        let next_request = state.rate_limiter.next_allowed_time(user_id).await;
        bot.send_message(
            msg.chat.id,
            format!(
                "⏰ **Rate limit reached!**\n\n\
                You can request tokens again in **{}**\n\n\
                This helps prevent abuse and ensures fair distribution! 🛡️",
                format_duration(next_request)
            )
        ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        return Ok(());
    }

    // Send processing message
    let processing_msg = bot.send_message(
        msg.chat.id,
        "🔄 **Processing your faucet request...**\n\n\
        Please wait while we validate and process your transaction ⏳"
    ).parse_mode(teloxide::types::ParseMode::Markdown).await?;

    // Request tokens from faucet API
    match state.faucet_client.request_tokens(&wallet_address).await {
        Ok(response) => {
            state.increment_stat("successful_requests");
            
            // Record the request
            let request = UserRequest {
                user_id,
                username: username.clone(),
                wallet_address: wallet_address.clone(),
                requested_at: Utc::now(),
                transaction_hash: response.transaction_hash.clone(),
                amount: response.amount,
            };
            
            state.user_requests.entry(user_id).or_insert_with(Vec::new).push(request);

            // Update the message with success
            bot.edit_message_text(
                msg.chat.id,
                processing_msg.id,
                format!(
                    "✅ **Faucet request successful!**\n\n\
                    💰 **Amount:** {} ARTHA\n\
                    🎯 **Recipient:** `{}`\n\
                    🔗 **Transaction:** `{}`\n\
                    ⏱️ **Time:** {}\n\n\
                    🎉 **Your tokens have been sent!**\n\n\
                    Use `/balance {}` to check your wallet balance! 💎",
                    response.amount,
                    wallet_address,
                    response.transaction_hash.unwrap_or("Processing...".to_string()),
                    Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                    wallet_address
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
        Err(e) => {
            state.increment_stat("failed_requests");
            error!("Faucet request failed: {}", e);
            
            bot.edit_message_text(
                msg.chat.id,
                processing_msg.id,
                format!(
                    "❌ **Faucet request failed!**\n\n\
                    **Error:** {}\n\n\
                    Please try again later or contact support if the issue persists.\n\n\
                    Use `/status` to check faucet availability! 🔧",
                    e
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
    }

    Ok(())
}

async fn handle_balance(bot: Bot, msg: Message, wallet_address: String, state: BotState) -> ResponseResult<()> {
    state.increment_stat("balance_checks");

    if !state.wallet_validator.is_valid_address(&wallet_address) {
        bot.send_message(
            msg.chat.id,
            "❌ **Invalid wallet address!**\n\nPlease provide a valid Ethereum-compatible address."
        ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        return Ok(());
    }

    match state.faucet_client.get_balance(&wallet_address).await {
        Ok(balance) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "💰 **Wallet Balance**\n\n\
                    🎯 **Address:** `{}`\n\
                    💎 **Balance:** {} ARTHA\n\
                    ⚡ **Network:** ArthaChain Testnet\n\n\
                    Need more tokens? Use `/faucet {}` 🚿",
                    wallet_address,
                    balance,
                    wallet_address
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
        Err(e) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "❌ **Failed to check balance**\n\n\
                    **Error:** {}\n\n\
                    Please try again later! 🔄",
                    e
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
    }

    Ok(())
}

async fn handle_status(bot: Bot, msg: Message, state: BotState) -> ResponseResult<()> {
    state.increment_stat("status_checks");

    match state.faucet_client.get_status().await {
        Ok(status) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "📊 **ArthaChain Faucet Status**\n\n\
                    🟢 **Status:** {}\n\
                    💰 **Amount per request:** {} ARTHA\n\
                    ⏰ **Cooldown period:** {} minutes\n\
                    🎯 **Total distributed:** {} ARTHA\n\n\
                    **Network Information:**\n\
                    ⚡ **TPS:** 20M+ transactions per second\n\
                    🛡️ **Security:** Quantum-resistant\n\
                    🔄 **VM Support:** EVM + WASM\n\
                    💎 **Gas Fees:** 50x cheaper than Ethereum\n\n\
                    Ready to request tokens! Use `/faucet YOUR_ADDRESS` 🚀",
                    if status.is_active { "Active" } else { "Inactive" },
                    status.amount_per_request,
                    status.cooldown_minutes,
                    status.total_distributed
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
        Err(e) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "❌ **Failed to get faucet status**\n\n\
                    **Error:** {}\n\n\
                    The faucet might be temporarily unavailable. Please try again later! 🔄",
                    e
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
    }

    Ok(())
}

async fn handle_help(bot: Bot, msg: Message, state: BotState) -> ResponseResult<()> {
    state.increment_stat("help_requests");
    
    let help_text = format!(
        "❓ **ArthaChain Faucet Bot Help**\n\n\
        **🚿 Request Tokens:**\n\
        `/faucet YOUR_WALLET_ADDRESS`\n\
        • Get 2.0 ARTHA tokens for testing\n\
        • Cooldown: 24 hours between requests\n\
        • Example: `/faucet 0x742d35...44e`\n\n\
        **💰 Check Balance:**\n\
        `/balance YOUR_WALLET_ADDRESS`\n\
        • View your current ARTHA balance\n\
        • Works with any Ethereum-compatible address\n\n\
        **📊 System Information:**\n\
        `/status` - Faucet status and network info\n\
        `/network` - Detailed network statistics\n\
        `/stats` - Bot usage statistics\n\n\
        **📝 Wallet Address Format:**\n\
        • Must start with `0x`\n\
        • Must be 42 characters long\n\
        • Example: `0x742d35Cc6634C0532925a3b844Bc454e4438f44e`\n\n\
        **🌟 ArthaChain Features:**\n\
        ⚡ 20M+ TPS (Transactions Per Second)\n\
        🛡️ Quantum-resistant security\n\
        🔄 Dual VM (EVM + WASM)\n\
        💎 Ultra-low gas fees\n\
        🔗 Cross-shard transactions\n\n\
        Need more help? Contact @arthachainsupport 💬"
    );

    bot.send_message(msg.chat.id, help_text)
        .parse_mode(teloxide::types::ParseMode::MarkdownV2)
        .await?;
    
    Ok(())
}

async fn handle_stats(bot: Bot, msg: Message, state: BotState) -> ResponseResult<()> {
    state.increment_stat("stats_requests");

    let stats = state.bot_stats.clone();
    let total_users = state.user_requests.len();
    
    let stats_text = format!(
        "📈 **Bot Statistics**\n\n\
        👥 **Total Users:** {}\n\
        📊 **Commands Processed:**\n\
        • Total: {}\n\
        • Start: {}\n\
        • Faucet Requests: {}\n\
        • Balance Checks: {}\n\
        • Status Checks: {}\n\
        • Help Requests: {}\n\n\
        **🚿 Faucet Performance:**\n\
        ✅ Successful: {}\n\
        ❌ Failed: {}\n\
        ⏰ Rate Limited: {}\n\
        🚫 Invalid Addresses: {}\n\n\
        **📊 Success Rate:** {:.1}%\n\n\
        Bot uptime: Since last restart 🚀",
        total_users,
        stats.get("total_commands").map(|v| *v).unwrap_or(0),
        stats.get("start_commands").map(|v| *v).unwrap_or(0),
        stats.get("faucet_requests").map(|v| *v).unwrap_or(0),
        stats.get("balance_checks").map(|v| *v).unwrap_or(0),
        stats.get("status_checks").map(|v| *v).unwrap_or(0),
        stats.get("help_requests").map(|v| *v).unwrap_or(0),
        stats.get("successful_requests").map(|v| *v).unwrap_or(0),
        stats.get("failed_requests").map(|v| *v).unwrap_or(0),
        stats.get("rate_limited").map(|v| *v).unwrap_or(0),
        stats.get("invalid_addresses").map(|v| *v).unwrap_or(0),
        {
            let successful = stats.get("successful_requests").map(|v| *v).unwrap_or(0) as f64;
            let total = stats.get("faucet_requests").map(|v| *v).unwrap_or(1) as f64;
            (successful / total) * 100.0
        }
    );

    bot.send_message(msg.chat.id, stats_text)
        .parse_mode(teloxide::types::ParseMode::MarkdownV2)
        .await?;
    
    Ok(())
}

async fn handle_network(bot: Bot, msg: Message, state: BotState) -> ResponseResult<()> {
    state.increment_stat("network_requests");

    match state.faucet_client.get_network_info().await {
        Ok(network) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "🌐 **ArthaChain Network Information**\n\n\
                    **📊 Performance:**\n\
                    ⚡ **Current TPS:** {}\n\
                    📦 **Block Height:** {}\n\
                    ⏱️ **Block Time:** {} seconds\n\
                    🔗 **Total Transactions:** {}\n\n\
                    **🏛️ Consensus:**\n\
                    🛡️ **Mechanism:** SVCP + SVBFT\n\
                    👥 **Active Validators:** {}\n\
                    🔐 **Quantum Protection:** Enabled\n\
                    🌊 **Cross-Shard:** Enabled\n\n\
                    **💎 Features:**\n\
                    🔄 **Dual VM:** EVM + WASM\n\
                    💰 **Gas Price:** {} GWEI (50x cheaper)\n\
                    🧮 **ZK Proofs:** {} verified\n\
                    🚨 **Fraud Detection:** AI-powered\n\n\
                    **🔗 Network ID:** 201910\n\
                    **⛓️ Chain ID:** 0x31426",
                    network.current_tps,
                    network.block_height,
                    network.block_time,
                    network.total_transactions,
                    network.active_validators,
                    network.gas_price_gwei,
                    network.zkp_verifications
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
        Err(e) => {
            bot.send_message(
                msg.chat.id,
                format!(
                    "❌ **Failed to get network information**\n\n\
                    **Error:** {}\n\n\
                    Please try again later! 🔄",
                    e
                )
            ).parse_mode(teloxide::types::ParseMode::Markdown).await?;
        }
    }

    Ok(())
}

fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}