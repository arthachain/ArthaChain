//! API Routes Documentation
//!
//! This module contains documentation for the available API routes.
//! All API endpoints follow the REST convention and return JSON responses.
//!
//! # Block Endpoints
//!
//! ## Get latest block
//! - **GET** `/api/blocks/latest`
//! - Returns information about the latest block in the chain
//!
//! ## Get block by hash
//! - **GET** `/api/blocks/{hash}`
//! - Returns information about a block with the given hash
//!
//! ## Get block by height
//! - **GET** `/api/blocks/height/{height}`
//! - Returns information about a block at the given height
//!
//! # Transaction Endpoints
//!
//! ## Get transaction by hash
//! - **GET** `/api/transactions/{hash}`
//! - Returns information about a transaction with the given hash
//!
//! ## Submit transaction
//! - **POST** `/api/transactions`
//! - Submits a new transaction to the network
//! - Request body must contain a valid transaction in JSON format
//!
//! # Account Endpoints
//!
//! ## Get account information
//! - **GET** `/api/accounts/{address}`
//! - Returns information about an account with the given address
//!
//! ## Get account transactions
//! - **GET** `/api/accounts/{address}/transactions`
//! - Returns a list of transactions for the given account
//! - Supports pagination with `page` and `page_size` query parameters
//!
//! # Network Status Endpoints
//!
//! ## Get node status
//! - **GET** `/api/status`
//! - Returns information about the current node status
//!
//! ## Get connected peers
//! - **GET** `/api/network/peers`
//! - Returns a list of connected peers
//!
//! # Consensus Endpoints
//!
//! ## Get consensus status
//! - **GET** `/api/consensus/status`
//! - Returns information about the current consensus status
//!
//! # WebSocket Endpoint
//!
//! ## Real-time updates
//! - **GET** `/api/ws`
//! - WebSocket endpoint for subscribing to real-time updates
//! - Supports the following events:
//!   - `new_block`: Notifies when a new block is added to the chain
//!   - `new_transaction`: Notifies when a new transaction is added to the mempool
//!   - `consensus_update`: Notifies when the consensus state changes 