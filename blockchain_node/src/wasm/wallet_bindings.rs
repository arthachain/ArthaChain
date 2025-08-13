use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys;

/// WASM bindings for wallet integration
#[wasm_bindgen]
pub struct ArthaChainWallet {
    chain_id: u64,
    rpc_url: String,
}

/// Wallet connection response
#[derive(Serialize, Deserialize)]
pub struct WalletConnectionResponse {
    pub connected: bool,
    pub address: Option<String>,
    pub chain_id: Option<String>,
    pub error: Option<String>,
}

/// Transaction request for WASM interface
#[derive(Serialize, Deserialize)]
pub struct WasmTransactionRequest {
    pub from: String,
    pub to: String,
    pub value: String,
    pub gas: Option<String>,
    pub gas_price: Option<String>,
    pub data: Option<String>,
}

/// Transaction response for WASM interface
#[derive(Serialize, Deserialize)]
pub struct WasmTransactionResponse {
    pub success: bool,
    pub transaction_hash: Option<String>,
    pub error: Option<String>,
}

#[wasm_bindgen]
impl ArthaChainWallet {
    /// Create a new wallet instance
    #[wasm_bindgen(constructor)]
    pub fn new(rpc_url: String) -> ArthaChainWallet {
        ArthaChainWallet {
            chain_id: 201910, // ArthaChain testnet chain ID
            rpc_url,
        }
    }

    /// Get the chain ID
    #[wasm_bindgen(getter)]
    pub fn chain_id(&self) -> u64 {
        self.chain_id
    }

    /// Get the RPC URL
    #[wasm_bindgen(getter)]
    pub fn rpc_url(&self) -> String {
        self.rpc_url.clone()
    }

    /// Connect to MetaMask or other Ethereum wallets
    #[wasm_bindgen]
    pub async fn connect_metamask(&self) -> Result<JsValue, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let ethereum = js_sys::Reflect::get(&window, &"ethereum".into())?;

        if ethereum.is_undefined() {
            return Ok(serde_wasm_bindgen::to_value(&WalletConnectionResponse {
                connected: false,
                address: None,
                chain_id: None,
                error: Some("MetaMask not detected".to_string()),
            })?);
        }

        // Request account access
        let request_method = js_sys::Reflect::get(&ethereum, &"request".into())?;
        let request_args = js_sys::Object::new();
        js_sys::Reflect::set(
            &request_args,
            &"method".into(),
            &"eth_requestAccounts".into(),
        )?;

        let accounts_promise = js_sys::Reflect::apply(
            &request_method.into(),
            &ethereum,
            &js_sys::Array::of1(&request_args),
        )?;

        let accounts =
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&accounts_promise))
                .await?;

        // Check if we're on the right chain
        let chain_id_method = js_sys::Reflect::get(&ethereum, &"request".into())?;
        let chain_id_args = js_sys::Object::new();
        js_sys::Reflect::set(&chain_id_args, &"method".into(), &"eth_chainId".into())?;

        let chain_id_promise = js_sys::Reflect::apply(
            &chain_id_method.into(),
            &ethereum,
            &js_sys::Array::of1(&chain_id_args),
        )?;

        let current_chain_id =
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&chain_id_promise))
                .await?;

        let accounts_array: js_sys::Array = accounts.into();
        let first_account = accounts_array.get(0);

        Ok(serde_wasm_bindgen::to_value(&WalletConnectionResponse {
            connected: !first_account.is_undefined(),
            address: if !first_account.is_undefined() {
                Some(first_account.as_string().unwrap_or_default())
            } else {
                None
            },
            chain_id: Some(current_chain_id.as_string().unwrap_or_default()),
            error: None,
        })?)
    }

    /// Add ArthaChain network to MetaMask
    #[wasm_bindgen]
    pub async fn add_network_to_metamask(&self) -> Result<JsValue, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let ethereum = js_sys::Reflect::get(&window, &"ethereum".into())?;

        if ethereum.is_undefined() {
            return Err("MetaMask not detected".into());
        }

        let request_method = js_sys::Reflect::get(&ethereum, &"request".into())?;

        // Create network configuration
        let network_config = js_sys::Object::new();
        js_sys::Reflect::set(&network_config, &"chainId".into(), &"0x31426".into())?; // 201910 in hex
        js_sys::Reflect::set(
            &network_config,
            &"chainName".into(),
            &"ArthaChain Testnet".into(),
        )?;

        let rpc_urls = js_sys::Array::new();
        rpc_urls.push(&self.rpc_url.clone().into());
        js_sys::Reflect::set(&network_config, &"rpcUrls".into(), &rpc_urls)?;

        let native_currency = js_sys::Object::new();
        js_sys::Reflect::set(&native_currency, &"name".into(), &"ARTHA".into())?;
        js_sys::Reflect::set(&native_currency, &"symbol".into(), &"ARTHA".into())?;
        js_sys::Reflect::set(&native_currency, &"decimals".into(), &18.into())?;
        js_sys::Reflect::set(&network_config, &"nativeCurrency".into(), &native_currency)?;

        let block_explorer_urls = js_sys::Array::new();
        block_explorer_urls.push(&"https://testnet.arthachain.online".into());
        js_sys::Reflect::set(
            &network_config,
            &"blockExplorerUrls".into(),
            &block_explorer_urls,
        )?;

        let request_params = js_sys::Array::new();
        request_params.push(&network_config);

        let request_args = js_sys::Object::new();
        js_sys::Reflect::set(
            &request_args,
            &"method".into(),
            &"wallet_addEthereumChain".into(),
        )?;
        js_sys::Reflect::set(&request_args, &"params".into(), &request_params)?;

        let add_network_promise = js_sys::Reflect::apply(
            &request_method.into(),
            &ethereum,
            &js_sys::Array::of1(&request_args),
        )?;

        let _result =
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&add_network_promise))
                .await?;

        Ok(JsValue::from(true))
    }

    /// Switch to ArthaChain network
    #[wasm_bindgen]
    pub async fn switch_to_arthachain(&self) -> Result<JsValue, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let ethereum = js_sys::Reflect::get(&window, &"ethereum".into())?;

        if ethereum.is_undefined() {
            return Err("MetaMask not detected".into());
        }

        let request_method = js_sys::Reflect::get(&ethereum, &"request".into())?;

        let switch_params = js_sys::Array::new();
        let chain_param = js_sys::Object::new();
        js_sys::Reflect::set(&chain_param, &"chainId".into(), &"0x31426".into())?; // 201910 in hex
        switch_params.push(&chain_param);

        let request_args = js_sys::Object::new();
        js_sys::Reflect::set(
            &request_args,
            &"method".into(),
            &"wallet_switchEthereumChain".into(),
        )?;
        js_sys::Reflect::set(&request_args, &"params".into(), &switch_params)?;

        let switch_promise = js_sys::Reflect::apply(
            &request_method.into(),
            &ethereum,
            &js_sys::Array::of1(&request_args),
        )?;

        match wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&switch_promise)).await
        {
            Ok(_) => Ok(JsValue::from(true)),
            Err(_) => {
                // If switch fails, try to add the network
                self.add_network_to_metamask().await
            }
        }
    }

    /// Send a transaction through the connected wallet
    #[wasm_bindgen]
    pub async fn send_transaction(&self, transaction_json: &str) -> Result<JsValue, JsValue> {
        let transaction: WasmTransactionRequest = serde_json::from_str(transaction_json)
            .map_err(|e| format!("Invalid transaction format: {}", e))?;

        let window = web_sys::window().ok_or("No window object")?;
        let ethereum = js_sys::Reflect::get(&window, &"ethereum".into())?;

        if ethereum.is_undefined() {
            return Ok(serde_wasm_bindgen::to_value(&WasmTransactionResponse {
                success: false,
                transaction_hash: None,
                error: Some("MetaMask not detected".to_string()),
            })?);
        }

        let request_method = js_sys::Reflect::get(&ethereum, &"request".into())?;

        // Create transaction object
        let tx_params = js_sys::Array::new();
        let tx_object = js_sys::Object::new();

        js_sys::Reflect::set(&tx_object, &"from".into(), &transaction.from.into())?;
        js_sys::Reflect::set(&tx_object, &"to".into(), &transaction.to.into())?;
        js_sys::Reflect::set(&tx_object, &"value".into(), &transaction.value.into())?;

        if let Some(gas) = &transaction.gas {
            js_sys::Reflect::set(&tx_object, &"gas".into(), &gas.clone().into())?;
        }

        if let Some(gas_price) = &transaction.gas_price {
            js_sys::Reflect::set(&tx_object, &"gasPrice".into(), &gas_price.clone().into())?;
        }

        if let Some(data) = &transaction.data {
            js_sys::Reflect::set(&tx_object, &"data".into(), &data.clone().into())?;
        }

        tx_params.push(&tx_object);

        let request_args = js_sys::Object::new();
        js_sys::Reflect::set(
            &request_args,
            &"method".into(),
            &"eth_sendTransaction".into(),
        )?;
        js_sys::Reflect::set(&request_args, &"params".into(), &tx_params)?;

        let send_promise = js_sys::Reflect::apply(
            &request_method.into(),
            &ethereum,
            &js_sys::Array::of1(&request_args),
        )?;

        match wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&send_promise)).await {
            Ok(tx_hash) => Ok(serde_wasm_bindgen::to_value(&WasmTransactionResponse {
                success: true,
                transaction_hash: Some(tx_hash.as_string().unwrap_or_default()),
                error: None,
            })?),
            Err(e) => Ok(serde_wasm_bindgen::to_value(&WasmTransactionResponse {
                success: false,
                transaction_hash: None,
                error: Some(format!("Transaction failed: {:?}", e)),
            })?),
        }
    }

    /// Get account balance
    #[wasm_bindgen]
    pub async fn get_balance(&self, address: &str) -> Result<JsValue, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let ethereum = js_sys::Reflect::get(&window, &"ethereum".into())?;

        if ethereum.is_undefined() {
            return Err("MetaMask not detected".into());
        }

        let request_method = js_sys::Reflect::get(&ethereum, &"request".into())?;

        let params = js_sys::Array::new();
        params.push(&address.into());
        params.push(&"latest".into());

        let request_args = js_sys::Object::new();
        js_sys::Reflect::set(&request_args, &"method".into(), &"eth_getBalance".into())?;
        js_sys::Reflect::set(&request_args, &"params".into(), &params)?;

        let balance_promise = js_sys::Reflect::apply(
            &request_method.into(),
            &ethereum,
            &js_sys::Array::of1(&request_args),
        )?;

        let balance =
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&balance_promise))
                .await?;

        Ok(balance)
    }
}

/// JavaScript utility functions
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Log to browser console
#[wasm_bindgen]
pub fn console_log(message: &str) {
    log(message);
}

/// Initialize the WASM wallet module
#[wasm_bindgen(start)]
pub fn init() {
    console_log("ArthaChain WASM Wallet module initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_creation() {
        let wallet = ArthaChainWallet::new("https://rpc.arthachain.online".to_string());
        assert_eq!(wallet.chain_id(), 201910);
        assert_eq!(wallet.rpc_url(), "https://rpc.arthachain.online");
    }
}
