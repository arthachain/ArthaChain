use axum::{
    extract::{Extension, Json, Path, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::gas_free::{GasFreeApp, GasFreeManager, GasFreeTxRequest};

/// Gas-Free Application Registration Request
#[derive(Debug, Deserialize)]
pub struct GasFreeAppRequest {
    pub app_id: String,
    pub company_name: String,
    pub app_type: String,
    pub duration: u64,
    pub max_tx_per_day: u64,
    pub gas_limit_per_tx: u64,
    pub allowed_tx_types: Vec<String>,
    pub company_signature: Vec<u8>,
}

/// Gas-Free Transaction Request
#[derive(Debug, Deserialize)]
pub struct GasFreeTransactionRequest {
    pub app_id: String,
    pub from_address: Vec<u8>,
    pub to_address: Vec<u8>,
    pub data: Vec<u8>,
    pub value: u64,
    pub gas_limit: u64,
    pub tx_type: String,
}

/// Gas-Free Application Response
#[derive(Debug, Serialize)]
pub struct GasFreeAppResponse {
    pub success: bool,
    pub message: String,
    pub app: Option<GasFreeApp>,
}

/// Gas-Free Transaction Response
#[derive(Debug, Serialize)]
pub struct GasFreeTransactionResponse {
    pub success: bool,
    pub message: String,
    pub is_gas_free: bool,
    pub app_details: Option<GasFreeApp>,
    pub gas_savings: Option<u64>,
}

/// Company Management Response
#[derive(Debug, Serialize)]
pub struct CompanyManagementResponse {
    pub success: bool,
    pub message: String,
    pub apps: Vec<GasFreeApp>,
    pub total_gas_saved: u64,
    pub active_apps: u64,
}

/// Gas-Free Dashboard HTML
pub async fn gas_free_dashboard() -> impl IntoResponse {
    let html = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArthaChain Gas-Free Applications</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.5em;
        }
        .stat-card .number {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .form-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .form-section h3 {
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .apps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .app-card {
            background: white;
            border: 2px solid #e1e5e9;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .app-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        .app-card h4 {
            margin: 0 0 15px 0;
            color: #667eea;
            font-size: 1.3em;
        }
        .app-status {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }
        .status-active {
            background: #d4edda;
            color: #155724;
        }
        .status-inactive {
            background: #f8d7da;
            color: #721c24;
        }
        .hidden {
            display: none;
        }
        .company-only {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border: 2px solid #ffa726;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .company-only h4 {
            margin: 0 0 10px 0;
            color: #e65100;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ArthaChain Gas-Free Applications</h1>
            <p>Enterprise-grade gas-free transaction management system</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Active Apps</h3>
                <div class="number" id="active-apps">0</div>
                <p>Gas-free applications</p>
            </div>
            <div class="stat-card">
                <h3>Total Gas Saved</h3>
                <div class="number" id="gas-saved">0</div>
                <p>Wei saved today</p>
            </div>
            <div class="stat-card">
                <h3>Daily Transactions</h3>
                <div class="number" id="daily-tx">0</div>
                <p>Gas-free transactions</p>
            </div>
            <div class="stat-card">
                <h3>Companies</h3>
                <div class="number" id="companies">4</div>
                <p>Whitelisted companies</p>
            </div>
        </div>

        <div class="company-only">
            <h4>🔒 Company-Only Section</h4>
            <p>This section is only accessible to whitelisted companies. Gas-free applications can only be created and managed by authorized entities.</p>
        </div>

        <div class="form-section">
            <h3>🏢 Register New Gas-Free Application</h3>
            <form id="gas-free-form">
                <div class="form-group">
                    <label for="app-id">Application ID:</label>
                    <input type="text" id="app-id" name="app-id" placeholder="e.g., product_launch_2024" required>
                </div>
                <div class="form-group">
                    <label for="company-name">Company Name:</label>
                    <select id="company-name" name="company-name" required>
                        <option value="">Select Company</option>
                        <option value="ArthaChain">ArthaChain</option>
                        <option value="ArthaCorp">ArthaCorp</option>
                        <option value="ArthaLabs">ArthaLabs</option>
                        <option value="ArthaVentures">ArthaVentures</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="app-type">Application Type:</label>
                    <select id="app-type" name="app-type" required>
                        <option value="">Select Type</option>
                        <option value="CompletelyFree">Completely Free</option>
                        <option value="Discounted">Discounted Gas</option>
                        <option value="LimitedFree">Limited Free</option>
                        <option value="SelectiveFree">Selective Free</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="duration">Duration (days):</label>
                    <input type="number" id="duration" name="duration" placeholder="30" min="1" max="365" required>
                </div>
                <div class="form-group">
                    <label for="max-tx-per-day">Max Transactions/Day:</label>
                    <input type="number" id="max-tx-per-day" name="max-tx-per-day" placeholder="1000" min="1" required>
                </div>
                <div class="form-group">
                    <label for="gas-limit-per-tx">Gas Limit per Transaction:</label>
                    <input type="number" id="gas-limit-per-tx" name="gas-limit-per-tx" placeholder="21000" min="1" required>
                </div>
                <div class="form-group">
                    <label for="allowed-tx-types">Allowed Transaction Types:</label>
                    <textarea id="allowed-tx-types" name="allowed-tx-types" placeholder="transfer,contract_call,contract_deploy" rows="3"></textarea>
                </div>
                <div class="form-group">
                    <label for="company-signature">Company Signature (Hex):</label>
                    <input type="text" id="company-signature" name="company-signature" placeholder="0x01020304..." required>
                </div>
                <button type="submit" class="btn">🚀 Register Application</button>
            </form>
        </div>

        <div class="form-section">
            <h3>💳 Check Gas-Free Eligibility</h3>
            <form id="eligibility-form">
                <div class="form-group">
                    <label for="check-app-id">Application ID:</label>
                    <input type="text" id="check-app-id" name="check-app-id" placeholder="Enter app ID to check" required>
                </div>
                <div class="form-group">
                    <label for="from-address">From Address:</label>
                    <input type="text" id="from-address" name="from-address" placeholder="0x..." required>
                </div>
                <div class="form-group">
                    <label for="to-address">To Address:</label>
                    <input type="text" id="to-address" name="to-address" placeholder="0x..." required>
                </div>
                <div class="form-group">
                    <label for="tx-type">Transaction Type:</label>
                    <input type="text" id="tx-type" name="tx-type" placeholder="transfer" required>
                </div>
                <div class="form-group">
                    <label for="gas-limit">Gas Limit:</label>
                    <input type="number" id="gas-limit" name="gas-limit" placeholder="21000" required>
                </div>
                <button type="submit" class="btn btn-secondary">🔍 Check Eligibility</button>
            </form>
        </div>

        <div class="form-section">
            <h3>📊 Active Gas-Free Applications</h3>
            <div id="active-apps-list" class="apps-grid">
                <p>Loading applications...</p>
            </div>
        </div>
    </div>

    <script>
        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
            loadActiveApps();
        });

        // Gas-Free Application Registration
        document.getElementById('gas-free-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                app_id: formData.get('app-id'),
                company_name: formData.get('company-name'),
                app_type: formData.get('app-type'),
                duration: parseInt(formData.get('duration')),
                max_tx_per_day: parseInt(formData.get('max-tx-per-day')),
                gas_limit_per_tx: parseInt(formData.get('gas-limit-per-tx')),
                allowed_tx_types: formData.get('allowed-tx-types').split(',').map(s => s.trim()),
                company_signature: formData.get('company-signature').startsWith('0x') ? 
                    formData.get('company-signature').slice(2) : formData.get('company-signature')
            };

            try {
                const response = await fetch('/api/v1/testnet/gas-free/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                if (result.success) {
                    alert('✅ Gas-free application registered successfully!');
                    e.target.reset();
                    loadStats();
                    loadActiveApps();
                } else {
                    alert('❌ Failed to register: ' + result.message);
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        });

        // Check Eligibility
        document.getElementById('eligibility-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                app_id: formData.get('check-app-id'),
                from_address: formData.get('from-address'),
                to_address: formData.get('to-address'),
                data: [],
                value: 0,
                gas_limit: parseInt(formData.get('gas-limit')),
                tx_type: formData.get('tx-type')
            };

            try {
                const response = await fetch('/api/v1/testnet/gas-free/check', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                if (result.is_gas_free) {
                    alert('🎉 Transaction is eligible for gas-free processing!');
                } else {
                    alert('ℹ️ Transaction is not eligible for gas-free processing.');
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        });

        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/v1/testnet/gas-free/stats');
                const stats = await response.json();
                
                document.getElementById('active-apps').textContent = stats.active_apps || 0;
                document.getElementById('gas-saved').textContent = stats.total_gas_saved || 0;
                document.getElementById('daily-tx').textContent = stats.daily_transactions || 0;
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        }

        // Load active applications
        async function loadActiveApps() {
            try {
                const response = await fetch('/api/v1/testnet/gas-free/apps');
                const apps = await response.json();
                
                const container = document.getElementById('active-apps-list');
                if (apps.length === 0) {
                    container.innerHTML = '<p>No active gas-free applications found.</p>';
                    return;
                }

                container.innerHTML = apps.map(app => `
                    <div class="app-card">
                        <h4>${app.app_id}</h4>
                        <p><strong>Company:</strong> ${app.company_name}</p>
                        <p><strong>Type:</strong> ${app.app_type}</p>
                        <p><strong>Daily Limit:</strong> ${app.max_tx_per_day}</p>
                        <p><strong>Gas Limit:</strong> ${app.gas_limit_per_tx}</p>
                        <span class="app-status ${app.is_active ? 'status-active' : 'status-inactive'}">
                            ${app.is_active ? 'Active' : 'Inactive'}
                        </span>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load apps:', error);
                document.getElementById('active-apps-list').innerHTML = '<p>Failed to load applications.</p>';
            }
        }
    </script>
</body>
</html>
    "#;

    (StatusCode::OK, html)
}

/// Register a new gas-free application (company only)
pub async fn register_gas_free_app(
    Extension(gas_free_manager): Extension<Arc<GasFreeManager>>,
    Json(payload): Json<GasFreeAppRequest>,
) -> impl IntoResponse {
    // Create gas-free app from request
    let app = GasFreeApp {
        app_id: payload.app_id,
        company_name: payload.company_name,
        app_type: match payload.app_type.as_str() {
            "CompletelyFree" => crate::gas_free::GasFreeAppType::CompletelyFree,
            "Discounted" => crate::gas_free::GasFreeAppType::Discounted { percentage: 50 },
            "LimitedFree" => crate::gas_free::GasFreeAppType::LimitedFree { max_gas: payload.gas_limit_per_tx },
            "SelectiveFree" => crate::gas_free::GasFreeAppType::SelectiveFree { operations: payload.allowed_tx_types.clone() },
            _ => crate::gas_free::GasFreeAppType::CompletelyFree,
        },
        duration: payload.duration * 86400, // Convert days to seconds
        start_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        end_time: 0, // Permanent for now
        max_tx_per_day: payload.max_tx_per_day,
        daily_tx_count: 0,
        last_reset: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        gas_limit_per_tx: payload.gas_limit_per_tx,
        allowed_tx_types: payload.allowed_tx_types,
        company_signature: hex::decode(payload.company_signature).unwrap_or_default(),
        is_active: true,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    match gas_free_manager.register_app(app.clone()).await {
        Ok(true) => {
            println!("🚀 New gas-free application registered: {}", app.app_id);
            (
                StatusCode::OK,
                Json(GasFreeAppResponse {
                    success: true,
                    message: "Gas-free application registered successfully".to_string(),
                    app: Some(app),
                }),
            )
        }
        Ok(false) => (
            StatusCode::BAD_REQUEST,
            Json(GasFreeAppResponse {
                success: false,
                message: "Company not authorized or invalid signature".to_string(),
                app: None,
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GasFreeAppResponse {
                success: false,
                message: format!("Registration failed: {}", e),
                app: None,
            }),
        ),
    }
}

/// Check if a transaction is eligible for gas-free processing
pub async fn check_gas_free_eligibility(
    Extension(gas_free_manager): Extension<Arc<GasFreeManager>>,
    Json(payload): Json<GasFreeTransactionRequest>,
) -> impl IntoResponse {
    let request = GasFreeTxRequest {
        app_id: payload.app_id,
        from_address: payload.from_address,
        to_address: payload.to_address,
        data: payload.data,
        value: payload.value,
        gas_limit: payload.gas_limit,
        tx_type: payload.tx_type,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    match gas_free_manager.is_gas_free_eligible(&request).await {
        Ok(Some(app)) => {
            let gas_savings = match app.app_type {
                crate::gas_free::GasFreeAppType::CompletelyFree => Some(payload.gas_limit * 20), // Assume 20 gwei gas price
                crate::gas_free::GasFreeAppType::Discounted { percentage } => {
                    Some((payload.gas_limit * 20 * percentage as u64) / 100)
                }
                _ => Some(0),
            };

            (
                StatusCode::OK,
                Json(GasFreeTransactionResponse {
                    success: true,
                    message: "Transaction is eligible for gas-free processing".to_string(),
                    is_gas_free: true,
                    app_details: Some(app),
                    gas_savings,
                }),
            )
        }
        Ok(None) => (
            StatusCode::OK,
            Json(GasFreeTransactionResponse {
                success: true,
                message: "Transaction is not eligible for gas-free processing".to_string(),
                is_gas_free: false,
                app_details: None,
                gas_savings: None,
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GasFreeTransactionResponse {
                success: false,
                message: format!("Check failed: {}", e),
                is_gas_free: false,
                app_details: None,
                gas_savings: None,
            }),
        ),
    }
}

/// Get all active gas-free applications
pub async fn get_active_gas_free_apps(
    Extension(gas_free_manager): Extension<Arc<GasFreeManager>>,
) -> impl IntoResponse {
    match gas_free_manager.get_active_apps().await {
        Ok(apps) => (StatusCode::OK, Json(apps)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Vec::<GasFreeApp>::new()),
        ),
    }
}

/// Get gas-free statistics
pub async fn get_gas_free_stats(
    Extension(gas_free_manager): Extension<Arc<GasFreeManager>>,
) -> impl IntoResponse {
    let apps = match gas_free_manager.get_active_apps().await {
        Ok(apps) => apps,
        Err(_) => vec![],
    };

    let stats = serde_json::json!({
        "active_apps": apps.len(),
        "total_gas_saved": 0, // Would be calculated from actual transactions
        "daily_transactions": 0, // Would be calculated from daily counters
        "companies": 4, // Hardcoded whitelist count
        "total_apps": apps.len()
    });

    (StatusCode::OK, Json(stats))
}

/// Process a gas-free transaction
pub async fn process_gas_free_transaction(
    Extension(gas_free_manager): Extension<Arc<GasFreeManager>>,
    Json(payload): Json<GasFreeTransactionRequest>,
) -> impl IntoResponse {
    let request = GasFreeTxRequest {
        app_id: payload.app_id,
        from_address: payload.from_address,
        to_address: payload.to_address,
        data: payload.data,
        value: payload.value,
        gas_limit: payload.gas_limit,
        tx_type: payload.tx_type,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    match gas_free_manager.process_gas_free_tx(&request).await {
        Ok(true) => (
            StatusCode::OK,
            Json(GasFreeTransactionResponse {
                success: true,
                message: "Gas-free transaction processed successfully".to_string(),
                is_gas_free: true,
                app_details: None,
                gas_savings: Some(payload.gas_limit * 20), // Assume 20 gwei gas price
            }),
        ),
        Ok(false) => (
            StatusCode::BAD_REQUEST,
            Json(GasFreeTransactionResponse {
                success: false,
                message: "Transaction not eligible for gas-free processing".to_string(),
                is_gas_free: false,
                app_details: None,
                gas_savings: None,
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GasFreeTransactionResponse {
                success: false,
                message: format!("Processing failed: {}", e),
                is_gas_free: false,
                app_details: None,
                gas_savings: None,
            }),
        ),
    }
}
