use axum::{
    extract::{Path, State, Json},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::node::Node;
use crate::identity::{
    ArthaDID, 
    ArthaDIDDocument, 
    ArthaDIDError, 
    DIDCreationResult, 
    AuthenticationResult
};
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct CreateDIDRequest {
    pub display_name: String,
    pub password: String,
    pub face_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct CreateDIDResponse {
    pub did: String,
    pub mnemonic: String,
    pub document: ArthaDIDDocument,
}

#[derive(Debug, Deserialize)]
pub struct AuthenticateDIDRequest {
    pub did: String,
    pub password: Option<String>,
    pub mnemonic: Option<String>,
    pub face_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct AuthenticateDIDResponse {
    pub authenticated: bool,
    pub document: Option<ArthaDIDDocument>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Create a new Artha DID
pub async fn create_did(
    State(node): State<Arc<Node>>,
    Json(request): Json<CreateDIDRequest>,
) -> Response {
    let did_manager = match node.did_manager() {
        Some(manager) => manager,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "DID manager not initialized".to_string(),
                }),
            )
                .into_response();
        }
    };

    let result = did_manager
        .create_did(
            &request.display_name,
            &request.password,
            request.face_embedding,
        )
        .await;

    match result {
        Ok(DIDCreationResult {
            did,
            mnemonic,
            document,
        }) => (
            StatusCode::CREATED,
            Json(CreateDIDResponse {
                did: did.to_string(),
                mnemonic,
                document,
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Resolve a DID document
pub async fn resolve_did(
    State(node): State<Arc<Node>>,
    Path(did_str): Path<String>,
) -> Response {
    let did_manager = match node.did_manager() {
        Some(manager) => manager,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "DID manager not initialized".to_string(),
                }),
            )
                .into_response();
        }
    };

    let did = match ArthaDID::from_str(&did_str) {
        Ok(did) => did,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid DID: {}", err),
                }),
            )
                .into_response();
        }
    };

    match did_manager.resolve(&did).await {
        Ok(document) => (StatusCode::OK, Json(document)).into_response(),
        Err(ArthaDIDError::NotFound) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("DID not found: {}", did_str),
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Authenticate a DID
pub async fn authenticate_did(
    State(node): State<Arc<Node>>,
    Json(request): Json<AuthenticateDIDRequest>,
) -> Response {
    let did_manager = match node.did_manager() {
        Some(manager) => manager,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "DID manager not initialized".to_string(),
                }),
            )
                .into_response();
        }
    };

    let did = match ArthaDID::from_str(&request.did) {
        Ok(did) => did,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid DID: {}", err),
                }),
            )
                .into_response();
        }
    };

    let result = did_manager
        .authenticate(
            &did,
            request.password.as_deref(),
            request.mnemonic.as_deref(),
            request.face_embedding,
        )
        .await;

    match result {
        Ok(AuthenticationResult {
            authenticated,
            document,
        }) => (
            StatusCode::OK,
            Json(AuthenticateDIDResponse {
                authenticated,
                document,
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
            .into_response(),
    }
} 