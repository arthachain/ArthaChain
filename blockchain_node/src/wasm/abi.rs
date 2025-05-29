use crate::wasm::types::{ContractMetadata, FunctionMetadata, ParameterMetadata, WasmError};
use bincode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ABI parameter types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AbiType {
    /// Boolean
    Bool,
    /// Unsigned 8-bit integer
    U8,
    /// Unsigned 16-bit integer
    U16,
    /// Unsigned 32-bit integer
    U32,
    /// Unsigned 64-bit integer
    U64,
    /// Unsigned 128-bit integer
    U128,
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Signed 32-bit integer
    I32,
    /// Signed 64-bit integer
    I64,
    /// Signed 128-bit integer
    I128,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// String
    String,
    /// Byte array
    Bytes,
    /// Array of type
    Array(Box<AbiType>),
    /// Optional type
    Option(Box<AbiType>),
    /// Tuple of types
    Tuple(Vec<AbiType>),
    /// Map with key and value types
    Map(Box<AbiType>, Box<AbiType>),
    /// Custom struct type
    Struct(String),
    /// Enum type
    Enum(String),
}

/// ABI for a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAbi {
    /// Function name
    pub name: String,
    /// Function inputs
    pub inputs: Vec<ParameterAbi>,
    /// Function outputs
    pub outputs: Vec<ParameterAbi>,
    /// Is function read-only (view)
    pub is_view: bool,
    /// Is function payable
    pub is_payable: bool,
}

/// ABI for a parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAbi {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub type_info: AbiType,
}

/// ABI for a contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractAbi {
    /// Contract name
    pub name: String,
    /// Contract version
    pub version: String,
    /// Contract functions
    pub functions: HashMap<String, FunctionAbi>,
    /// Custom types
    pub types: HashMap<String, StructAbi>,
}

/// ABI for a struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructAbi {
    /// Struct name
    pub name: String,
    /// Struct fields
    pub fields: Vec<ParameterAbi>,
}

impl ContractAbi {
    /// Create a new contract ABI
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            functions: HashMap::new(),
            types: HashMap::new(),
        }
    }

    /// Add a function to the ABI
    pub fn add_function(&mut self, function: FunctionAbi) {
        self.functions.insert(function.name.clone(), function);
    }

    /// Add a struct type to the ABI
    pub fn add_struct(&mut self, struct_def: StructAbi) {
        self.types.insert(struct_def.name.clone(), struct_def);
    }

    /// Get a function by name
    pub fn get_function(&self, name: &str) -> Option<&FunctionAbi> {
        self.functions.get(name)
    }

    /// Serialize the ABI to JSON
    pub fn to_json(&self) -> Result<String, WasmError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| WasmError::Internal(format!("Failed to serialize ABI: {}", e)))
    }

    /// Deserialize the ABI from JSON
    pub fn from_json(json: &str) -> Result<Self, WasmError> {
        serde_json::from_str(json)
            .map_err(|e| WasmError::Internal(format!("Failed to deserialize ABI: {}", e)))
    }

    /// Convert to contract metadata
    pub fn to_metadata(&self) -> ContractMetadata {
        // Convert functions to metadata format
        let functions = self
            .functions
            .values()
            .map(|f| {
                // Convert parameters to metadata format
                let inputs = f
                    .inputs
                    .iter()
                    .map(|p| ParameterMetadata {
                        name: p.name.clone(),
                        type_name: format!("{:?}", p.type_info),
                    })
                    .collect();

                let outputs = f
                    .outputs
                    .iter()
                    .map(|p| ParameterMetadata {
                        name: p.name.clone(),
                        type_name: format!("{:?}", p.type_info),
                    })
                    .collect();

                FunctionMetadata {
                    name: f.name.clone(),
                    inputs,
                    outputs,
                    is_view: f.is_view,
                    is_payable: f.is_payable,
                }
            })
            .collect();

        // Create a dummy hash for now - this would be calculated from the contract bytecode
        let hash = [0u8; 32];

        ContractMetadata {
            name: self.name.clone(),
            version: self.version.clone(),
            author: "unknown".to_string(),
            functions,
            hash,
        }
    }
}

/// Encode function arguments according to ABI
pub fn encode_args<T: Serialize>(args: &T) -> Result<Vec<u8>, WasmError> {
    bincode::serialize(args)
        .map_err(|e| WasmError::Internal(format!("Failed to encode arguments: {}", e)))
}

/// Decode function arguments according to ABI
pub fn decode_args<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, WasmError> {
    bincode::deserialize(bytes)
        .map_err(|e| WasmError::Internal(format!("Failed to decode arguments: {}", e)))
}

/// Encode function result according to ABI
pub fn encode_result<T: Serialize>(result: &T) -> Result<Vec<u8>, WasmError> {
    bincode::serialize(result)
        .map_err(|e| WasmError::Internal(format!("Failed to encode result: {}", e)))
}

/// Decode function result according to ABI
pub fn decode_result<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, WasmError> {
    bincode::deserialize(bytes)
        .map_err(|e| WasmError::Internal(format!("Failed to decode result: {}", e)))
}
