use std::sync::Arc;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use log::{debug, warn, error};
use wasmer::{Instance, Module, Store, Value, Function, FunctionType, Type, Memory, MemoryType, Imports};
use wasmer::AsStoreRef;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::wasm::types::{WasmContractAddress, WasmError, WasmExecutionResult};
use crate::storage::Storage;
use crate::crypto::hash::Hash;

/// Debug error
#[derive(Debug, Error)]
pub enum DebugError {
    #[error("Invalid breakpoint: {0}")]
    InvalidBreakpoint(String),
    #[error("Breakpoint not found: {0}")]
    BreakpointNotFound(String),
    #[error("Stack trace error: {0}")]
    StackTraceError(String),
    #[error("Debug session error: {0}")]
    DebugSessionError(String),
}

/// Breakpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    /// Function name
    pub function: String,
    /// Instruction offset
    pub offset: u32,
    /// Condition (optional)
    pub condition: Option<String>,
    /// Hit count
    pub hit_count: u32,
    /// Enabled
    pub enabled: bool,
}

/// Stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// Instruction offset
    pub offset: u32,
    /// Local variables
    pub locals: HashMap<String, Value>,
    /// Arguments
    pub arguments: Vec<Value>,
}

/// Stack trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackTrace {
    /// Stack frames
    pub frames: Vec<StackFrame>,
    /// Current instruction
    pub current_instruction: u32,
    /// Memory state
    pub memory_state: Vec<u8>,
}

/// Debug session
pub struct DebugSession {
    /// Contract address
    contract_address: WasmContractAddress,
    /// Wasmer store
    store: Store,
    /// Contract module
    module: Module,
    /// Contract instance
    instance: Instance,
    /// Breakpoints
    breakpoints: HashMap<String, Vec<Breakpoint>>,
    /// Stack trace
    stack_trace: Option<StackTrace>,
    /// Debug mode
    debug_mode: bool,
    /// Step mode
    step_mode: bool,
    /// Last instruction
    last_instruction: u32,
}

impl DebugSession {
    /// Create a new debug session
    pub fn new(
        contract_address: WasmContractAddress,
        store: Store,
        module: Module,
        instance: Instance,
    ) -> Self {
        Self {
            contract_address,
            store,
            module,
            instance,
            breakpoints: HashMap::new(),
            stack_trace: None,
            debug_mode: false,
            step_mode: false,
            last_instruction: 0,
        }
    }

    /// Enable debug mode
    pub fn enable_debug_mode(&mut self) {
        self.debug_mode = true;
    }

    /// Disable debug mode
    pub fn disable_debug_mode(&mut self) {
        self.debug_mode = false;
    }

    /// Enable step mode
    pub fn enable_step_mode(&mut self) {
        self.step_mode = true;
    }

    /// Disable step mode
    pub fn disable_step_mode(&mut self) {
        self.step_mode = false;
    }

    /// Add breakpoint
    pub fn add_breakpoint(&mut self, breakpoint: Breakpoint) -> Result<(), DebugError> {
        let function_breakpoints = self.breakpoints
            .entry(breakpoint.function.clone())
            .or_insert_with(Vec::new);

        // Check if breakpoint already exists
        if function_breakpoints.iter().any(|bp| bp.offset == breakpoint.offset) {
            return Err(DebugError::InvalidBreakpoint(
                format!("Breakpoint already exists at offset {}", breakpoint.offset)
            ));
        }

        function_breakpoints.push(breakpoint);
        Ok(())
    }

    /// Remove breakpoint
    pub fn remove_breakpoint(&mut self, function: &str, offset: u32) -> Result<(), DebugError> {
        let function_breakpoints = self.breakpoints.get_mut(function)
            .ok_or_else(|| DebugError::BreakpointNotFound(function.to_string()))?;

        function_breakpoints.retain(|bp| bp.offset != offset);
        Ok(())
    }

    /// Enable breakpoint
    pub fn enable_breakpoint(&mut self, function: &str, offset: u32) -> Result<(), DebugError> {
        let function_breakpoints = self.breakpoints.get_mut(function)
            .ok_or_else(|| DebugError::BreakpointNotFound(function.to_string()))?;

        for bp in function_breakpoints {
            if bp.offset == offset {
                bp.enabled = true;
                return Ok(());
            }
        }

        Err(DebugError::BreakpointNotFound(
            format!("Breakpoint not found at offset {}", offset)
        ))
    }

    /// Disable breakpoint
    pub fn disable_breakpoint(&mut self, function: &str, offset: u32) -> Result<(), DebugError> {
        let function_breakpoints = self.breakpoints.get_mut(function)
            .ok_or_else(|| DebugError::BreakpointNotFound(function.to_string()))?;

        for bp in function_breakpoints {
            if bp.offset == offset {
                bp.enabled = false;
                return Ok(());
            }
        }

        Err(DebugError::BreakpointNotFound(
            format!("Breakpoint not found at offset {}", offset)
        ))
    }

    /// Get breakpoints
    pub fn get_breakpoints(&self) -> &HashMap<String, Vec<Breakpoint>> {
        &self.breakpoints
    }

    /// Get stack trace
    pub fn get_stack_trace(&self) -> Option<&StackTrace> {
        self.stack_trace.as_ref()
    }

    /// Update stack trace
    pub fn update_stack_trace(&mut self, frame: StackFrame) {
        let mut stack_trace = self.stack_trace.take().unwrap_or_else(|| StackTrace {
            frames: Vec::new(),
            current_instruction: 0,
            memory_state: Vec::new(),
        });

        stack_trace.frames.push(frame);
        stack_trace.current_instruction = self.last_instruction;

        // Get memory state
        if let Ok(memory) = self.instance.exports.get_memory("memory") {
            let view = memory.view(&self.store);
            stack_trace.memory_state = view.data().to_vec();
        }

        self.stack_trace = Some(stack_trace);
    }

    /// Clear stack trace
    pub fn clear_stack_trace(&mut self) {
        self.stack_trace = None;
    }

    /// Step execution
    pub fn step(&mut self) -> Result<(), DebugError> {
        if !self.step_mode {
            return Err(DebugError::DebugSessionError("Step mode not enabled".to_string()));
        }

        // TODO: Implement step execution
        // This should execute one instruction and update the stack trace

        Ok(())
    }

    /// Continue execution
    pub fn continue_execution(&mut self) -> Result<(), DebugError> {
        if !self.debug_mode {
            return Err(DebugError::DebugSessionError("Debug mode not enabled".to_string()));
        }

        // TODO: Implement continue execution
        // This should continue execution until a breakpoint is hit

        Ok(())
    }

    /// Check breakpoint
    pub fn check_breakpoint(&mut self, function: &str, offset: u32) -> bool {
        if !self.debug_mode {
            return false;
        }

        if let Some(function_breakpoints) = self.breakpoints.get(function) {
            for bp in function_breakpoints {
                if bp.enabled && bp.offset == offset {
                    bp.hit_count += 1;
                    return true;
                }
            }
        }

        false
    }

    /// Get local variables
    pub fn get_local_variables(&self, frame_index: usize) -> Result<HashMap<String, Value>, DebugError> {
        if let Some(stack_trace) = &self.stack_trace {
            if let Some(frame) = stack_trace.frames.get(frame_index) {
                return Ok(frame.locals.clone());
            }
        }

        Err(DebugError::StackTraceError("Frame not found".to_string()))
    }

    /// Get arguments
    pub fn get_arguments(&self, frame_index: usize) -> Result<Vec<Value>, DebugError> {
        if let Some(stack_trace) = &self.stack_trace {
            if let Some(frame) = stack_trace.frames.get(frame_index) {
                return Ok(frame.arguments.clone());
            }
        }

        Err(DebugError::StackTraceError("Frame not found".to_string()))
    }

    /// Get memory state
    pub fn get_memory_state(&self) -> Result<Vec<u8>, DebugError> {
        if let Some(stack_trace) = &self.stack_trace {
            return Ok(stack_trace.memory_state.clone());
        }

        Err(DebugError::StackTraceError("Stack trace not available".to_string()))
    }

    /// Get current instruction
    pub fn get_current_instruction(&self) -> u32 {
        self.last_instruction
    }

    /// Set current instruction
    pub fn set_current_instruction(&mut self, instruction: u32) {
        self.last_instruction = instruction;
    }
}

/// Debug manager
pub struct DebugManager {
    /// Debug sessions
    sessions: HashMap<WasmContractAddress, DebugSession>,
}

impl DebugManager {
    /// Create a new debug manager
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Create debug session
    pub fn create_session(
        &mut self,
        contract_address: WasmContractAddress,
        store: Store,
        module: Module,
        instance: Instance,
    ) -> Result<(), DebugError> {
        if self.sessions.contains_key(&contract_address) {
            return Err(DebugError::DebugSessionError(
                format!("Debug session already exists for contract {}", contract_address)
            ));
        }

        let session = DebugSession::new(contract_address.clone(), store, module, instance);
        self.sessions.insert(contract_address, session);

        Ok(())
    }

    /// Get debug session
    pub fn get_session(&self, contract_address: &WasmContractAddress) -> Option<&DebugSession> {
        self.sessions.get(contract_address)
    }

    /// Get debug session mutably
    pub fn get_session_mut(&mut self, contract_address: &WasmContractAddress) -> Option<&mut DebugSession> {
        self.sessions.get_mut(contract_address)
    }

    /// Remove debug session
    pub fn remove_session(&mut self, contract_address: &WasmContractAddress) -> Result<(), DebugError> {
        if !self.sessions.contains_key(contract_address) {
            return Err(DebugError::DebugSessionError(
                format!("Debug session not found for contract {}", contract_address)
            ));
        }

        self.sessions.remove(contract_address);
        Ok(())
    }

    /// Get all debug sessions
    pub fn get_sessions(&self) -> &HashMap<WasmContractAddress, DebugSession> {
        &self.sessions
    }
} 