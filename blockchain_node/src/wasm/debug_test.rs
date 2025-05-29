


use wasmer::AsStoreRef;
use wasmer::{Instance, Module, Store, Type, Value};
use wasmer_compiler::Cranelift;
use wasmer_engine::Universal;

use crate::wasm::debug::{
    Breakpoint, DebugManager, LocalVariable, MemoryState, StackFrame,
};


/// Test WASM module
const TEST_WASM: &[u8] = include_bytes!("../../tests/test_contract.wasm");

#[test]
fn test_debug_session_creation() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();

    assert_eq!(session.session_id, session_id);
    assert!(!session.is_debug_mode);
    assert!(!session.is_step_mode);
    assert!(session.breakpoints.is_empty());
    assert!(session.stack_trace.is_empty());
}

#[test]
fn test_breakpoint_management() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);

    // Add breakpoint
    let breakpoint = Breakpoint {
        function_name: "test_function".to_string(),
        instruction_index: 1,
        condition: None,
    };

    debug_manager.add_breakpoint(&session_id, breakpoint.clone());
    let session = debug_manager.get_session(&session_id).unwrap();
    assert_eq!(session.breakpoints.len(), 1);

    // Disable breakpoint
    debug_manager.disable_breakpoint(&session_id, 0);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(!session.breakpoints[0].enabled);

    // Enable breakpoint
    debug_manager.enable_breakpoint(&session_id, 0);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(session.breakpoints[0].enabled);

    // Remove breakpoint
    debug_manager.remove_breakpoint(&session_id, 0);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(session.breakpoints.is_empty());
}

#[test]
fn test_stack_trace_management() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);

    // Update stack trace
    let stack_frame = StackFrame {
        function_name: "test_function".to_string(),
        instruction_index: 1,
        locals: vec![LocalVariable {
            name: "param0".to_string(),
            value: Value::I32(42),
        }],
    };

    debug_manager.update_stack_trace(&session_id, vec![stack_frame.clone()]);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert_eq!(session.stack_trace.len(), 1);
    assert_eq!(session.stack_trace[0].function_name, "test_function");

    // Update memory state
    let memory_state = MemoryState {
        address: 0,
        value: vec![1, 2, 3, 4],
    };

    debug_manager.update_memory_state(&session_id, memory_state.clone());
    let session = debug_manager.get_session(&session_id).unwrap();
    assert_eq!(session.memory_state.len(), 1);
    assert_eq!(session.memory_state[0].address, 0);
}

#[test]
fn test_debug_mode_control() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);

    // Enable debug mode
    debug_manager.enable_debug_mode(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(session.is_debug_mode);

    // Enable step mode
    debug_manager.enable_step_mode(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(session.is_step_mode);

    // Disable debug mode
    debug_manager.disable_debug_mode(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(!session.is_debug_mode);
    assert!(!session.is_step_mode);
}

#[test]
fn test_breakpoint_hit() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);

    // Add breakpoint
    let breakpoint = Breakpoint {
        function_name: "test_function".to_string(),
        instruction_index: 1,
        condition: None,
    };

    debug_manager.add_breakpoint(&session_id, breakpoint);

    // Simulate breakpoint hit
    let hit = debug_manager.check_breakpoint(&session_id, "test_function", 1);
    assert!(hit);

    // Check non-matching breakpoint
    let hit = debug_manager.check_breakpoint(&session_id, "test_function", 2);
    assert!(!hit);
}

#[test]
fn test_debug_session_removal() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);
    assert!(debug_manager.get_session(&session_id).is_some());

    debug_manager.remove_session(&session_id);
    assert!(debug_manager.get_session(&session_id).is_none());
}

#[test]
fn test_debug_session_management() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    // Create session
    debug_manager.create_session(&session_id);
    assert!(debug_manager.get_session(&session_id).is_some());

    // Try to create duplicate session
    debug_manager.create_session(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert_eq!(session.session_id, session_id);

    // Remove session
    debug_manager.remove_session(&session_id);
    assert!(debug_manager.get_session(&session_id).is_none());
}

#[test]
fn test_step_execution() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);
    debug_manager.enable_step_mode(&session_id);

    // Simulate step execution
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(session.is_step_mode);

    // Disable step mode
    debug_manager.disable_step_mode(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(!session.is_step_mode);
}

#[test]
fn test_continue_execution() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);
    debug_manager.enable_debug_mode(&session_id);

    // Simulate continue execution
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(session.is_debug_mode);

    // Disable debug mode
    debug_manager.disable_debug_mode(&session_id);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert!(!session.is_debug_mode);
}

#[test]
fn test_instruction_tracking() {
    let debug_manager = DebugManager::new();
    let session_id = "test_session".to_string();

    debug_manager.create_session(&session_id);

    // Update current instruction
    debug_manager.update_current_instruction(&session_id, "test_function", 1);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert_eq!(session.current_function, "test_function");
    assert_eq!(session.current_instruction, 1);

    // Update stack trace with current instruction
    let stack_frame = StackFrame {
        function_name: "test_function".to_string(),
        instruction_index: 1,
        locals: vec![LocalVariable {
            name: "param0".to_string(),
            value: Value::I32(42),
        }],
    };

    debug_manager.update_stack_trace(&session_id, vec![stack_frame]);
    let session = debug_manager.get_session(&session_id).unwrap();
    assert_eq!(session.stack_trace.len(), 1);
    assert_eq!(session.stack_trace[0].instruction_index, 1);
}
