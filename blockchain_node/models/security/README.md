# Security AI Models

This directory contains configuration and model files for the blockchain node's security AI system.

## Structure

- `model_config.json`: Configuration parameters for the security scoring system
- `security_model.onnx`: (To be added) ONNX format model for security evaluation
- `feature_mapping.json`: (To be added) Mapping of input features for the model

## Usage

The AI engine loads these models during initialization. The models are used to:

1. Evaluate transaction risk
2. Score validator nodes
3. Detect potential security threats

## Development

When developing custom security models:
1. Train using your preferred ML framework
2. Export to ONNX format
3. Update the configuration files
4. Place in this directory
5. Restart the node or use the reload API

The security models should accept standardized inputs as defined in the `ai_engine/security.rs` module. 