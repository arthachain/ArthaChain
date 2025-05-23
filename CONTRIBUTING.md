# Contributing to Artha Chain

Thank you for considering contributing to Artha Chain! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a positive and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue on our GitHub repository with the following information:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots or logs (if applicable)
- Environment information (OS, Rust version, etc.)

### Suggesting Features

Feature suggestions are always welcome. Please submit an issue with:

- A clear and descriptive title
- Detailed description of the proposed feature
- Any relevant examples or mock-ups
- Explanation of why this feature would be useful

### Pull Requests

We welcome pull requests! Here's how to submit one:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a new Pull Request

### Development Workflow

1. Pick an issue to work on or create a new one
2. Discuss your approach in the issue thread
3. Fork and clone the repository
4. Set up your development environment
5. Make your changes with tests
6. Ensure all tests pass
7. Submit a pull request

## Priority Development Areas

If you're looking for areas to contribute, check our [ROADMAP.md](ROADMAP.md) for components that need implementation. Current high-priority areas include:

1. WASM Virtual Machine host functions
2. Cross-shard transaction coordination
3. Byzantine Fault Tolerance implementation
4. Formal verification system

## Development Environment Setup

### Prerequisites

- Rust 1.70 or higher
- RocksDB dependencies
- Cargo and Rustup

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/DiigooSai/ArthaChain.git
cd ArthaChain

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential librocksdb-dev libssl-dev pkg-config

# Build the project
cargo build
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific tests
cargo test -- --nocapture test_name

# Run with logging
RUST_LOG=debug cargo test
```

## Coding Guidelines

### Rust Style Guide

- Follow the [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/README.html)
- Run `cargo fmt` before committing
- Use `cargo clippy` to check for common mistakes

### Documentation

- Document all public APIs
- Include examples where appropriate
- Keep documentation up-to-date with code changes

### Tests

- Write unit tests for all new functionality
- Write integration tests for components
- Ensure tests are fast and deterministic

## Review Process

All submissions require review. We use GitHub pull requests for this purpose.

1. Submit your pull request
2. Automated tests will run
3. A maintainer will review your code
4. Feedback will be provided if changes are needed
5. Once approved, your changes will be merged

## License

By contributing to Artha Chain, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have any questions or need help, please:

- Open an issue on GitHub
- Join our [Discord](https://discord.gg/arthachain) community
- Contact us on [Telegram](https://t.me/arthachain)

Thank you for contributing to Artha Chain! 