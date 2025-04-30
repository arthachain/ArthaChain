# Artha Chain Testnet API

This documentation provides all the information needed to integrate with the Artha Chain testnet API from external applications.

## Overview

The Artha Chain testnet API allows developers to interact with the Artha Chain blockchain through a RESTful HTTP interface and WebSocket connections. This enables:

- Querying blockchain state (blocks, transactions, accounts)
- Submitting transactions to the network
- Receiving real-time updates via WebSockets
- Monitoring network status and metrics
- Requesting test tokens from the faucet

## Getting Started

1. Ensure the testnet is running (see [Testnet Setup](./docs/setup.md))
2. Use the API endpoints described in this documentation
3. Reference the example code for common integration patterns

## API Endpoints

The testnet exposes the following API endpoints:

| Validator | API Endpoint |
|-----------|--------------|
| validator1 | http://localhost:3000 |
| validator2 | http://localhost:3001 |
| validator3 | http://localhost:3002 |
| validator4 | http://localhost:3003 |

## Documentation

- [API Reference](./docs/api-reference.md) - Complete API endpoint documentation
- [Data Models](./docs/data-models.md) - Descriptions of request/response data structures
- [Authentication](./docs/authentication.md) - Authentication methods (if applicable)
- [Error Handling](./docs/error-handling.md) - Error codes and troubleshooting
- [WebSocket API](./docs/websocket.md) - Real-time event subscription
- [Faucet API](./docs/faucet.md) - Requesting test tokens

## Example Code

- [JavaScript/Node.js](./examples/javascript/README.md)
- [Python](./examples/python/README.md)

## Client SDKs

- [JavaScript SDK](./client-sdk/javascript/README.md)
- [Python SDK](./client-sdk/python/README.md)

## Support

For questions or issues related to the Testnet API, please contact the Artha Chain team or open an issue in the GitHub repository.

## License

This documentation and example code is provided under the [LICENSE](./LICENSE) terms. 