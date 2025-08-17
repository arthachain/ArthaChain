# ArthChain Explorer Frontend

A modern, real-time blockchain explorer for the ArthChain network built with React, Vite, and Tailwind CSS.

## 🚀 Quick Start

### Development

```bash
# Install dependencies (use npm if pnpm not available)
npm install --legacy-peer-deps

# Start development server
npm run dev
```

### Production Build

```bash
# Build for production
npm run build:production

# Test production build locally
npm run preview
```

### Deploy to Testnet

```bash
# Deploy to testnet domains
npm run deploy:testnet
```

## 🌐 Live Deployment

The explorer is configured for deployment on:
- **Primary**: https://testnet.arthachain.online
- **Secondary**: https://testnet.arthachain.in

## ✅ Features Implemented

### Real-Time Data
- ✅ Connected to live ArthChain testnet APIs
- ✅ Removed all mock data
- ✅ Real-time block and transaction updates
- ✅ Live network statistics

### Explorer Features
- ✅ **Blocks Page** - Browse and search blockchain blocks
- ✅ **Transactions Page** - View transaction details and history
- ✅ **Addresses Page** - Account information and transaction history
- ✅ **Contracts Page** - Smart contract explorer (EVM & WASM)
- ✅ **Dashboard** - Network overview with key statistics
- ✅ **Analytics** - Network charts and insights
- ✅ **Search** - Universal search for blocks, transactions, addresses

### Technical Features
- ✅ Mobile-responsive design
- ✅ Modern React components with Shadcn/ui
- ✅ Real-time data fetching
- ✅ Production-optimized builds
- ✅ SEO-friendly meta tags
- ✅ Progressive Web App capabilities

## 🔧 Configuration

### Environment Variables

The app uses these environment configurations:

**Production (.env.production)**:
```env
VITE_API_BASE_URL=https://testnet.arthachain.online
VITE_NETWORK_NAME=ArthChain Testnet
VITE_NETWORK_ID=201766
VITE_CHAIN_ID=0x31426
```

### API Endpoints

Connected to ArthChain testnet API endpoints:
- `/api/status` - Network status
- `/api/blocks/*` - Block data
- `/api/transactions/*` - Transaction data
- `/api/accounts/*` - Address information
- `/api/consensus/*` - Consensus data
- `/api/ai/*` - AI engine status

## 📦 Deployment

### Option 1: Automated Deployment Script

```bash
npm run deploy:testnet
```

This creates `arthachain-explorer-testnet.tar.gz` ready for web server deployment.

### Option 2: Manual Build and Deploy

```bash
# 1. Build the project
npm run build:production

# 2. Upload the dist/ folder contents to your web server
# 3. Configure web server (see DEPLOYMENT.md for details)
```

### Web Server Configuration

See `DEPLOYMENT.md` for detailed nginx and Apache configurations including:
- SSL/HTTPS setup
- Client-side routing support
- CORS headers for API calls
- Security headers
- Performance optimizations

## 🛠 Development

### Project Structure

```
src/
├── components/          # React components
│   ├── explorer/       # Core explorer pages
│   ├── analytics/      # Analytics and charts
│   ├── layout/         # Layout components
│   ├── search/         # Search functionality
│   ├── tools/          # Utility tools
│   └── ui/            # UI components (Shadcn)
├── services/           # API services
├── stores/            # State management (Zustand)
├── constants/         # Configuration constants
├── utils/             # Utility functions
└── hooks/             # Custom React hooks
```

### Key Components

- **Dashboard** - Network overview with real-time stats
- **BlocksPage** - Block explorer with search and pagination
- **TransactionsPage** - Transaction viewer with filtering
- **AddressesPage** - Address details and transaction history
- **ContractsPage** - Smart contract explorer
- **Analytics** - Network analytics and charts

### API Service

The `apiService.js` handles all blockchain API interactions:
- Network statistics
- Block and transaction data
- Address information
- Search functionality
- SVCP consensus data
- AI engine metrics

## 🔗 Integration

### ArthChain Network

- **Chain ID**: 201766 (0x31426)
- **Network Name**: ArthChain Testnet
- **Symbol**: ARTHA
- **RPC URL**: https://testnet.arthachain.online

### Features Supported

- **SVCP Consensus** - Social Verification Consensus Protocol
- **Quantum-Resistant** - Quantum-resistant cryptography
- **AI-Enhanced** - AI-powered fraud detection
- **Dual Runtime** - EVM and WASM smart contracts
- **Cross-Shard** - Multi-shard architecture

## 📈 Performance

- **Optimized Builds** - Code splitting and lazy loading
- **Real-time Updates** - Efficient data polling
- **Mobile Responsive** - Optimized for all devices
- **Fast Loading** - Under 3s initial load time
- **SEO Optimized** - Search engine friendly

## 🐛 Troubleshooting

### Common Issues

1. **Dependency Conflicts**
   ```bash
   npm install --legacy-peer-deps
   ```

2. **API Connection Issues**
   - Check if testnet.arthachain.online is accessible
   - Verify CORS configuration
   - Check browser network tab for errors

3. **Build Issues**
   ```bash
   npm run build -- --verbose
   ```

### Support

For issues or questions:
1. Check the browser console for errors
2. Verify API endpoints are responding
3. Ensure proper web server configuration
4. Check network connectivity

## 📄 License

Part of the ArthChain blockchain project.

---

**Live Explorer**: https://testnet.arthachain.online

Built with ❤️ for the ArthChain community
