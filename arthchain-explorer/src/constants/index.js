// Constants for blockchain explorer

export const NETWORKS = {
  ARTHACHAIN: {
    id: 201766,
    name: 'ArthChain Local Node',
    symbol: 'ARTHA',
    rpcUrl: 'http://localhost:8080',
    apiUrl: 'http://localhost:8080',
    explorerUrl: 'http://localhost:3333',
    chainId: '0x31426',
    color: '#00ff88' // ArthChain's primary color
  }
};

export const DEFAULT_NETWORK = NETWORKS.ARTHACHAIN;

export const TRANSACTION_TYPES = {
  TRANSFER: 'transfer',
  CONTRACT_CALL: 'contract_call',
  CONTRACT_CREATION: 'contract_creation',
  TOKEN_TRANSFER: 'token_transfer'
};

export const TRANSACTION_STATUS = {
  SUCCESS: 'success',
  FAILED: 'failed',
  PENDING: 'pending'
};

export const ADDRESS_TYPES = {
  EOA: 'externally_owned_account',
  CONTRACT: 'contract'
};

export const SEARCH_TYPES = {
  BLOCK: 'block',
  TRANSACTION: 'transaction',
  ADDRESS: 'address',
  TOKEN: 'token'
};

export const API_ENDPOINTS = {
  // Core blockchain endpoints
  BLOCKS: '/api/blocks',
  BLOCKS_LATEST: '/api/blocks/latest',
  BLOCKS_RECENT: '/api/explorer/blocks/recent',
  TRANSACTIONS: '/api/transactions',
  TRANSACTIONS_RECENT: '/api/explorer/transactions/recent',
  ACCOUNTS: '/api/accounts',
  SEARCH: '/api/search',
  STATS: '/api/stats',
  STATUS: '/api/status',
  HEALTH: '/api/health',
  
  // SVCP Consensus endpoints
  CONSENSUS: '/api/consensus',
  CONSENSUS_STATUS: '/api/consensus/status',
  CONSENSUS_SOCIAL_SCORE: '/api/consensus/social-score',
  CONSENSUS_AI_VALIDATION: '/api/consensus/ai-validation',
  
  // AI Engine endpoints
  AI_STATUS: '/api/ai',
  AI_NEURAL_STATUS: '/api/ai/neural-status',
  AI_LEARNING_METRICS: '/api/ai/learning-metrics',
  AI_SOCIAL_VERIFICATION: '/api/ai/social-verification',
  
  // zkML endpoints
  ZKML_STATUS: '/api/zkp/zkml/status',
  ZKML_INFO: '/api/zkp/zkml',
  
  // ZK Proof endpoints
  ZKP_INFO: '/api/zkp',
  ZKP_STATUS: '/api/zkp/status',
  
  // Sharding endpoints
  SHARDS: '/shards',
  
  // WASM endpoints
  WASM_INFO: '/wasm',
  
  // Security endpoints
  FRAUD_DASHBOARD: '/api/fraud/dashboard',
  
  // Utility endpoints
  VALIDATORS: '/api/validators',
  FAUCET: '/api/faucet',
  METRICS: '/metrics'
};

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 25,
  MAX_PAGE_SIZE: 100
};

export const REFRESH_INTERVALS = {
  BLOCKS: 12000, // 12 seconds
  TRANSACTIONS: 5000, // 5 seconds
  STATS: 30000 // 30 seconds
};

export const CHART_COLORS = {
  PRIMARY: '#00ff88',
  SECONDARY: '#8b5cf6',
  WARNING: '#fbbf24',
  SUCCESS: '#10b981',
  ERROR: '#ef4444'
};

export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536
};

export const ANIMATION_DURATIONS = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500
};

export const LOCAL_STORAGE_KEYS = {
  SELECTED_NETWORK: 'selectedNetwork',
  THEME_PREFERENCE: 'themePreference',
  BOOKMARKS: 'bookmarks',
  SEARCH_HISTORY: 'searchHistory'
};

export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error occurred. Please try again.',
  INVALID_ADDRESS: 'Invalid address format.',
  INVALID_HASH: 'Invalid transaction or block hash.',
  NOT_FOUND: 'The requested resource was not found.',
  RATE_LIMITED: 'Too many requests. Please wait and try again.'
};

export const SUCCESS_MESSAGES = {
  COPIED_TO_CLIPBOARD: 'Copied to clipboard!',
  BOOKMARK_ADDED: 'Bookmark added successfully.',
  BOOKMARK_REMOVED: 'Bookmark removed successfully.'
};

