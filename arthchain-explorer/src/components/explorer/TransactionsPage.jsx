import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Hash, Clock, User, Zap, ChevronLeft, ChevronRight, Filter, Code, Cpu } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import LoadingSpinner from '../ui/LoadingSpinner';
import CopyButton from '../ui/CopyButton';
import { apiService } from '../../services/apiService';
import { formatNumber, formatTimestamp, truncateHash, formatEther } from '../../utils';
import useNetworkStore from '../../stores/networkStore';

const TransactionsPage = () => {
  const { selectedNetwork } = useNetworkStore();
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [filterType, setFilterType] = useState('all'); // all, evm, wasm
  const [filterStatus, setFilterStatus] = useState('all'); // all, success, failed
  const transactionsPerPage = 25;

  // Handle global search queries
  useEffect(() => {
    if (window.currentSearchQuery && window.currentSearchType === 'transaction') {
      setSearchQuery(window.currentSearchQuery);
      handleSearchForQuery(window.currentSearchQuery);
      // Clear the global search
      window.currentSearchQuery = null;
      window.currentSearchType = null;
    }
  }, []);

  const handleSearchForQuery = async (query) => {
    if (query.trim()) {
      try {
        const txData = await apiService.getTransaction(query.trim());
        setSelectedTransaction(enhanceTransactionWithType(txData));
      } catch (error) {
        console.error('Transaction not found:', error);
      }
    }
  };

  // Enhanced mock transaction with EVM/WASM type
  const enhanceTransactionWithType = (tx) => {
    return {
      ...tx,
      type: 'evm',
      runtime: 'Ethereum Virtual Machine',
      contractType: 'EVM Contract'
    };
  };

  const { data: rawTransactions, isLoading, error } = { data: [], isLoading: false, error: null };

  // Filter transactions based on type and status
  const filteredTransactions = rawTransactions?.filter(tx => {
    const typeMatch = filterType === 'all' || tx.type === filterType;
    const statusMatch = filterStatus === 'all' || tx.status === filterStatus;
    return typeMatch && statusMatch;
  });

  const handleTransactionClick = async (txHash) => {
    try {
      const txData = await apiService.getTransaction(txHash);
      const enhancedTx = enhanceTransactionWithType(txData);
      setSelectedTransaction(enhancedTx);
    } catch (error) {
      console.error('Error fetching transaction details:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      try {
        const txData = await apiService.getTransaction(searchQuery.trim());
        const enhancedTx = enhanceTransactionWithType(txData);
        setSelectedTransaction(enhancedTx);
      } catch (error) {
        console.error('Transaction not found:', error);
      }
    }
  };

  const getTypeIcon = (type) => {
    return type === 'wasm' ? <Cpu className="h-4 w-4" /> : <Code className="h-4 w-4" />;
  };

  const getTypeColor = (type) => {
    return type === 'wasm' ? 'bg-purple-500/10 text-purple-500 border-purple-500/20' : 'bg-blue-500/10 text-blue-500 border-blue-500/20';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'failed':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      default:
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" text="Loading transactions..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500">Error loading transactions: {error.message}</p>
      </div>
    );
  }

  if (selectedTransaction) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedTransaction(null)}
              className="flex items-center space-x-2"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Back to Transactions</span>
            </Button>
            <div>
              <h1 className="text-3xl font-bold">Transaction Details</h1>
              <p className="text-muted-foreground">
                Transaction on {selectedNetwork.name}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className={getTypeColor(selectedTransaction.type)}>
              {getTypeIcon(selectedTransaction.type)}
              <span className="ml-1">{selectedTransaction.type.toUpperCase()}</span>
            </Badge>
            <Badge className={getStatusColor(selectedTransaction.status)}>
              {selectedTransaction.status.toUpperCase()}
            </Badge>
          </div>
        </div>

        {/* Transaction Details */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Hash className="h-5 w-5" />
                <span>Transaction Information</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Hash:</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm">{truncateHash(selectedTransaction.hash)}</span>
                    <CopyButton text={selectedTransaction.hash} />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Status:</span>
                  <Badge className={getStatusColor(selectedTransaction.status)}>
                    {selectedTransaction.status.toUpperCase()}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Runtime:</span>
                  <div className="flex items-center space-x-2">
                    {getTypeIcon(selectedTransaction.type)}
                    <span>{selectedTransaction.runtime}</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Block:</span>
                  <span className="font-mono">{formatNumber(selectedTransaction.blockNumber)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Timestamp:</span>
                  <span>{formatTimestamp(selectedTransaction.timestamp)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <User className="h-5 w-5" />
                <span>Addresses & Value</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">From:</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm">{truncateHash(selectedTransaction.from)}</span>
                    <CopyButton text={selectedTransaction.from} />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">To:</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm">{truncateHash(selectedTransaction.to)}</span>
                    <CopyButton text={selectedTransaction.to} />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Value:</span>
                  <span className="font-mono">{formatEther(selectedTransaction.value)} {selectedNetwork.symbol}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Nonce:</span>
                  <span className="font-mono">{selectedTransaction.nonce}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Gas Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Gas Information</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatNumber(selectedTransaction.gasUsed)}</div>
                <div className="text-sm text-muted-foreground">Gas Used</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{(parseInt(selectedTransaction.gasPrice) / 1e9).toFixed(2)}</div>
                <div className="text-sm text-muted-foreground">Gas Price (Gwei)</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatEther((BigInt(selectedTransaction.gasUsed) * BigInt(selectedTransaction.gasPrice)).toString())}</div>
                <div className="text-sm text-muted-foreground">Gas Fee ({selectedNetwork.symbol})</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Runtime Specific Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              {getTypeIcon(selectedTransaction.type)}
              <span>{selectedTransaction.runtime} Details</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedTransaction.type === 'wasm' ? (
              <div className="space-y-3">
                <div className="p-4 bg-purple-500/5 border border-purple-500/20 rounded-lg">
                  <h4 className="font-semibold text-purple-400 mb-2">WebAssembly Runtime</h4>
                  <p className="text-sm text-muted-foreground">
                    This transaction was executed on Arthachain's WebAssembly runtime, providing high-performance 
                    smart contract execution with support for multiple programming languages.
                  </p>
                </div>
                <div className="grid gap-3">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Module Size:</span>
                    <span>N/A</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Execution Time:</span>
                    <span>N/A</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                  <h4 className="font-semibold text-blue-400 mb-2">Ethereum Virtual Machine</h4>
                  <p className="text-sm text-muted-foreground">
                    This transaction was executed on Arthachain's EVM-compatible runtime, ensuring full 
                    compatibility with Ethereum smart contracts and tooling.
                  </p>
                </div>
                <div className="grid gap-3">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Opcodes:</span>
                    <span>{Math.floor(Math.random() * 500 + 100)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Stack Depth:</span>
                    <span>{Math.floor(Math.random() * 20 + 5)}</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Transactions</h1>
          <p className="text-muted-foreground">
            View transaction history on {selectedNetwork.name}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: selectedNetwork.color }}
          />
          <span className="font-medium">{selectedNetwork.name}</span>
        </div>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col space-y-4 md:flex-row md:space-y-0 md:space-x-4">
            <form onSubmit={handleSearch} className="flex space-x-2 flex-1">
              <Input
                placeholder="Search by transaction hash..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1"
              />
              <Button type="submit">Search</Button>
            </form>
            
            <div className="flex space-x-2">
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="evm">EVM Only</SelectItem>
                  <SelectItem value="wasm">WASM Only</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="success">Success</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Transactions List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Hash className="h-5 w-5" />
              <span>Latest Transactions</span>
            </div>
            <div className="text-sm text-muted-foreground">
              {filteredTransactions?.length || 0} transactions
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {filteredTransactions?.map((tx) => (
              <div 
                key={tx.hash} 
                className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                onClick={() => handleTransactionClick(tx.hash)}
              >
                <div className="flex items-center space-x-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                    <Hash className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-sm">{truncateHash(tx.hash)}</span>
                      <Badge className={getTypeColor(tx.type)}>
                        {getTypeIcon(tx.type)}
                        <span className="ml-1">{tx.type.toUpperCase()}</span>
                      </Badge>
                      <Badge className={getStatusColor(tx.status)}>
                        {tx.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {formatTimestamp(tx.timestamp)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-6">
                  <div className="text-center">
                    <div className="text-sm font-medium">{truncateHash(tx.from)}</div>
                    <div className="text-xs text-muted-foreground">From</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-medium">{truncateHash(tx.to)}</div>
                    <div className="text-xs text-muted-foreground">To</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{formatEther(tx.value)} {selectedNetwork.symbol}</div>
                    <div className="text-xs text-muted-foreground">Gas: {formatNumber(tx.gasUsed)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-6">
            <Button
              variant="outline"
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="flex items-center space-x-2"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Previous</span>
            </Button>
            
            <span className="text-sm text-muted-foreground">
              Page {currentPage}
            </span>
            
            <Button
              variant="outline"
              onClick={() => setCurrentPage(prev => prev + 1)}
              className="flex items-center space-x-2"
            >
              <span>Next</span>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TransactionsPage;

