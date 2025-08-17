import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Code, Cpu, Hash, Clock, User, Zap, ChevronLeft, ChevronRight, Search, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import LoadingSpinner from '../ui/LoadingSpinner';
import CopyButton from '../ui/CopyButton';
import { apiService } from '../../services/apiService';
import { formatNumber, formatTimestamp, truncateHash, formatEther } from '../../utils';
import useNetworkStore from '../../stores/networkStore';

const ContractsPage = () => {
  const { selectedNetwork } = useNetworkStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedContract, setSelectedContract] = useState(null);
  const [contractFilter, setContractFilter] = useState('all'); // all, evm, wasm
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(false);

  // Contract data will be fetched from API
  const [contracts, setContracts] = useState([]);

  const filteredContracts = contracts.filter(contract => {
    return contractFilter === 'all' || contract.type === contractFilter;
  });

  const handleSearch = async (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      setLoading(true);
      try {
        // Find contract by address from API
        const contract = contracts.find(c => 
          c.address.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.name.toLowerCase().includes(searchQuery.toLowerCase())
        );
        
        if (contract) {
          setSelectedContract(contract);
        } else {
          setSelectedContract(null);
        }
      } catch (error) {
        console.error('Error searching contract:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  const getTypeIcon = (type) => {
    return type === 'wasm' ? <Cpu className="h-4 w-4" /> : <Code className="h-4 w-4" />;
  };

  const getTypeColor = (type) => {
    return type === 'wasm' 
      ? 'bg-purple-500/10 text-purple-500 border-purple-500/20' 
      : 'bg-blue-500/10 text-blue-500 border-blue-500/20';
  };

  const getVerificationColor = (verified) => {
    return verified 
      ? 'bg-green-500/10 text-green-500 border-green-500/20'
      : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
  };

  if (selectedContract) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedContract(null)}
              className="flex items-center space-x-2"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Back to Contracts</span>
            </Button>
            <div>
              <h1 className="text-3xl font-bold">{selectedContract.name}</h1>
              <p className="text-muted-foreground">
                Smart contract on {selectedNetwork.name}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className={getTypeColor(selectedContract.type)}>
              {getTypeIcon(selectedContract.type)}
              <span className="ml-1">{selectedContract.type.toUpperCase()}</span>
            </Badge>
            <Badge className={getVerificationColor(selectedContract.verified)}>
              {selectedContract.verified ? 'Verified' : 'Unverified'}
            </Badge>
          </div>
        </div>

        {/* Contract Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Hash className="h-5 w-5" />
              <span>Contract Overview</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatEther(selectedContract.balance)}</div>
                <div className="text-sm text-muted-foreground">Balance ({selectedNetwork.symbol})</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatNumber(selectedContract.transactionCount)}</div>
                <div className="text-sm text-muted-foreground">Total Transactions</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatNumber(selectedContract.deploymentBlock)}</div>
                <div className="text-sm text-muted-foreground">Deployment Block</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{Math.floor((Date.now() - selectedContract.deploymentTime) / (1000 * 60 * 60 * 24))}</div>
                <div className="text-sm text-muted-foreground">Days Active</div>
              </div>
            </div>
            
            <div className="mt-6 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Contract Address:</span>
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-sm">{truncateHash(selectedContract.address)}</span>
                  <CopyButton text={selectedContract.address} />
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Runtime:</span>
                <div className="flex items-center space-x-2">
                  {getTypeIcon(selectedContract.type)}
                  <span>{selectedContract.runtime}</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Deployed:</span>
                <span>{formatTimestamp(Math.floor(selectedContract.deploymentTime / 1000))}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Runtime Details */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              {getTypeIcon(selectedContract.type)}
              <span>{selectedContract.runtime} Details</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedContract.type === 'wasm' ? (
              <div className="space-y-4">
                <div className="p-4 bg-purple-500/5 border border-purple-500/20 rounded-lg">
                  <h4 className="font-semibold text-purple-400 mb-2">WebAssembly Runtime</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    {selectedContract.description}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    This contract runs on Arthachain's WebAssembly runtime, providing high-performance 
                    execution with support for multiple programming languages including Rust, C++, and AssemblyScript.
                  </p>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Module Size:</span>
                    <span>{Math.floor(Math.random() * 200 + 50)} KB</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Memory Usage:</span>
                    <span>{Math.floor(Math.random() * 16 + 4)} MB</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Avg Execution Time:</span>
                    <span>{Math.floor(Math.random() * 50 + 10)} ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Gas Efficiency:</span>
                    <span>{Math.floor(Math.random() * 30 + 70)}% better</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                  <h4 className="font-semibold text-blue-400 mb-2">Ethereum Virtual Machine</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    {selectedContract.description}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    This contract runs on Arthachain's EVM-compatible runtime, ensuring full compatibility 
                    with Ethereum smart contracts, tools, and development frameworks.
                  </p>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Bytecode Size:</span>
                    <span>{Math.floor(Math.random() * 50 + 20)} KB</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Solidity Version:</span>
                    <span>0.8.{Math.floor(Math.random() * 20 + 10)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Optimization:</span>
                    <span>{Math.random() > 0.5 ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Constructor Args:</span>
                    <span>{Math.floor(Math.random() * 5)}</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Contract Tabs */}
        <Tabs defaultValue="transactions" className="space-y-4">
          <TabsList>
            <TabsTrigger value="transactions">Transactions</TabsTrigger>
            <TabsTrigger value="events">Events</TabsTrigger>
            <TabsTrigger value="code">Code</TabsTrigger>
          </TabsList>

          <TabsContent value="transactions">
            <Card>
              <CardHeader>
                <CardTitle>Recent Transactions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <Hash className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">Transaction history would be displayed here.</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="events">
            <Card>
              <CardHeader>
                <CardTitle>Contract Events</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <Zap className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">Contract events would be displayed here.</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="code">
            <Card>
              <CardHeader>
                <CardTitle>Contract Code</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">
                    {selectedContract.verified 
                      ? 'Verified contract source code would be displayed here.'
                      : 'This contract is not verified. Only bytecode is available.'
                    }
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Smart Contracts</h1>
          <p className="text-muted-foreground">
            Explore smart contracts on {selectedNetwork.name}
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
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Search className="h-5 w-5" />
            <span>Contract Search</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col space-y-4 md:flex-row md:space-y-0 md:space-x-4">
            <form onSubmit={handleSearch} className="flex space-x-2 flex-1">
              <Input
                placeholder="Search by contract address or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1"
              />
              <Button type="submit" disabled={loading}>
                {loading ? <LoadingSpinner size="sm" /> : 'Search'}
              </Button>
            </form>
            
            <Select value={contractFilter} onValueChange={setContractFilter}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Contracts</SelectItem>
                <SelectItem value="evm">EVM Only</SelectItem>
                <SelectItem value="wasm">WASM Only</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Contracts List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Code className="h-5 w-5" />
              <span>Popular Contracts</span>
            </div>
            <div className="text-sm text-muted-foreground">
              {filteredContracts.length} contracts
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {filteredContracts.map((contract, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                onClick={() => setSelectedContract(contract)}
              >
                <div className="flex items-center space-x-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                    {getTypeIcon(contract.type)}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold">{contract.name}</span>
                      <Badge className={getTypeColor(contract.type)}>
                        {contract.type.toUpperCase()}
                      </Badge>
                      <Badge className={getVerificationColor(contract.verified)}>
                        {contract.verified ? 'Verified' : 'Unverified'}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {truncateHash(contract.address)} • {formatTimestamp(Math.floor(contract.deploymentTime / 1000))}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-6">
                  <div className="text-center">
                    <div className="text-sm font-medium">{formatNumber(contract.transactionCount)}</div>
                    <div className="text-xs text-muted-foreground">Transactions</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-medium">{formatEther(contract.balance)}</div>
                    <div className="text-xs text-muted-foreground">Balance ({selectedNetwork.symbol})</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">{contract.runtime}</div>
                    <div className="text-xs text-muted-foreground">Runtime</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ContractsPage;

