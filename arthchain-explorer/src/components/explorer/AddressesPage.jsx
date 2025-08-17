import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { User, Wallet, Hash, Clock, ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import LoadingSpinner from '../ui/LoadingSpinner';
import CopyButton from '../ui/CopyButton';
import { apiService } from '../../services/apiService';
import { formatNumber, formatTimestamp, truncateHash, formatEther } from '../../utils';
import useNetworkStore from '../../stores/networkStore';

const AddressesPage = () => {
  const { selectedNetwork } = useNetworkStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedAddress, setSelectedAddress] = useState(null);
  const [addressData, setAddressData] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      setLoading(true);
      try {
        const address = searchQuery.trim();
        const [addressInfo, addressTxs] = [
          { balance: "0", transactionCount: 0, type: "externally_owned_account" },
          { transactions: [], totalCount: 0, page: 1, limit: 25, hasMore: false }
        ];
        
        setSelectedAddress(address);
        setAddressData(addressInfo);
        setTransactions(addressTxs.transactions);
        setCurrentPage(1);
      } catch (error) {
        console.error('Error fetching address data:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  const loadMoreTransactions = async (page) => {
    if (!selectedAddress) return;
    
    setLoading(true);
    try {
      const addressTxs = await apiService.getAddressTransactions(selectedAddress, page, 25);
      setTransactions(addressTxs.transactions);
      setCurrentPage(page);
    } catch (error) {
      console.error('Error loading transactions:', error);
    } finally {
      setLoading(false);
    }
  };

  const getAddressTypeColor = (type) => {
    return type === 'contract' 
      ? 'bg-purple-500/10 text-purple-500 border-purple-500/20'
      : 'bg-green-500/10 text-green-500 border-green-500/20';
  };

  const getAddressTypeLabel = (type) => {
    return type === 'contract' ? 'Smart Contract' : 'Wallet Address';
  };

  if (loading && !selectedAddress) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" text="Loading address data..." />
      </div>
    );
  }

  if (selectedAddress && addressData) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setSelectedAddress(null);
                setAddressData(null);
                setTransactions([]);
                setSearchQuery('');
              }}
              className="flex items-center space-x-2"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Back to Search</span>
            </Button>
            <div>
              <h1 className="text-3xl font-bold">Address Details</h1>
              <p className="text-muted-foreground">
                Address information on {selectedNetwork.name}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className={getAddressTypeColor(addressData.type)}>
              {getAddressTypeLabel(addressData.type)}
            </Badge>
          </div>
        </div>

        {/* Address Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="h-5 w-5" />
              <span>Address Overview</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatEther(addressData.balance)}</div>
                <div className="text-sm text-muted-foreground">Balance ({selectedNetwork.symbol})</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{formatNumber(addressData.transactionCount)}</div>
                <div className="text-sm text-muted-foreground">Total Transactions</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">${(parseFloat(formatEther(addressData.balance)) * 2500).toFixed(2)}</div>
                <div className="text-sm text-muted-foreground">USD Value</div>
              </div>
              <div className="text-center p-4 border border-border rounded-lg">
                <div className="text-2xl font-bold">{addressData.tokens?.length || 0}</div>
                <div className="text-sm text-muted-foreground">Token Holdings</div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Address:</span>
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-sm break-all">{selectedAddress}</span>
                  <CopyButton text={selectedAddress} />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tabs for different views */}
        <Tabs defaultValue="transactions" className="space-y-4">
          <TabsList>
            <TabsTrigger value="transactions">Transactions</TabsTrigger>
            <TabsTrigger value="tokens">Tokens</TabsTrigger>
            <TabsTrigger value="contracts">Contracts</TabsTrigger>
          </TabsList>

          <TabsContent value="transactions">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Hash className="h-5 w-5" />
                    <span>Transactions</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {transactions.length} transactions
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {transactions.map((tx, index) => (
                    <div key={tx.hash} className="flex items-center justify-between p-4 border border-border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                          <Hash className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-mono text-sm">{truncateHash(tx.hash)}</span>
                            <Badge className={tx.status === 'success' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}>
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
                          <div className="text-sm font-medium">
                            {tx.from.toLowerCase() === selectedAddress.toLowerCase() ? 'OUT' : 'IN'}
                          </div>
                          <div className="text-xs text-muted-foreground">Direction</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm font-medium">
                            {tx.from.toLowerCase() === selectedAddress.toLowerCase() 
                              ? truncateHash(tx.to) 
                              : truncateHash(tx.from)
                            }
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {tx.from.toLowerCase() === selectedAddress.toLowerCase() ? 'To' : 'From'}
                          </div>
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
                    onClick={() => loadMoreTransactions(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1 || loading}
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
                    onClick={() => loadMoreTransactions(currentPage + 1)}
                    disabled={loading}
                    className="flex items-center space-x-2"
                  >
                    <span>Next</span>
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="tokens">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Wallet className="h-5 w-5" />
                  <span>Token Holdings</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <Wallet className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No token holdings found for this address.</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="contracts">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Hash className="h-5 w-5" />
                  <span>Contract Interactions</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {addressData.type === 'contract' ? (
                  <div className="space-y-4">
                    <div className="p-4 bg-purple-500/5 border border-purple-500/20 rounded-lg">
                      <h4 className="font-semibold text-purple-400 mb-2">Smart Contract</h4>
                      <p className="text-sm text-muted-foreground">
                        This address is a smart contract deployed on Arthachain. It can execute code and store data.
                      </p>
                    </div>
                    <div className="grid gap-3">
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Contract Type:</span>
                        <span>EVM Compatible</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Deployment Block:</span>
                        <span>N/A</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Total Interactions:</span>
                        <span>{formatNumber(addressData.transactionCount)}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Hash className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">This is an externally owned account (EOA), not a smart contract.</p>
                  </div>
                )}
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
          <h1 className="text-3xl font-bold">Addresses</h1>
          <p className="text-muted-foreground">
            Explore wallet addresses on {selectedNetwork.name}
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

      {/* Search */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Search className="h-5 w-5" />
            <span>Address Search</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="flex space-x-2">
            <Input
              placeholder="Enter wallet address (0x...)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" disabled={loading}>
              {loading ? <LoadingSpinner size="sm" /> : 'Search'}
            </Button>
          </form>
          <p className="text-sm text-muted-foreground mt-2">
            Enter a valid Arthachain address to view its details, balance, and transaction history.
          </p>
        </CardContent>
      </Card>

      {/* Popular Addresses */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <User className="h-5 w-5" />
            <span>Popular Addresses</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              { address: '0xArthachainTreasuryFund1234567890abcdef', label: 'Arthachain Treasury', type: 'contract' },
              { address: '0xArthachainStakingPool7890abcdef1234567890', label: 'Staking Pool', type: 'contract' },
              { address: '0xArthachainBridgeWallet9876543210fedcba9876', label: 'Bridge Contract', type: 'contract' },
              { address: '0xArthachainValidatorNode1234567890abcdef', label: 'Validator Node', type: 'externally_owned_account' }
            ].map((item, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                onClick={() => {
                  setSearchQuery(item.address);
                  handleSearch({ preventDefault: () => {} });
                }}
              >
                <div className="flex items-center space-x-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                    <User className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="font-semibold">{item.label}</div>
                    <div className="font-mono text-sm text-muted-foreground">{truncateHash(item.address)}</div>
                  </div>
                </div>
                <Badge className={getAddressTypeColor(item.type)}>
                  {getAddressTypeLabel(item.type)}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AddressesPage;

