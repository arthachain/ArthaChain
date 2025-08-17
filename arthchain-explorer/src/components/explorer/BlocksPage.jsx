import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Blocks, Clock, Hash, User, Zap, ChevronLeft, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import LoadingSpinner from '../ui/LoadingSpinner';
import CopyButton from '../ui/CopyButton';
import { apiService } from '../../services/apiService';
import { formatNumber, formatTimestamp, truncateHash, formatBytes } from '../../utils';
import useNetworkStore from '../../stores/networkStore';

const BlocksPage = () => {
  const { selectedNetwork } = useNetworkStore();
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedBlock, setSelectedBlock] = useState(null);
  const blocksPerPage = 20;

  // Handle global search queries
  useEffect(() => {
    if (window.currentSearchQuery && window.currentSearchType === 'block') {
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
        const blockData = await apiService.getBlock(query.trim());
        setSelectedBlock(blockData);
      } catch (error) {
        console.error('Block not found:', error);
      }
    }
  };

  // Fetch latest blocks from API
  const [blocks, setBlocks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchBlocks = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const latestBlocks = await apiService.getLatestBlocks(10);
        setBlocks(latestBlocks);
      } catch (err) {
        console.error('Error fetching blocks:', err);
        setError('Failed to fetch blocks');
      } finally {
        setIsLoading(false);
      }
    };

    fetchBlocks();
    
    // Refresh every 10 seconds
    const interval = setInterval(fetchBlocks, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleBlockClick = async (block) => {
    console.log('Block clicked:', block);
    setSelectedBlock(block);
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    handleSearchForQuery(searchQuery);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" text="Loading blocks..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-500">Error loading blocks: {error.message}</p>
      </div>
    );
  }

  if (selectedBlock) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedBlock(null)}
              className="flex items-center space-x-2"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Back to Blocks</span>
            </Button>
            <div>
              <h1 className="text-3xl font-bold">Block #{formatNumber(selectedBlock.number)}</h1>
              <p className="text-muted-foreground">
                Block details on {selectedNetwork.name}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: selectedNetwork.color }}
            />
            <span className="font-medium">{selectedNetwork.name}</span>
          </div>
        </div>

        {/* Block Details */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Blocks className="h-5 w-5" />
                <span>Block Information</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Block Height:</span>
                  <span className="font-mono">{formatNumber(selectedBlock.number)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Hash:</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm">{truncateHash(selectedBlock.hash)}</span>
                    <CopyButton text={selectedBlock.hash} />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Timestamp:</span>
                  <span>{formatTimestamp(selectedBlock.timestamp)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Miner:</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm">{truncateHash(selectedBlock.miner)}</span>
                    <CopyButton text={selectedBlock.miner} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Zap className="h-5 w-5" />
                <span>Gas & Size</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Gas Used:</span>
                  <span className="font-mono">{formatNumber(selectedBlock.gasUsed)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Gas Limit:</span>
                  <span className="font-mono">{formatNumber(selectedBlock.gasLimit)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Gas Utilization:</span>
                  <span>{((parseInt(selectedBlock.gasUsed) / parseInt(selectedBlock.gasLimit)) * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Block Size:</span>
                  <span>{formatBytes(selectedBlock.size)}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Transactions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Hash className="h-5 w-5" />
                <span>Transactions ({selectedBlock.transactionCount})</span>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {selectedBlock.transactions?.slice(0, 10).map((tx, index) => (
                <div key={tx.hash} className="flex items-center justify-between p-3 border border-border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="text-sm text-muted-foreground">#{index}</div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{truncateHash(tx.hash)}</span>
                        <CopyButton text={tx.hash} />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {truncateHash(tx.from)} → {truncateHash(tx.to)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{(parseInt(tx.value) / 1e18).toFixed(4)} {selectedNetwork.symbol}</div>
                    <div className="text-xs text-muted-foreground">Gas: {formatNumber(tx.gasUsed)}</div>
                  </div>
                </div>
              ))}
            </div>
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
          <h1 className="text-3xl font-bold">Blocks</h1>
          <p className="text-muted-foreground">
            Browse blockchain blocks on {selectedNetwork.name}
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
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="flex space-x-2">
            <Input
              placeholder="Search by block number or hash..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1"
            />
            <Button type="submit">Search</Button>
          </form>
        </CardContent>
      </Card>

      {/* Blocks List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Blocks className="h-5 w-5" />
            <span>Latest Blocks</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {blocks?.map((block) => (
              <div 
                key={block.hash} 
                className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                onClick={() => handleBlockClick(block.number)}
              >
                <div className="flex items-center space-x-4">
                  <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                    <Blocks className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="font-semibold">Block #{formatNumber(block.number)}</div>
                    <div className="text-sm text-muted-foreground">
                      {formatTimestamp(block.timestamp)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-6">
                  <div className="text-center">
                    <div className="text-sm font-medium">{block.transactionCount}</div>
                    <div className="text-xs text-muted-foreground">Transactions</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-medium">{formatBytes(block.size)}</div>
                    <div className="text-xs text-muted-foreground">Size</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-medium">
                      {((parseInt(block.gasUsed) / parseInt(block.gasLimit)) * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">Gas Used</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{truncateHash(block.miner)}</div>
                    <div className="text-xs text-muted-foreground">Miner</div>
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

export default BlocksPage;

