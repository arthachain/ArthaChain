import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, Blocks, Users, Zap, TrendingUp, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import BlockCard from './BlockCard';
import TransactionCard from './TransactionCard';
import { apiService } from '../../services/apiService';
import { formatNumber, formatRelativeTime } from '../../utils';
import useNetworkStore from '../../stores/networkStore';

const StatCard = ({ title, value, icon: Icon, change, description }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium">{title}</CardTitle>
      <Icon className="h-4 w-4 text-muted-foreground" />
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">{value}</div>
      {change && (
        <div className="flex items-center space-x-1 text-xs text-muted-foreground">
          <TrendingUp className="h-3 w-3" />
          <span>{change}</span>
        </div>
      )}
      {description && (
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      )}
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const { selectedNetwork } = useNetworkStore();
  const [stats, setStats] = useState(null);
  const [latestBlocks, setLatestBlocks] = useState([]);
  const [latestTransactions, setLatestTransactions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [statsData, blocksData, transactionsData] = await Promise.all([
          apiService.getNetworkStats(),
          apiService.getLatestBlocks(5),
          apiService.getLatestTransactions(5)
        ]);

        setStats(statsData);
        setLatestBlocks(blocksData);
        setLatestTransactions(transactionsData);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Set up auto-refresh
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [selectedNetwork]);

  const handleBlockClick = (block) => {
    console.log('Block clicked:', block);
    // Navigate to blocks page and store block data for display
    if (window.navigateTo) {
      window.navigateTo('blocks');
      // Store the selected block for the blocks page to display
      window.selectedBlockData = block;
    }
  };

  const handleTransactionClick = (transaction) => {
    console.log('Transaction clicked:', transaction);
    // Navigate to transactions page and store transaction data for display
    if (window.navigateTo) {
      window.navigateTo('transactions');
      // Store the selected transaction for the transactions page to display
      window.selectedTransactionData = transaction;
    }
  };

  const handleViewAllBlocks = () => {
    if (window.navigateTo) {
      window.navigateTo('blocks');
    }
  };

  const handleViewAllTransactions = () => {
    if (window.navigateTo) {
      window.navigateTo('transactions');
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="space-y-0 pb-2">
                <div className="h-4 bg-muted rounded w-3/4"></div>
              </CardHeader>
              <CardContent>
                <div className="h-8 bg-muted rounded w-1/2 mb-2"></div>
                <div className="h-3 bg-muted rounded w-full"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full max-w-none">
      {/* Page Header */}
      <div className="flex flex-col space-y-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Network Overview</h1>
        <p className="text-muted-foreground">Real-time statistics for Arthachain</p>
      </div>

      {/* Network Status Indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
          <span className="text-sm font-medium">Arthachain</span>
        </div>
      </div>

      {/* Main Stats Grid */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Block Height"
          value={formatNumber(stats.blockHeight)}
          icon={Blocks}
          change="+0.5%"
          description="Current block number"
        />
        <StatCard
          title="Pending Transactions"
          value={formatNumber(stats.pendingTxCount || 0)}
          icon={Activity}
          change="+2.1%"
          description="Transactions in mempool"
        />
        <StatCard
          title="Active Addresses"
          value={formatNumber(stats.activeAddresses)}
          icon={Users}
          change="+1.8%"
          description="24h active addresses"
        />
        <StatCard
          title="Avg Block Time"
          value={`${typeof stats.avgBlockTime === 'number' ? stats.avgBlockTime.toFixed(1) : stats.avgBlockTime || 0}s`}
          icon={Clock}
          change="-0.3%"
          description="Average time between blocks"
        />
      </div>

      {/* Network Details */}
      <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Network Hash Rate</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.hashRate}</div>
            <p className="text-sm text-muted-foreground">Current network hash rate</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Difficulty</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.difficulty}</div>
            <p className="text-sm text-muted-foreground">Mining difficulty</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Total Supply</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(parseInt(stats.totalSupply))} ARTHA</div>
            <p className="text-sm text-muted-foreground">Total ARTHA supply</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Test Tokens</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(parseInt(stats.testTokensAvailable || 500000000))} ARTHA</div>
            <p className="text-sm text-muted-foreground">Available for testing</p>
          </CardContent>
        </Card>
      </div>

      {/* Latest Blocks and Transactions */}
      <div className="grid gap-6 grid-cols-1 xl:grid-cols-2">
        {/* Latest Blocks */}
        <div className="w-full">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <h2 className="text-xl font-semibold">Latest Blocks</h2>
            <Button variant="outline" size="sm" onClick={handleViewAllBlocks} className="w-full sm:w-auto">
              View All Blocks
            </Button>
          </div>
          <div className="space-y-4">
            {latestBlocks.map((block) => (
              <BlockCard 
                key={block.hash} 
                block={block} 
                onClick={handleBlockClick}
              />
            ))}
          </div>
        </div>

        {/* Latest Transactions */}
        <div className="w-full">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <h2 className="text-xl font-semibold">Latest Transactions</h2>
            <Button variant="outline" size="sm" onClick={handleViewAllTransactions} className="w-full sm:w-auto">
              View All Transactions
            </Button>
          </div>
          <div className="space-y-4">
            {latestTransactions.map((transaction) => (
              <TransactionCard 
                key={transaction.hash} 
                transaction={transaction} 
                onClick={handleTransactionClick}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

