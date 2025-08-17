import React, { useState, useEffect } from 'react';
import { Activity, Blocks, Zap, TrendingUp, PieChart, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PieChart as RechartsPieChart, Cell, ResponsiveContainer, Tooltip, Pie } from 'recharts';
import NetworkChart from './NetworkChart';
import useNetworkStore from '../../stores/networkStore';
import { formatNumber, formatEther } from '../../utils';

// Mock data for pie charts
const generatePieData = (type) => {
  const colors = ['#00ff88', '#8b5cf6', '#fbbf24', '#10b981', '#ef4444'];
  
  switch (type) {
    case 'gasUsage':
      return [
        { name: 'DeFi', value: 35, color: colors[0] },
        { name: 'NFTs', value: 25, color: colors[1] },
        { name: 'DEX', value: 20, color: colors[2] },
        { name: 'Gaming', value: 12, color: colors[3] },
        { name: 'Others', value: 8, color: colors[4] }
      ];
    case 'topAddresses':
      return [
        { name: 'Exchange Wallets', value: 40, color: colors[0] },
        { name: 'DeFi Protocols', value: 30, color: colors[1] },
        { name: 'Individual Users', value: 20, color: colors[2] },
        { name: 'Smart Contracts', value: 10, color: colors[3] }
      ];
    default:
      return [];
  }
};

const PieChartCard = ({ title, data, icon: Icon }) => {
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium">{data.name}</p>
          <p className="text-sm text-muted-foreground">{data.value}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          {Icon && <Icon className="h-5 w-5" />}
          <span>{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </RechartsPieChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 space-y-2">
          {data.map((item, index) => (
            <div key={index} className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span>{item.name}</span>
              </div>
              <span className="font-medium">{item.value}%</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

const Analytics = () => {
  const { selectedNetwork } = useNetworkStore();
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Network Analytics</h1>
          <p className="text-muted-foreground">
            Detailed analytics and insights for {selectedNetwork.name}
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

      {/* Analytics Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Time Series Charts */}
          <div className="grid gap-6 lg:grid-cols-2">
            <NetworkChart
              title="Daily Blocks"
              type="blocks"
              icon={Blocks}
              color="#00ff88"
            />
            <NetworkChart
              title="Daily Transactions"
              type="transactions"
              icon={Activity}
              color="#8b5cf6"
            />
          </div>

          {/* Pie Charts */}
          <div className="grid gap-6 lg:grid-cols-2">
            <PieChartCard
              title="Gas Usage by Category"
              data={generatePieData('gasUsage')}
              icon={PieChart}
            />
            <PieChartCard
              title="Top Address Types"
              data={generatePieData('topAddresses')}
              icon={BarChart3}
            />
          </div>
        </TabsContent>

        <TabsContent value="transactions" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <NetworkChart
              title="Average Gas Price"
              type="gasPrice"
              icon={Zap}
              color="#fbbf24"
            />
            <NetworkChart
              title="Transaction Volume"
              type="transactions"
              icon={Activity}
              color="#10b981"
            />
          </div>

          {/* Transaction Statistics */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg Transaction Fee</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">0.0023 ETH</div>
                <div className="flex items-center space-x-1 text-xs text-green-500">
                  <TrendingUp className="h-3 w-3" />
                  <span>-5.2%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">98.7%</div>
                <div className="flex items-center space-x-1 text-xs text-green-500">
                  <TrendingUp className="h-3 w-3" />
                  <span>+0.3%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg Confirmation Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">13.2s</div>
                <div className="flex items-center space-x-1 text-xs text-green-500">
                  <TrendingUp className="h-3 w-3" />
                  <span>-1.8%</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="network" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <NetworkChart
              title="Network Hash Rate"
              type="hashRate"
              icon={Zap}
              color="#ef4444"
            />
            <NetworkChart
              title="Block Production"
              type="blocks"
              icon={Blocks}
              color="#00ff88"
            />
          </div>

          {/* Network Health Metrics */}
          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Network Uptime</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">99.98%</div>
                <div className="text-xs text-muted-foreground">Last 30 days</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Active Nodes</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">8,247</div>
                <div className="flex items-center space-x-1 text-xs text-green-500">
                  <TrendingUp className="h-3 w-3" />
                  <span>+2.1%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Network Difficulty</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">15.76T</div>
                <div className="text-xs text-muted-foreground">Current difficulty</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Block Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">12.1s</div>
                <div className="text-xs text-muted-foreground">Average</div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Analytics;

