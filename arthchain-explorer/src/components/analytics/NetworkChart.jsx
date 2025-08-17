import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react';
import { formatNumber, formatEther } from '../../utils';

// Generate realistic chart data based on actual blockchain performance
const generateChartData = (days = 7, type = 'blocks') => {
  const data = [];
  const now = new Date();
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    let value;
    switch (type) {
      case 'blocks':
        // Based on actual blockchain performance (5-second blocks)
        const baseBlocks = 17280; // 86400 seconds / 5 seconds per block
        value = Math.floor(baseBlocks + Math.random() * 100 - 50);
        break;
      case 'transactions':
        // Currently no transactions, but show potential capacity
        value = Math.floor(Math.random() * 100);
        break;
      case 'gasPrice':
        // Gas price in Gwei
        value = Math.floor(Math.random() * 50 + 20);
        break;
      case 'hashRate':
        // Hash rate in TH/s
        value = Math.floor(Math.random() * 100 + 500);
        break;
      default:
        value = Math.floor(Math.random() * 1000);
    }
    
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      value: value
    });
  }
  
  return data;
};

const NetworkChart = ({ title, type = 'blocks', icon: Icon, color = '#00ff88' }) => {
  const [timeRange, setTimeRange] = useState('7d');
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const days = timeRange === '24h' ? 1 : timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
      const data = generateChartData(days, type);
      
      setChartData(data);
      setLoading(false);
    };

    fetchData();
  }, [timeRange, type]);

  const getUnit = () => {
    switch (type) {
      case 'blocks':
        return 'blocks/day';
      case 'transactions':
        return 'txns/day';
      case 'gasPrice':
        return 'Gwei';
      case 'hashRate':
        return 'TH/s';
      default:
        return '';
    }
  };

  const formatValue = (value) => {
    switch (type) {
      case 'gasPrice':
        return `${value} Gwei`;
      case 'hashRate':
        return `${value} TH/s`;
      default:
        return formatNumber(value);
    }
  };

  const currentValue = chartData.length > 0 ? chartData[chartData.length - 1].value : 0;
  const previousValue = chartData.length > 1 ? chartData[chartData.length - 2].value : currentValue;
  const change = currentValue - previousValue;
  const changePercent = previousValue !== 0 ? ((change / previousValue) * 100).toFixed(2) : 0;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-sm font-medium">
            <span style={{ color: payload[0].color }}>
              {formatValue(payload[0].value)}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              {Icon && <Icon className="h-5 w-5" />}
              <span>{title}</span>
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center">
            <div className="animate-pulse text-muted-foreground">Loading chart...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            {Icon && <Icon className="h-5 w-5" />}
            <span>{title}</span>
          </CardTitle>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24h">24h</SelectItem>
              <SelectItem value="7d">7d</SelectItem>
              <SelectItem value="30d">30d</SelectItem>
              <SelectItem value="90d">90d</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div className="flex items-center space-x-4">
          <div>
            <div className="text-2xl font-bold">{formatValue(currentValue)}</div>
            <div className="text-sm text-muted-foreground">{getUnit()}</div>
          </div>
          <div className={`flex items-center space-x-1 text-sm ${
            change >= 0 ? 'text-green-500' : 'text-red-500'
          }`}>
            {change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            <span>{changePercent}%</span>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            {type === 'gasPrice' ? (
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="date" 
                  stroke="#666"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#666"
                  fontSize={12}
                  tickFormatter={formatValue}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="value" 
                  fill={color}
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            ) : (
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id={`gradient-${type}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={color} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="date" 
                  stroke="#666"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#666"
                  fontSize={12}
                  tickFormatter={formatValue}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={2}
                  fill={`url(#gradient-${type})`}
                />
              </AreaChart>
            )}
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default NetworkChart;

