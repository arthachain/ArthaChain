import React from 'react';
import { ArrowRight, Clock, Hash, Zap, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatDate, formatRelativeTime, truncateHash, formatEther, formatGwei, calculateGasFee } from '../../utils';
import { TRANSACTION_STATUS } from '../../constants';

const TransactionCard = ({ transaction, onClick }) => {
  const handleClick = () => {
    onClick?.(transaction);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case TRANSACTION_STATUS.SUCCESS:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case TRANSACTION_STATUS.FAILED:
        return <XCircle className="h-4 w-4 text-red-500" />;
      case TRANSACTION_STATUS.PENDING:
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case TRANSACTION_STATUS.SUCCESS:
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case TRANSACTION_STATUS.FAILED:
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case TRANSACTION_STATUS.PENDING:
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const gasFee = calculateGasFee(transaction.gasUsed, transaction.gasPrice);

  return (
    <Card className="hover:bg-card/80 transition-colors cursor-pointer animate-fade-in" onClick={handleClick}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Hash className="h-4 w-4 text-muted-foreground" />
            <span className="font-mono text-sm font-medium">
              {truncateHash(transaction.hash)}
            </span>
          </div>
          <Badge className={getStatusColor(transaction.status)}>
            <div className="flex items-center space-x-1">
              {getStatusIcon(transaction.status)}
              <span className="capitalize">{transaction.status}</span>
            </div>
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* From/To Addresses */}
        <div className="flex items-center space-x-2">
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">From</div>
            <div className="font-mono text-sm truncate">{truncateHash(transaction.from)}</div>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">To</div>
            <div className="font-mono text-sm truncate">{truncateHash(transaction.to)}</div>
          </div>
        </div>

        {/* Value */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-muted-foreground">Value</div>
            <div className="text-sm font-medium">
              {formatEther(transaction.value)} ETH
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-muted-foreground">Gas Fee</div>
            <div className="text-sm font-medium">
              {parseFloat(gasFee).toFixed(6)} ETH
            </div>
          </div>
        </div>

        {/* Gas Details */}
        <div className="flex items-center justify-between text-sm">
          <div>
            <span className="text-muted-foreground">Gas Used: </span>
            <span>{parseInt(transaction.gasUsed).toLocaleString()}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Gas Price: </span>
            <span>{formatGwei(transaction.gasPrice)} Gwei</span>
          </div>
        </div>

        {/* Timestamp */}
        <div className="flex items-center space-x-2">
          <Clock className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm">{formatRelativeTime(transaction.timestamp)}</div>
            <div className="text-xs text-muted-foreground">{formatDate(transaction.timestamp)}</div>
          </div>
        </div>

        {/* Block Info */}
        <div className="flex items-center justify-between text-sm">
          <div>
            <span className="text-muted-foreground">Block: </span>
            <span className="font-medium">{transaction.blockNumber.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Position: </span>
            <span>{transaction.transactionIndex}</span>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-2">
          <Button 
            variant="outline" 
            size="sm" 
            className="w-full"
            onClick={(e) => {
              e.stopPropagation();
              handleClick();
            }}
          >
            View Details
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default TransactionCard;

