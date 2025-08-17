import React from 'react';
import { Clock, Hash, User, Zap, HardDrive } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatDate, formatRelativeTime, truncateHash, formatNumber, formatBytes } from '../../utils';

const BlockCard = ({ block, onClick }) => {
  const handleClick = () => {
    onClick?.(block);
  };

  return (
    <Card className="hover:bg-card/80 transition-colors cursor-pointer animate-fade-in" onClick={handleClick}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">
            Block #{formatNumber(block.height)}
          </CardTitle>
          <Badge variant="secondary" className="text-xs">
            {block.tx_count} txns
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Block Hash */}
        <div className="flex items-center space-x-2">
          <Hash className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">Hash</div>
            <div className="font-mono text-sm truncate">{truncateHash(block.hash)}</div>
          </div>
        </div>

        {/* Timestamp */}
        <div className="flex items-center space-x-2">
          <Clock className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">Timestamp</div>
            <div className="text-sm">{formatRelativeTime(block.timestamp)}</div>
            <div className="text-xs text-muted-foreground">{formatDate(block.timestamp)}</div>
          </div>
        </div>

        {/* Miner */}
        <div className="flex items-center space-x-2">
          <User className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">Miner</div>
            <div className="font-mono text-sm truncate">{truncateHash(block.proposer || '0x0000000000000000000000000000000000000000')}</div>
          </div>
        </div>

        {/* Gas Usage */}
        <div className="flex items-center space-x-2">
          <Zap className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">Gas Used</div>
            <div className="text-sm">
              {formatNumber(parseInt(block.gasUsed || 0))} / {formatNumber(parseInt(block.gasLimit || 30000000))}
            </div>
            <div className="w-full bg-muted rounded-full h-2 mt-1">
              <div 
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ 
                  width: `${Math.min((parseInt(block.gasUsed) / parseInt(block.gasLimit)) * 100, 100)}%` 
                }}
              />
            </div>
          </div>
        </div>

        {/* Block Size */}
        <div className="flex items-center space-x-2">
          <HardDrive className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm text-muted-foreground">Size</div>
            <div className="text-sm">{formatBytes(block.size)}</div>
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

export default BlockCard;

