import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { copyToClipboard } from '../../utils';
import { cn } from '@/lib/utils';

const CopyButton = ({ 
  text, 
  size = 'sm', 
  variant = 'ghost',
  className,
  showText = false,
  successMessage = 'Copied!'
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e) => {
    e.stopPropagation();
    
    const success = await copyToClipboard(text);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant={variant}
            size={size}
            onClick={handleCopy}
            className={cn(
              "transition-colors",
              copied && "text-green-500",
              className
            )}
          >
            {copied ? (
              <Check className="h-4 w-4" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
            {showText && (
              <span className="ml-2">
                {copied ? successMessage : 'Copy'}
              </span>
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{copied ? successMessage : 'Copy to clipboard'}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default CopyButton;

