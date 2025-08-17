import React, { useState, useEffect, useRef } from 'react';
import { Search, Clock, Hash, User, Blocks, Bookmark } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import useSearchStore from '../../stores/searchStore';
import { getSearchType, truncateHash } from '../../utils';
import { cn } from '@/lib/utils';

const SearchSuggestions = ({ 
  query, 
  isVisible, 
  onSelect, 
  onClose,
  className 
}) => {
  const { searchHistory, bookmarks } = useSearchStore();
  const [suggestions, setSuggestions] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const suggestionRefs = useRef([]);

  useEffect(() => {
    if (!query.trim()) {
      // Show recent searches and bookmarks when no query
      const recentSuggestions = searchHistory.slice(0, 5).map(item => ({
        type: 'history',
        value: item,
        searchType: getSearchType(item),
        icon: Clock
      }));

      const bookmarkSuggestions = bookmarks.slice(0, 3).map(item => ({
        type: 'bookmark',
        value: item.value,
        searchType: item.type,
        label: item.label,
        icon: Bookmark
      }));

      setSuggestions([...recentSuggestions, ...bookmarkSuggestions]);
    } else {
      // Generate suggestions based on query
      const querySuggestions = [];
      const searchType = getSearchType(query);

      if (searchType !== 'unknown') {
        querySuggestions.push({
          type: 'suggestion',
          value: query,
          searchType,
          icon: getIconForType(searchType)
        });
      }

      // Add matching history items
      const matchingHistory = searchHistory
        .filter(item => item.toLowerCase().includes(query.toLowerCase()))
        .slice(0, 3)
        .map(item => ({
          type: 'history',
          value: item,
          searchType: getSearchType(item),
          icon: Clock
        }));

      // Add matching bookmarks
      const matchingBookmarks = bookmarks
        .filter(item => 
          item.value.toLowerCase().includes(query.toLowerCase()) ||
          (item.label && item.label.toLowerCase().includes(query.toLowerCase()))
        )
        .slice(0, 2)
        .map(item => ({
          type: 'bookmark',
          value: item.value,
          searchType: item.type,
          label: item.label,
          icon: Bookmark
        }));

      setSuggestions([...querySuggestions, ...matchingHistory, ...matchingBookmarks]);
    }

    setSelectedIndex(-1);
  }, [query, searchHistory, bookmarks]);

  const getIconForType = (type) => {
    switch (type) {
      case 'address':
        return User;
      case 'transaction':
        return Hash;
      case 'block':
        return Blocks;
      default:
        return Search;
    }
  };

  const getTypeLabel = (type) => {
    switch (type) {
      case 'address':
        return 'Address';
      case 'transaction':
        return 'Transaction';
      case 'block':
        return 'Block';
      default:
        return 'Search';
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'address':
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      case 'transaction':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'block':
        return 'bg-purple-500/10 text-purple-500 border-purple-500/20';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const handleSelect = (suggestion) => {
    onSelect?.(suggestion.value, suggestion.searchType);
    onClose?.();
  };

  const handleKeyDown = (e) => {
    if (!isVisible || suggestions.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
          handleSelect(suggestions[selectedIndex]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        onClose?.();
        break;
    }
  };

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isVisible, suggestions, selectedIndex]);

  useEffect(() => {
    if (selectedIndex >= 0 && suggestionRefs.current[selectedIndex]) {
      suggestionRefs.current[selectedIndex].scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      });
    }
  }, [selectedIndex]);

  if (!isVisible || suggestions.length === 0) {
    return null;
  }

  return (
    <Card className={cn(
      "absolute top-full left-0 right-0 mt-2 z-50 max-h-96 overflow-y-auto",
      className
    )}>
      <CardContent className="p-0">
        {!query.trim() && (
          <div className="px-4 py-2 border-b border-border">
            <div className="text-sm font-medium text-muted-foreground">
              Recent & Bookmarks
            </div>
          </div>
        )}
        
        <div className="py-2">
          {suggestions.map((suggestion, index) => {
            const Icon = suggestion.icon;
            const isSelected = index === selectedIndex;
            
            return (
              <Button
                key={`${suggestion.type}-${suggestion.value}-${index}`}
                ref={el => suggestionRefs.current[index] = el}
                variant="ghost"
                className={cn(
                  "w-full justify-start h-auto p-3 text-left hover:bg-muted/50",
                  isSelected && "bg-muted"
                )}
                onClick={() => handleSelect(suggestion)}
              >
                <div className="flex items-center space-x-3 w-full">
                  <Icon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-sm truncate">
                        {suggestion.searchType === 'address' || suggestion.searchType === 'transaction'
                          ? truncateHash(suggestion.value)
                          : suggestion.value
                        }
                      </span>
                      
                      {suggestion.searchType !== 'unknown' && (
                        <Badge className={getTypeColor(suggestion.searchType)}>
                          {getTypeLabel(suggestion.searchType)}
                        </Badge>
                      )}
                    </div>
                    
                    {suggestion.label && (
                      <div className="text-xs text-muted-foreground mt-1">
                        {suggestion.label}
                      </div>
                    )}
                  </div>
                  
                  {suggestion.type === 'history' && (
                    <Clock className="h-3 w-3 text-muted-foreground" />
                  )}
                  
                  {suggestion.type === 'bookmark' && (
                    <Bookmark className="h-3 w-3 text-primary" />
                  )}
                </div>
              </Button>
            );
          })}
        </div>
        
        {query.trim() && (
          <div className="px-4 py-2 border-t border-border">
            <div className="text-xs text-muted-foreground">
              Press Enter to search • Use ↑↓ to navigate • Esc to close
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SearchSuggestions;

