import React, { useState, useRef, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import SearchSuggestions from './SearchSuggestions';
import useSearchStore from '../../stores/searchStore';
import { getSearchType, debounce } from '../../utils';

const EnhancedSearch = ({ 
  onSearch, 
  placeholder = "Search by address, transaction hash, or block number...",
  className 
}) => {
  const { 
    searchQuery, 
    setSearchQuery, 
    addToHistory, 
    setSearching,
    setSearchError 
  } = useSearchStore();
  
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [localQuery, setLocalQuery] = useState(searchQuery);
  const inputRef = useRef(null);
  const containerRef = useRef(null);

  // Debounced function to update global search query
  const debouncedSetQuery = debounce((query) => {
    setSearchQuery(query);
  }, 300);

  useEffect(() => {
    debouncedSetQuery(localQuery);
  }, [localQuery]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setLocalQuery(value);
    setShowSuggestions(true);
    setSearchError(null);
  };

  const handleInputFocus = () => {
    setShowSuggestions(true);
  };

  const handleSearch = async (query = localQuery, searchType = null) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const detectedType = searchType || getSearchType(trimmedQuery);
    
    setSearching(true);
    setShowSuggestions(false);
    
    try {
      // Add to search history
      addToHistory(trimmedQuery);
      
      // Trigger search callback
      await onSearch?.(trimmedQuery, detectedType);
      
    } catch (error) {
      console.error('Search error:', error);
      setSearchError(error.message || 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    handleSearch();
  };

  const handleSuggestionSelect = (value, searchType) => {
    setLocalQuery(value);
    setSearchQuery(value);
    handleSearch(value, searchType);
  };

  const handleClear = () => {
    setLocalQuery('');
    setSearchQuery('');
    setShowSuggestions(false);
    setSearchError(null);
    inputRef.current?.focus();
  };

  const handleCloseSuggestions = () => {
    setShowSuggestions(false);
  };

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <form onSubmit={handleFormSubmit} className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        
        <Input
          ref={inputRef}
          type="text"
          placeholder={placeholder}
          value={localQuery}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          className="pl-10 pr-10 w-full bg-input border-border focus:ring-primary"
          autoComplete="off"
        />
        
        {localQuery && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={handleClear}
            className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8 hover:bg-muted"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </form>

      <SearchSuggestions
        query={localQuery}
        isVisible={showSuggestions}
        onSelect={handleSuggestionSelect}
        onClose={handleCloseSuggestions}
      />
    </div>
  );
};

export default EnhancedSearch;

