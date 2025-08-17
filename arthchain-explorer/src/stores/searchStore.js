import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { LOCAL_STORAGE_KEYS } from '../constants';

const useSearchStore = create(
  persist(
    (set, get) => ({
      // Search state
      searchQuery: '',
      searchResults: [],
      isSearching: false,
      searchError: null,
      
      // Search history
      searchHistory: [],
      
      // Bookmarks
      bookmarks: [],
      
      // Actions
      setSearchQuery: (query) => {
        set({ searchQuery: query });
      },
      
      setSearchResults: (results) => {
        set({ searchResults: results });
      },
      
      setSearching: (searching) => {
        set({ isSearching: searching });
      },
      
      setSearchError: (error) => {
        set({ 
          searchError: error,
          isSearching: false
        });
      },
      
      // Add to search history
      addToHistory: (query) => {
        const { searchHistory } = get();
        const newHistory = [
          query,
          ...searchHistory.filter(item => item !== query)
        ].slice(0, 10); // Keep only last 10 searches
        
        set({ searchHistory: newHistory });
      },
      
      // Clear search history
      clearHistory: () => {
        set({ searchHistory: [] });
      },
      
      // Bookmark management
      addBookmark: (item) => {
        const { bookmarks } = get();
        const exists = bookmarks.find(bookmark => 
          bookmark.type === item.type && bookmark.value === item.value
        );
        
        if (!exists) {
          set({ 
            bookmarks: [...bookmarks, { ...item, timestamp: Date.now() }]
          });
          return true;
        }
        return false;
      },
      
      removeBookmark: (item) => {
        const { bookmarks } = get();
        set({
          bookmarks: bookmarks.filter(bookmark => 
            !(bookmark.type === item.type && bookmark.value === item.value)
          )
        });
      },
      
      isBookmarked: (item) => {
        const { bookmarks } = get();
        return bookmarks.some(bookmark => 
          bookmark.type === item.type && bookmark.value === item.value
        );
      },
      
      // Clear all search data
      clearSearch: () => {
        set({
          searchQuery: '',
          searchResults: [],
          isSearching: false,
          searchError: null
        });
      }
    }),
    {
      name: LOCAL_STORAGE_KEYS.SEARCH_HISTORY,
      partialize: (state) => ({ 
        searchHistory: state.searchHistory,
        bookmarks: state.bookmarks
      })
    }
  )
);

export default useSearchStore;

