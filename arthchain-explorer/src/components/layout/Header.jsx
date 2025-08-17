import React from 'react';
import { Menu, Bookmark, Settings, Home, Blocks, Activity, Users, FileText, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import EnhancedSearch from '../search/EnhancedSearch';
import useNetworkStore from '../../stores/networkStore';

const Header = ({ onMenuClick, onSearch, currentPath }) => {
  const { selectedNetwork, networks, setSelectedNetwork } = useNetworkStore();

  const handleNetworkChange = (network) => {
    setSelectedNetwork(network);
  };

  const handleNavigation = (route) => {
    if (window.navigateTo) {
      window.navigateTo(route);
    }
  };

  const handleBookmarkClick = () => {
    if (window.navigateTo) {
      window.navigateTo('bookmarks');
    }
  };

  const handleSettingsClick = () => {
    if (window.navigateTo) {
      window.navigateTo('settings');
    }
  };

  const handleSearch = async (query, searchType) => {
    console.log('Search:', query, searchType);
    
    // Navigate to appropriate page based on search type
    if (searchType === 'block') {
      window.navigateTo('blocks');
    } else if (searchType === 'transaction') {
      window.navigateTo('transactions');
    } else if (searchType === 'address') {
      window.navigateTo('addresses');
    } else if (searchType === 'contract') {
      window.navigateTo('contracts');
    } else {
      // Default to dashboard for general searches
      window.navigateTo('dashboard');
    }
    
    // Store search query for the target page to use
    window.currentSearchQuery = query;
    window.currentSearchType = searchType;
    
    onSearch?.(query, searchType);
  };

  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'blocks', label: 'Blocks', icon: Blocks },
    { id: 'transactions', label: 'Transactions', icon: Activity },
    { id: 'addresses', label: 'Addresses', icon: Users },
    { id: 'contracts', label: 'Contracts', icon: FileText },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo and Navigation Items */}
          <div className="flex items-center space-x-4">
            {/* Logo */}
            <div 
              className="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => handleNavigation('dashboard')}
            >
              <img 
                src="/arthaexpo-logo.avif" 
                alt="ArthaExpo" 
                className="h-8 w-auto"
              />
            </div>
            
            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={onMenuClick}
              className="md:hidden"
            >
              <Menu className="h-5 w-5" />
            </Button>
            
            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-1">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPath === item.id;
                return (
                  <Button
                    key={item.id}
                    variant={isActive ? "default" : "ghost"}
                    size="sm"
                    onClick={() => handleNavigation(item.id)}
                    className="flex items-center space-x-2"
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </Button>
                );
              })}
            </div>
          </div>

          {/* Enhanced Search Bar */}
          <div className="flex-1 max-w-2xl mx-4">
            <EnhancedSearch onSearch={handleSearch} />
          </div>

          {/* Network Selector and Actions */}
          <div className="flex items-center space-x-2">
            {/* Network Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="hidden sm:flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: selectedNetwork.color }}
                  />
                  <span>{selectedNetwork.name}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {networks.map((network) => (
                  <DropdownMenuItem
                    key={network.id}
                    onClick={() => handleNetworkChange(network)}
                    className="flex items-center space-x-2"
                  >
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: network.color }}
                    />
                    <span>{network.name}</span>
                    {selectedNetwork.id === network.id && (
                      <span className="ml-auto text-primary">✓</span>
                    )}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Bookmarks */}
            <Button 
              variant="ghost" 
              size="icon" 
              className="hidden lg:flex"
              onClick={handleBookmarkClick}
            >
              <Bookmark className="h-5 w-5" />
            </Button>

            {/* Settings */}
            <Button 
              variant="ghost" 
              size="icon" 
              className="hidden lg:flex"
              onClick={handleSettingsClick}
            >
              <Settings className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;