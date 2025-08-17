import React, { useState } from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import AppRouter from './AppRouter';
import Footer from './Footer';

const Layout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentPath, setCurrentPath] = useState('dashboard');

  React.useEffect(() => {
    const handleNavigation = (event) => {
      if (event.detail && event.detail.route) {
        setCurrentPath(event.detail.route);
      }
    };

    window.addEventListener('navigate', handleNavigation);
    return () => window.removeEventListener('navigate', handleNavigation);
  }, []);

  const handleMenuClick = () => {
    setSidebarOpen(true);
  };

  const handleSidebarClose = () => {
    setSidebarOpen(false);
  };

  const handleSearch = (query, searchType) => {
    console.log('Search:', query, searchType);
    // In a real app, this would trigger navigation or search results
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header with integrated navigation */}
      <Header 
        onMenuClick={handleMenuClick}
        onSearch={handleSearch}
        currentPath={currentPath}
      />
      
      <div className="flex flex-1">
        {/* Mobile Sidebar (only visible on mobile when opened) */}
        <Sidebar 
          isOpen={sidebarOpen}
          onClose={handleSidebarClose}
          currentPath={currentPath}
        />
        
        {/* Main Content */}
        <main className="flex-1 min-h-0 w-full">
          <div className="container mx-auto px-3 sm:px-4 lg:px-6 xl:px-8 py-4 sm:py-6 max-w-7xl">
            <AppRouter />
          </div>
        </main>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
};
export default Layout;

