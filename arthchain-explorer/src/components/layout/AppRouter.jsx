import React, { useState } from 'react';
import Dashboard from '../explorer/Dashboard';
import BlocksPage from '../explorer/BlocksPage';
import TransactionsPage from '../explorer/TransactionsPage';
import AddressesPage from '../explorer/AddressesPage';
import ContractsPage from '../explorer/ContractsPage';
import Analytics from '../analytics/Analytics';
import BookmarksPage from '../tools/BookmarksPage';
import SettingsPage from '../tools/SettingsPage';

const AppRouter = () => {
  const [currentView, setCurrentView] = useState('dashboard');

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />;
      case 'blocks':
        return <BlocksPage />;
      case 'transactions':
        return <TransactionsPage />;
      case 'addresses':
        return <AddressesPage />;
      case 'contracts':
        return <ContractsPage />;
      case 'analytics':
        return <Analytics />;
      case 'bookmarks':
        return <BookmarksPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <Dashboard />;
    }
  };

  // Make setCurrentView available globally for navigation
  React.useEffect(() => {
    window.navigateTo = setCurrentView;
  }, []);

  return (
    <div className="flex-1 p-6">
      {renderView()}
    </div>
  );
};

export default AppRouter;

