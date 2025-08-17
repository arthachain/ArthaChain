import React from 'react';
import { 
  Home, 
  Blocks, 
  Activity, 
  Users, 
  FileText, 
  BarChart3, 
  Bookmark,
  Settings,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const sidebarItems = [
  {
    title: 'Dashboard',
    icon: Home,
    href: '/',
    description: 'Network overview and statistics'
  },
  {
    title: 'Blocks',
    icon: Blocks,
    href: '/blocks',
    description: 'Browse blockchain blocks'
  },
  {
    title: 'Transactions',
    icon: Activity,
    href: '/transactions',
    description: 'View transaction history'
  },
  {
    title: 'Addresses',
    icon: Users,
    href: '/addresses',
    description: 'Explore wallet addresses'
  },
  {
    title: 'Contracts',
    icon: FileText,
    href: '/contracts',
    description: 'Smart contract explorer'
  },
  {
    title: 'Analytics',
    icon: BarChart3,
    href: '/analytics',
    description: 'Network analytics and charts'
  }
];

const utilityItems = [
  {
    title: 'Bookmarks',
    icon: Bookmark,
    href: '/bookmarks',
    description: 'Your saved items'
  },
  {
    title: 'Settings',
    icon: Settings,
    href: '/settings',
    description: 'Application settings'
  }
];

const Sidebar = ({ isOpen, onClose, currentPath = '/' }) => {
  const handleItemClick = (href) => {
    // Map href to view names
    const viewMap = {
      '/': 'dashboard',
      '/blocks': 'blocks',
      '/transactions': 'transactions',
      '/addresses': 'addresses',
      '/contracts': 'contracts',
      '/analytics': 'analytics',
      '/bookmarks': 'bookmarks',
      '/settings': 'settings'
    };
    
    const viewName = viewMap[href] || 'dashboard';
    
    // Use the global navigation function
    if (window.navigateTo) {
      window.navigateTo(viewName);
    }
    
    onClose?.();
  };

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <aside className={cn(
        "fixed left-0 top-0 z-50 h-screen w-64 transform bg-card border-r border-border transition-transform duration-300 ease-in-out md:hidden",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="flex h-16 items-center justify-between px-4 border-b border-border">
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg">Navigation</span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="md:hidden"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4">
            <div className="space-y-2">
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Explorer
              </div>
              
              {sidebarItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPath === item.href;
                
                return (
                  <Button
                    key={item.href}
                    variant={isActive ? "secondary" : "ghost"}
                    className={cn(
                      "w-full justify-start h-auto p-3 text-left",
                      isActive && "bg-primary/10 text-primary border-primary/20"
                    )}
                    onClick={() => handleItemClick(item.href)}
                  >
                    <div className="flex items-start space-x-3">
                      <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium">{item.title}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {item.description}
                        </div>
                      </div>
                    </div>
                  </Button>
                );
              })}
            </div>

            <div className="mt-8 space-y-2">
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Tools
              </div>
              
              {utilityItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPath === item.href;
                
                return (
                  <Button
                    key={item.href}
                    variant={isActive ? "secondary" : "ghost"}
                    className={cn(
                      "w-full justify-start h-auto p-3 text-left",
                      isActive && "bg-primary/10 text-primary border-primary/20"
                    )}
                    onClick={() => handleItemClick(item.href)}
                  >
                    <div className="flex items-start space-x-3">
                      <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium">{item.title}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {item.description}
                        </div>
                      </div>
                    </div>
                  </Button>
                );
              })}
            </div>
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-border">
            <div className="text-xs text-muted-foreground text-center">
              <div>ArthaExpo v1.0</div>
              <div className="mt-1">Modular Blockchain Explorer</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;

