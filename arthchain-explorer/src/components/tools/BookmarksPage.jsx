import React, { useState, useEffect } from 'react';
import { Bookmark, Hash, User, Code, Trash2, Plus, Search } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import CopyButton from '../ui/CopyButton';
import { truncateHash, formatTimestamp } from '../../utils';
import useNetworkStore from '../../stores/networkStore';

const BookmarksPage = () => {
  const { selectedNetwork } = useNetworkStore();
  const [bookmarks, setBookmarks] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newBookmark, setNewBookmark] = useState({
    address: '',
    name: '',
    type: 'address',
    description: ''
  });

  // Load bookmarks from localStorage on component mount
  useEffect(() => {
    const savedBookmarks = localStorage.getItem('arthachain_bookmarks');
    if (savedBookmarks) {
      setBookmarks(JSON.parse(savedBookmarks));
    } else {
      // Initialize with some default bookmarks
      const defaultBookmarks = [
        {
          id: '1',
          address: '0xArthachainTreasuryFund1234567890abcdef',
          name: 'Arthachain Treasury',
          type: 'address',
          description: 'Official treasury fund for Arthachain ecosystem',
          createdAt: Date.now() - 86400000 * 10
        },
        {
          id: '2',
          address: '0xArthachainDEXContract1234567890abcdef',
          name: 'Arthachain DEX',
          type: 'contract',
          description: 'Decentralized exchange smart contract',
          createdAt: Date.now() - 86400000 * 5
        },
        {
          id: '3',
          address: '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b1',
          name: 'Genesis Block',
          type: 'block',
          description: 'The first block on Arthachain',
          createdAt: Date.now() - 86400000 * 30
        }
      ];
      setBookmarks(defaultBookmarks);
      localStorage.setItem('arthachain_bookmarks', JSON.stringify(defaultBookmarks));
    }
  }, []);

  // Save bookmarks to localStorage whenever bookmarks change
  useEffect(() => {
    localStorage.setItem('arthachain_bookmarks', JSON.stringify(bookmarks));
  }, [bookmarks]);

  const handleAddBookmark = () => {
    if (newBookmark.address && newBookmark.name) {
      const bookmark = {
        id: Date.now().toString(),
        ...newBookmark,
        createdAt: Date.now()
      };
      setBookmarks(prev => [bookmark, ...prev]);
      setNewBookmark({ address: '', name: '', type: 'address', description: '' });
      setIsAddDialogOpen(false);
    }
  };

  const handleDeleteBookmark = (id) => {
    setBookmarks(prev => prev.filter(bookmark => bookmark.id !== id));
  };

  const filteredBookmarks = bookmarks.filter(bookmark =>
    bookmark.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    bookmark.address.toLowerCase().includes(searchQuery.toLowerCase()) ||
    bookmark.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getTypeIcon = (type) => {
    switch (type) {
      case 'address':
        return <User className="h-4 w-4" />;
      case 'contract':
        return <Code className="h-4 w-4" />;
      case 'transaction':
        return <Hash className="h-4 w-4" />;
      case 'block':
        return <Hash className="h-4 w-4" />;
      default:
        return <Bookmark className="h-4 w-4" />;
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'address':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      case 'contract':
        return 'bg-purple-500/10 text-purple-500 border-purple-500/20';
      case 'transaction':
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      case 'block':
        return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
      default:
        return 'bg-gray-500/10 text-gray-500 border-gray-500/20';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Bookmarks</h1>
          <p className="text-muted-foreground">
            Your saved items on {selectedNetwork.name}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: selectedNetwork.color }}
          />
          <span className="font-medium">{selectedNetwork.name}</span>
        </div>
      </div>

      {/* Search and Add */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col space-y-4 md:flex-row md:space-y-0 md:space-x-4">
            <div className="flex space-x-2 flex-1">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search bookmarks..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            
            <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
              <DialogTrigger asChild>
                <Button className="flex items-center space-x-2">
                  <Plus className="h-4 w-4" />
                  <span>Add Bookmark</span>
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add New Bookmark</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="address">Address/Hash</Label>
                    <Input
                      id="address"
                      placeholder="0x..."
                      value={newBookmark.address}
                      onChange={(e) => setNewBookmark(prev => ({ ...prev, address: e.target.value }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="name">Name</Label>
                    <Input
                      id="name"
                      placeholder="My Bookmark"
                      value={newBookmark.name}
                      onChange={(e) => setNewBookmark(prev => ({ ...prev, name: e.target.value }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="type">Type</Label>
                    <Select value={newBookmark.type} onValueChange={(value) => setNewBookmark(prev => ({ ...prev, type: value }))}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="address">Address</SelectItem>
                        <SelectItem value="contract">Contract</SelectItem>
                        <SelectItem value="transaction">Transaction</SelectItem>
                        <SelectItem value="block">Block</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="description">Description (Optional)</Label>
                    <Textarea
                      id="description"
                      placeholder="Add a description..."
                      value={newBookmark.description}
                      onChange={(e) => setNewBookmark(prev => ({ ...prev, description: e.target.value }))}
                    />
                  </div>
                  <div className="flex justify-end space-x-2">
                    <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAddBookmark}>
                      Add Bookmark
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardContent>
      </Card>

      {/* Bookmarks List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Bookmark className="h-5 w-5" />
              <span>Your Bookmarks</span>
            </div>
            <div className="text-sm text-muted-foreground">
              {filteredBookmarks.length} bookmarks
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {filteredBookmarks.length === 0 ? (
            <div className="text-center py-12">
              <Bookmark className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">
                {searchQuery ? 'No bookmarks match your search.' : 'No bookmarks saved yet.'}
              </p>
              {!searchQuery && (
                <Button 
                  className="mt-4" 
                  onClick={() => setIsAddDialogOpen(true)}
                >
                  Add Your First Bookmark
                </Button>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {filteredBookmarks.map((bookmark) => (
                <div 
                  key={bookmark.id}
                  className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                      {getTypeIcon(bookmark.type)}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold">{bookmark.name}</span>
                        <Badge className={getTypeColor(bookmark.type)}>
                          {bookmark.type.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {truncateHash(bookmark.address)} • Added {formatTimestamp(Math.floor(bookmark.createdAt / 1000))}
                      </div>
                      {bookmark.description && (
                        <div className="text-sm text-muted-foreground mt-1">
                          {bookmark.description}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <CopyButton text={bookmark.address} />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteBookmark(bookmark.id)}
                      className="text-red-500 hover:text-red-600 hover:bg-red-500/10"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Statistics */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <div className="text-2xl font-bold">{bookmarks.filter(b => b.type === 'address').length}</div>
              <div className="text-sm text-muted-foreground">Addresses</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <div className="text-2xl font-bold">{bookmarks.filter(b => b.type === 'contract').length}</div>
              <div className="text-sm text-muted-foreground">Contracts</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <div className="text-2xl font-bold">{bookmarks.filter(b => b.type === 'transaction').length}</div>
              <div className="text-sm text-muted-foreground">Transactions</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <div className="text-2xl font-bold">{bookmarks.filter(b => b.type === 'block').length}</div>
              <div className="text-sm text-muted-foreground">Blocks</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default BookmarksPage;

