import React, { useState, useEffect } from 'react';
import { Settings, Monitor, Palette, Bell, Database, Shield, Globe, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import useNetworkStore from '../../stores/networkStore';

const SettingsPage = () => {
  const { selectedNetwork } = useNetworkStore();
  const [settings, setSettings] = useState({
    theme: 'dark',
    language: 'en',
    currency: 'USD',
    notifications: {
      newBlocks: true,
      priceAlerts: false,
      systemUpdates: true
    },
    display: {
      compactMode: false,
      showTestnets: false,
      autoRefresh: true,
      refreshInterval: 30
    },
    privacy: {
      analytics: true,
      crashReports: true,
      personalizedAds: false
    },
    advanced: {
      debugMode: false,
      experimentalFeatures: false,
      cacheSize: 100
    }
  });

  // Load settings from localStorage on component mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('arthachain_settings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  // Save settings to localStorage whenever settings change
  useEffect(() => {
    localStorage.setItem('arthachain_settings', JSON.stringify(settings));
  }, [settings]);

  const updateSetting = (path, value) => {
    setSettings(prev => {
      const newSettings = { ...prev };
      const keys = path.split('.');
      let current = newSettings;
      
      for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
      }
      
      current[keys[keys.length - 1]] = value;
      return newSettings;
    });
  };

  const resetSettings = () => {
    const defaultSettings = {
      theme: 'dark',
      language: 'en',
      currency: 'USD',
      notifications: {
        newBlocks: true,
        priceAlerts: false,
        systemUpdates: true
      },
      display: {
        compactMode: false,
        showTestnets: false,
        autoRefresh: true,
        refreshInterval: 30
      },
      privacy: {
        analytics: true,
        crashReports: true,
        personalizedAds: false
      },
      advanced: {
        debugMode: false,
        experimentalFeatures: false,
        cacheSize: 100
      }
    };
    setSettings(defaultSettings);
  };

  const clearCache = () => {
    // Clear various cache items
    const cacheKeys = [
      'arthachain_blocks_cache',
      'arthachain_transactions_cache',
      'arthachain_addresses_cache',
      'arthachain_search_history'
    ];
    
    cacheKeys.forEach(key => {
      localStorage.removeItem(key);
    });
    
    // Show success message (in a real app, you'd use a toast notification)
    alert('Cache cleared successfully!');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground">
            Configure your Arthachain explorer preferences
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

      {/* Appearance Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Palette className="h-5 w-5" />
            <span>Appearance</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="theme">Theme</Label>
              <p className="text-sm text-muted-foreground">Choose your preferred color scheme</p>
            </div>
            <Select value={settings.theme} onValueChange={(value) => updateSetting('theme', value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="system">System</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="language">Language</Label>
              <p className="text-sm text-muted-foreground">Select your preferred language</p>
            </div>
            <Select value={settings.language} onValueChange={(value) => updateSetting('language', value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="es">Español</SelectItem>
                <SelectItem value="fr">Français</SelectItem>
                <SelectItem value="de">Deutsch</SelectItem>
                <SelectItem value="zh">中文</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="currency">Currency</Label>
              <p className="text-sm text-muted-foreground">Display prices in your preferred currency</p>
            </div>
            <Select value={settings.currency} onValueChange={(value) => updateSetting('currency', value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="USD">USD</SelectItem>
                <SelectItem value="EUR">EUR</SelectItem>
                <SelectItem value="GBP">GBP</SelectItem>
                <SelectItem value="JPY">JPY</SelectItem>
                <SelectItem value="BTC">BTC</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Display Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Monitor className="h-5 w-5" />
            <span>Display</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="compact-mode">Compact Mode</Label>
              <p className="text-sm text-muted-foreground">Show more information in less space</p>
            </div>
            <Switch
              id="compact-mode"
              checked={settings.display.compactMode}
              onCheckedChange={(checked) => updateSetting('display.compactMode', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="show-testnets">Show Testnets</Label>
              <p className="text-sm text-muted-foreground">Display test networks in network selector</p>
            </div>
            <Switch
              id="show-testnets"
              checked={settings.display.showTestnets}
              onCheckedChange={(checked) => updateSetting('display.showTestnets', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="auto-refresh">Auto Refresh</Label>
              <p className="text-sm text-muted-foreground">Automatically refresh data</p>
            </div>
            <Switch
              id="auto-refresh"
              checked={settings.display.autoRefresh}
              onCheckedChange={(checked) => updateSetting('display.autoRefresh', checked)}
            />
          </div>

          {settings.display.autoRefresh && (
            <>
              <Separator />
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Refresh Interval</Label>
                  <span className="text-sm text-muted-foreground">{settings.display.refreshInterval}s</span>
                </div>
                <Slider
                  value={[settings.display.refreshInterval]}
                  onValueChange={([value]) => updateSetting('display.refreshInterval', value)}
                  max={120}
                  min={5}
                  step={5}
                  className="w-full"
                />
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Notification Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bell className="h-5 w-5" />
            <span>Notifications</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="new-blocks">New Blocks</Label>
              <p className="text-sm text-muted-foreground">Get notified when new blocks are mined</p>
            </div>
            <Switch
              id="new-blocks"
              checked={settings.notifications.newBlocks}
              onCheckedChange={(checked) => updateSetting('notifications.newBlocks', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="price-alerts">Price Alerts</Label>
              <p className="text-sm text-muted-foreground">Receive alerts for significant price changes</p>
            </div>
            <Switch
              id="price-alerts"
              checked={settings.notifications.priceAlerts}
              onCheckedChange={(checked) => updateSetting('notifications.priceAlerts', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="system-updates">System Updates</Label>
              <p className="text-sm text-muted-foreground">Get notified about system updates and maintenance</p>
            </div>
            <Switch
              id="system-updates"
              checked={settings.notifications.systemUpdates}
              onCheckedChange={(checked) => updateSetting('notifications.systemUpdates', checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Privacy Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Shield className="h-5 w-5" />
            <span>Privacy</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="analytics">Analytics</Label>
              <p className="text-sm text-muted-foreground">Help improve the app by sharing usage data</p>
            </div>
            <Switch
              id="analytics"
              checked={settings.privacy.analytics}
              onCheckedChange={(checked) => updateSetting('privacy.analytics', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="crash-reports">Crash Reports</Label>
              <p className="text-sm text-muted-foreground">Automatically send crash reports to help fix bugs</p>
            </div>
            <Switch
              id="crash-reports"
              checked={settings.privacy.crashReports}
              onCheckedChange={(checked) => updateSetting('privacy.crashReports', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="personalized-ads">Personalized Ads</Label>
              <p className="text-sm text-muted-foreground">Show ads based on your interests</p>
            </div>
            <Switch
              id="personalized-ads"
              checked={settings.privacy.personalizedAds}
              onCheckedChange={(checked) => updateSetting('privacy.personalizedAds', checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Advanced</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="debug-mode">Debug Mode</Label>
              <p className="text-sm text-muted-foreground">Enable debug logging and developer tools</p>
            </div>
            <Switch
              id="debug-mode"
              checked={settings.advanced.debugMode}
              onCheckedChange={(checked) => updateSetting('advanced.debugMode', checked)}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="experimental-features">Experimental Features</Label>
              <p className="text-sm text-muted-foreground">Enable beta features (may be unstable)</p>
            </div>
            <Switch
              id="experimental-features"
              checked={settings.advanced.experimentalFeatures}
              onCheckedChange={(checked) => updateSetting('advanced.experimentalFeatures', checked)}
            />
          </div>

          <Separator />

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Cache Size</Label>
              <span className="text-sm text-muted-foreground">{settings.advanced.cacheSize} MB</span>
            </div>
            <Slider
              value={[settings.advanced.cacheSize]}
              onValueChange={([value]) => updateSetting('advanced.cacheSize', value)}
              max={500}
              min={50}
              step={25}
              className="w-full"
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label>Clear Cache</Label>
              <p className="text-sm text-muted-foreground">Clear all cached data to free up space</p>
            </div>
            <Button variant="outline" onClick={clearCache}>
              <Database className="h-4 w-4 mr-2" />
              Clear Cache
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Reset Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <RefreshCw className="h-5 w-5" />
            <span>Reset</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <Label>Reset All Settings</Label>
              <p className="text-sm text-muted-foreground">Restore all settings to their default values</p>
            </div>
            <Button variant="destructive" onClick={resetSettings}>
              Reset Settings
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SettingsPage;

