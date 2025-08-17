import React, { useState } from 'react';
import { Mail, Send, Github, Twitter, Linkedin, MessageCircle, Globe, FileText, Users, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';

const Footer = () => {
  const [email, setEmail] = useState('');
  const [isSubscribed, setIsSubscribed] = useState(false);

  const handleSubscribe = (e) => {
    e.preventDefault();
    if (email) {
      setIsSubscribed(true);
      setEmail('');
      setTimeout(() => setIsSubscribed(false), 3000);
    }
  };

  const handleLinkClick = (href, linkName) => {
    // Handle internal navigation
    if (href === '#') {
      const navigationMap = {
        'Dashboard': 'dashboard',
        'Blocks': 'blocks',
        'Transactions': 'transactions',
        'Addresses': 'addresses',
        'Smart Contracts': 'contracts',
        'Analytics': 'analytics'
      };
      
      const viewName = navigationMap[linkName];
      if (viewName && window.navigateTo) {
        window.navigateTo(viewName);
      }
    } else {
      // Handle external links
      window.open(href, '_blank');
    }
  };

  const footerLinks = {
    'Explorer': [
      { name: 'Dashboard', href: '#', icon: Globe },
      { name: 'Blocks', href: '#', icon: Zap },
      { name: 'Transactions', href: '#', icon: FileText },
      { name: 'Addresses', href: '#', icon: Users },
      { name: 'Smart Contracts', href: '#', icon: FileText },
      { name: 'Analytics', href: '#', icon: Zap }
    ],
    'Resources': [
      { name: 'Documentation', href: '#', icon: FileText },
      { name: 'API Reference', href: '#', icon: FileText },
      { name: 'White Paper', href: '#', icon: FileText },
      { name: 'Developer Guide', href: '#', icon: FileText },
      { name: 'Community', href: '#', icon: Users },
      { name: 'Support', href: '#', icon: MessageCircle }
    ],
    'Network': [
      { name: 'Mainnet', href: '#', icon: Globe },
      { name: 'Testnet', href: '#', icon: Globe },
      { name: 'Network Stats', href: '#', icon: Zap },
      { name: 'Validators', href: '#', icon: Users },
      { name: 'Governance', href: '#', icon: Users },
      { name: 'Staking', href: '#', icon: Zap }
    ],
    'Company': [
      { name: 'About Us', href: '#', icon: Users },
      { name: 'Team', href: '#', icon: Users },
      { name: 'Careers', href: '#', icon: Users },
      { name: 'Press Kit', href: '#', icon: FileText },
      { name: 'Contact', href: '#', icon: MessageCircle },
      { name: 'Privacy Policy', href: '#', icon: FileText }
    ]
  };

  const socialLinks = [
    { name: 'Twitter', icon: Twitter, href: '#', color: 'hover:text-blue-400' },
    { name: 'GitHub', icon: Github, href: '#', color: 'hover:text-gray-300' },
    { name: 'LinkedIn', icon: Linkedin, href: '#', color: 'hover:text-blue-500' },
    { name: 'Discord', icon: MessageCircle, href: '#', color: 'hover:text-purple-400' }
  ];

  return (
    <footer className="bg-card border-t border-border mt-auto">
      {/* Newsletter Section */}
      <div className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div>
              <h3 className="text-2xl font-bold mb-2">Get weekly news on blockchain.</h3>
              <p className="text-muted-foreground">
                Stay updated with the latest developments in the Arthachain ecosystem, 
                new features, and blockchain insights delivered to your inbox.
              </p>
            </div>
            <div className="lg:justify-self-end w-full lg:w-auto">
              <form onSubmit={handleSubscribe} className="flex flex-col sm:flex-row gap-3 max-w-md">
                <Input
                  type="email"
                  placeholder="Your mail address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="flex-1 bg-background border-border"
                  required
                />
                <Button 
                  type="submit" 
                  className="gradient-primary text-black font-semibold px-6 whitespace-nowrap"
                  disabled={isSubscribed}
                >
                  {isSubscribed ? (
                    <>
                      <Mail className="h-4 w-4 mr-2" />
                      Subscribed!
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Subscribe
                    </>
                  )}
                </Button>
              </form>
              {isSubscribed && (
                <p className="text-sm text-green-500 mt-2">
                  Thank you for subscribing to our newsletter!
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Footer Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8">
          {/* Logo and Description */}
          <div className="lg:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <img 
                src="/arthaexpo-logo.avif" 
                alt="ArthaExpo" 
                className="h-8 w-auto"
              />
              <span className="font-bold text-xl">ArthaExpo</span>
            </div>
            <p className="text-muted-foreground mb-6 max-w-sm">
              The most comprehensive blockchain explorer for Arthachain. 
              Explore blocks, transactions, addresses, and smart contracts 
              with support for both EVM and WASM runtimes.
            </p>
            
            {/* Decorative Image */}
            <div className="mb-6">
              <img 
                src="/footer-image.avif" 
                alt="Arthachain Technology" 
                className="w-full max-w-xs h-24 object-cover rounded-lg opacity-80"
              />
            </div>
            
            {/* Social Links */}
            <div className="flex space-x-4">
              {socialLinks.map((social) => {
                const Icon = social.icon;
                return (
                  <a
                    key={social.name}
                    href={social.href}
                    className={`text-muted-foreground transition-colors ${social.color}`}
                    aria-label={social.name}
                    onClick={(e) => {
                      e.preventDefault();
                      handleLinkClick(social.href, social.name);
                    }}
                  >
                    <Icon className="h-5 w-5" />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Footer Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="font-semibold text-foreground mb-4">{category}</h4>
              <ul className="space-y-3">
                {links.map((link) => {
                  const Icon = link.icon;
                  return (
                    <li key={link.name}>
                      <a
                        href={link.href}
                        className="text-muted-foreground hover:text-primary transition-colors flex items-center space-x-2 group"
                        onClick={(e) => {
                          e.preventDefault();
                          handleLinkClick(link.href, link.name);
                        }}
                      >
                        <Icon className="h-4 w-4 opacity-60 group-hover:opacity-100 transition-opacity" />
                        <span>{link.name}</span>
                      </a>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex flex-col md:flex-row items-center space-y-2 md:space-y-0 md:space-x-6">
              <p className="text-sm text-muted-foreground">
                © 2025 ArthaExpo. All rights reserved.
              </p>
              <div className="flex items-center space-x-4 text-sm">
                <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                  Terms of Service
                </a>
                <span className="text-muted-foreground">•</span>
                <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                  Privacy Policy
                </a>
                <span className="text-muted-foreground">•</span>
                <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                  Cookie Policy
                </a>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                <span>Network Status: Online</span>
              </div>
              <div className="text-sm text-muted-foreground">
                Block Height: 18,756,234
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Decorative Elements */}
      <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent"></div>
    </footer>
  );
};

export default Footer;

