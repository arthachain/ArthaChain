# ArthChain Explorer Deployment Guide

This document provides instructions for deploying the ArthChain Explorer to testnet environments.

## Prerequisites

- Node.js 18+ and pnpm
- Web server (nginx, apache, or similar)
- SSL certificates for HTTPS (recommended)

## Environment Configuration

The explorer is configured to connect to:
- **Primary Domain**: https://testnet.arthachain.online
- **Secondary Domain**: https://testnet.arthachain.in

## Quick Deployment

### 1. Build for Production

```bash
# Install dependencies
pnpm install

# Build for production
pnpm run build:production
```

### 2. Deploy using Script

```bash
# Create deployment package
pnpm run deploy:testnet
```

This creates `arthachain-explorer-testnet.tar.gz` ready for deployment.

### 3. Manual Deployment

```bash
# On your server
wget [upload-url]/arthachain-explorer-testnet.tar.gz
tar -xzf arthachain-explorer-testnet.tar.gz -C /var/www/arthachain-explorer
```

## Web Server Configuration

### Nginx Configuration

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name testnet.arthachain.online testnet.arthachain.in;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name testnet.arthachain.online testnet.arthachain.in;

    ssl_certificate /path/to/ssl/certificate.crt;
    ssl_certificate_key /path/to/ssl/private.key;

    root /var/www/arthachain-explorer;
    index index.html;

    # Enable gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # CORS headers for API requests
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;

    # Handle client-side routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # No cache for index.html
    location = /index.html {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
}
```

### Apache Configuration

```apache
<VirtualHost *:80>
    ServerName testnet.arthachain.online
    ServerAlias testnet.arthachain.in
    Redirect permanent / https://testnet.arthachain.online/
</VirtualHost>

<VirtualHost *:443>
    ServerName testnet.arthachain.online
    ServerAlias testnet.arthachain.in
    DocumentRoot /var/www/arthachain-explorer

    SSLEngine on
    SSLCertificateFile /path/to/ssl/certificate.crt
    SSLCertificateKeyFile /path/to/ssl/private.key

    # Enable compression
    LoadModule deflate_module modules/mod_deflate.so
    <Location />
        SetOutputFilter DEFLATE
        SetEnvIfNoCase Request_URI \
            \.(?:gif|jpe?g|png)$ no-gzip dont-vary
        SetEnvIfNoCase Request_URI \
            \.(?:exe|t?gz|zip|bz2|sit|rar)$ no-gzip dont-vary
    </Location>

    # Handle client-side routing
    <Directory "/var/www/arthachain-explorer">
        RewriteEngine On
        RewriteBase /
        RewriteRule ^index\.html$ - [L]
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>

    # Security headers
    Header always set X-Frame-Options SAMEORIGIN
    Header always set X-Content-Type-Options nosniff
    Header always set X-XSS-Protection "1; mode=block"
    
    # CORS headers
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, OPTIONS"
    Header always set Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range"
</VirtualHost>
```

## Features Enabled

✅ **Real-time blockchain data** - Connected to ArthChain testnet APIs  
✅ **Block explorer** - Browse blocks, transactions, and addresses  
✅ **Smart contracts** - View EVM and WASM contracts  
✅ **Network analytics** - Real-time charts and statistics  
✅ **Search functionality** - Find blocks, transactions, and addresses  
✅ **Mobile responsive** - Optimized for all devices  

## API Endpoints

The explorer connects to these ArthChain testnet endpoints:

- **Base API**: `https://testnet.arthachain.online/api/`
- **Status**: `/api/status`
- **Blocks**: `/api/blocks/*`
- **Transactions**: `/api/transactions/*`
- **Accounts**: `/api/accounts/*`
- **Consensus**: `/api/consensus/*`
- **AI Engine**: `/api/ai/*`

## Monitoring

Monitor the deployment with:

```bash
# Check if the site is accessible
curl -I https://testnet.arthachain.online

# Check API connectivity
curl https://testnet.arthachain.online/api/status
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API endpoints are accessible
   - Check CORS configuration
   - Ensure SSL certificates are valid

2. **Routing Issues**
   - Verify web server is configured for SPA routing
   - Check `.htaccess` or nginx configuration

3. **Performance Issues**
   - Enable gzip compression
   - Configure proper caching headers
   - Monitor server resources

### Support

For deployment support, contact the ArthChain development team or create an issue in the project repository.

## Updates

To update the deployed explorer:

1. Pull latest changes from repository
2. Run `pnpm run deploy:testnet`
3. Replace files on the web server
4. Clear any CDN caches if applicable
