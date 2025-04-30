const axios = require('axios');
const crypto = require('crypto');

class SvdbClient {
    constructor(config) {
        this.baseUrl = config.baseUrl;
        this.timeout = config.timeout || 30000;
        this.encryptionKey = config.encryptionKey;
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async store(key, data, options = {}) {
        try {
            let processedData = data;
            if (options.encrypted && this.encryptionKey) {
                processedData = this.encryptData(data);
            }
            
            const response = await this.client.post('/store', {
                key,
                data: Buffer.from(processedData).toString('base64'),
                encrypted: options.encrypted || false
            });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async retrieve(key) {
        try {
            const response = await this.client.post('/retrieve', { key });
            if (response.data.success && response.data.data) {
                let data = Buffer.from(response.data.data, 'base64');
                if (response.data.encrypted && this.encryptionKey) {
                    data = this.decryptData(data);
                }
                return data;
            }
            throw new Error(response.data.message || 'Failed to retrieve data');
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async delete(key) {
        try {
            const response = await this.client.post('/delete', { key });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async verify(key, hash) {
        try {
            const response = await this.client.post('/verify', { key, hash });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async exists(key) {
        try {
            const response = await this.client.post('/exists', { key });
            return response.data.exists;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async batchStore(operations) {
        try {
            const processedOperations = operations.map(op => ({
                key: op.key,
                data: Buffer.from(op.data).toString('base64'),
                encrypted: op.encrypted || false
            }));

            const response = await this.client.post('/batch/store', {
                operations: processedOperations
            });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async batchRetrieve(keys) {
        try {
            const response = await this.client.post('/batch/retrieve', { keys });
            if (response.data.success && response.data.data) {
                return response.data.data.map(item => ({
                    key: item.key,
                    data: Buffer.from(item.data, 'base64'),
                    encrypted: item.encrypted
                }));
            }
            throw new Error(response.data.message || 'Failed to retrieve batch data');
        } catch (error) {
            throw this.handleError(error);
        }
    }

    async batchDelete(keys) {
        try {
            const response = await this.client.post('/batch/delete', { keys });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    encryptData(data) {
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipheriv('aes-256-gcm', this.encryptionKey, iv);
        const encrypted = Buffer.concat([cipher.update(data), cipher.final()]);
        const authTag = cipher.getAuthTag();
        return Buffer.concat([iv, authTag, encrypted]);
    }

    decryptData(encryptedData) {
        const iv = encryptedData.slice(0, 16);
        const authTag = encryptedData.slice(16, 32);
        const encrypted = encryptedData.slice(32);
        const decipher = crypto.createDecipheriv('aes-256-gcm', this.encryptionKey, iv);
        decipher.setAuthTag(authTag);
        return Buffer.concat([decipher.update(encrypted), decipher.final()]);
    }

    handleError(error) {
        if (error.response) {
            return new Error(`SVDB Error: ${error.response.data.message || error.message}`);
        }
        return error;
    }
}

module.exports = SvdbClient; 