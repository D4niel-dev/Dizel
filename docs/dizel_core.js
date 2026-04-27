// dizel_core.js - Client-Side Engine for Dizel SPA

const DB_NAME = 'DizelDB';
const DB_VERSION = 1;

/** 
 * IndexedDB Minimal Async Wrapper
 */
class DBManager {
    static async connect() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('settings')) {
                    db.createObjectStore('settings');
                }
                if (!db.objectStoreNames.contains('history')) {
                    db.createObjectStore('history', { keyPath: 'id' });
                }
            };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    static async getSetting(key, defaultValue = null) {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('settings', 'readonly');
            const store = tx.objectStore('settings');
            const req = store.get(key);
            req.onsuccess = () => resolve(req.result !== undefined ? req.result : defaultValue);
            req.onerror = () => resolve(defaultValue);
        });
    }

    static async setSetting(key, value) {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('settings', 'readwrite');
            tx.objectStore('settings').put(value, key);
            tx.oncomplete = () => resolve(true);
        });
    }

    static async saveSession(sessionObj) {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('history', 'readwrite');
            tx.objectStore('history').put(sessionObj);
            tx.oncomplete = () => resolve(true);
        });
    }

    static async loadAllSessions() {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('history', 'readonly');
            const req = tx.objectStore('history').getAll();
            req.onsuccess = () => resolve(req.result || []);
        });
    }
}

/**
 * Global Configuration Cache
 */
const Config = {
    apiKey: '',
    apiProvider: 'openai',
    localOllamaUrl: 'http://localhost:11434',
    activeModel: 'llama3', // default
    targetBackend: 'ollama' // 'ollama' or 'api'
};

/**
 * Provider Routing Engine
 */
class Providers {
    static async streamOllama(prompt, onToken) {
        let fullText = "";
        try {
            const response = await fetch(`${Config.localOllamaUrl}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: Config.activeModel,
                    prompt: prompt,
                    stream: true
                })
            });

            if (!response.ok) throw new Error("Ollama connection failed. Is it running with CORS enabled?");

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while(true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(l => l.trim() !== '');
                for(let line of lines) {
                    const data = JSON.parse(line);
                    fullText += data.response;
                    onToken(data.response, fullText);
                }
            }
        } catch (e) {
            onToken(`\n\n**Error:** ${e.message}`, fullText);
        }
        return fullText;
    }

    static async streamOpenAILike(prompt, onToken) {
        let fullText = "";
        try {
            const response = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${Config.apiKey}`
                },
                body: JSON.stringify({
                    model: Config.activeModel,
                    messages: [{role: "user", content: prompt}],
                    stream: true
                })
            });

            if (!response.ok) throw new Error("API request failed. Check your API Keys.");
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.startsWith('data: '));
                
                for(let line of lines) {
                    const dataStr = line.replace('data: ', '').trim();
                    if(dataStr === '[DONE]') return fullText;
                    
                    const data = JSON.parse(dataStr);
                    const token = data.choices[0]?.delta?.content || "";
                    fullText += token;
                    onToken(token, fullText);
                }
            }
        } catch (e) {
            onToken(`\n\n**Error:** ${e.message}`, fullText);
        }
        return fullText;
    }
}

/**
 * Main Inference Interface
 */
class LLMEngine {
    static async init() {
        // Hydrate config from DB
        Config.apiKey = await DBManager.getSetting('apiKey', '');
        Config.targetBackend = await DBManager.getSetting('targetBackend', 'ollama');
        Config.localOllamaUrl = await DBManager.getSetting('localOllamaUrl', 'http://localhost:11434');
        Config.activeModel = await DBManager.getSetting('activeModel', 'llama3');
    }

    static async generateStream(prompt, onTokenCallback) {
        if (Config.targetBackend === 'ollama') {
            return await Providers.streamOllama(prompt, onTokenCallback);
        } else {
            if(!Config.apiKey) {
                onTokenCallback("\n\n**Error:** No API Key configured. Please open Settings and define your Provider key.", "");
                return;
            }
            return await Providers.streamOpenAILike(prompt, onTokenCallback);
        }
    }
}

// Bootstrap Engine on Load
window.addEventListener('DOMContentLoaded', () => {
    LLMEngine.init();
});

// Attach to global window
window.DBManager = DBManager;
window.LLMEngine = LLMEngine;
window.Config = Config;
