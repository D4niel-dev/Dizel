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

    // --- History CRUD ---
    static async saveSession(sessionObj) {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('history', 'readwrite');
            tx.objectStore('history').put(sessionObj);
            tx.oncomplete = () => resolve(true);
        });
    }

    static async loadSession(id) {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('history', 'readonly');
            const req = tx.objectStore('history').get(id);
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => resolve(null);
        });
    }

    static async listSessions() {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('history', 'readonly');
            const req = tx.objectStore('history').getAll();
            req.onsuccess = () => {
                const sessions = req.result || [];
                // Sort by pinned + date desc
                sessions.sort((a, b) => {
                    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
                    return new Date(b.created).getTime() - new Date(a.created).getTime();
                });
                resolve(sessions);
            };
        });
    }

    static async deleteSession(id) {
        const db = await this.connect();
        return new Promise((resolve) => {
            const tx = db.transaction('history', 'readwrite');
            tx.objectStore('history').delete(id);
            tx.oncomplete = () => resolve(true);
        });
    }

    static async togglePin(id) {
        const session = await this.loadSession(id);
        if (session) {
            session.pinned = !session.pinned;
            await this.saveSession(session);
            return session.pinned;
        }
        return false;
    }

    static async searchSessions(query) {
        const sessions = await this.listSessions();
        if(!query) return sessions;
        const q = query.toLowerCase();
        return sessions.filter(s => s.title && s.title.toLowerCase().includes(q));
    }
}

/**
 * Web Crypto API Vault for API Keys
 */
class CryptoVault {
    static async getDerivedKey() {
        const enc = new TextEncoder();
        const keyMaterial = await crypto.subtle.importKey(
            "raw",
            enc.encode("dizel_demo_vault_secure_password_salt_xyz"),
            { name: "PBKDF2" },
            false,
            ["deriveBits", "deriveKey"]
        );
        return await crypto.subtle.deriveKey(
            {
                name: "PBKDF2",
                salt: enc.encode("static_salt_for_demo_purposes_only"),
                iterations: 100000,
                hash: "SHA-256"
            },
            keyMaterial,
            { name: "AES-GCM", length: 256 },
            true,
            ["encrypt", "decrypt"]
        );
    }
    
    static async encrypt(text) {
        if (!text) return null;
        const key = await this.getDerivedKey();
        const iv = crypto.getRandomValues(new Uint8Array(12));
        const enc = new TextEncoder();
        const encrypted = await crypto.subtle.encrypt(
            { name: "AES-GCM", iv: iv },
            key,
            enc.encode(text)
        );
        return {
            iv: Array.from(iv),
            data: Array.from(new Uint8Array(encrypted))
        };
    }
    
    static async decrypt(blob) {
        if (!blob || !blob.iv || !blob.data) return '';
        try {
            const key = await this.getDerivedKey();
            const iv = new Uint8Array(blob.iv);
            const data = new Uint8Array(blob.data);
            const decrypted = await crypto.subtle.decrypt(
                { name: "AES-GCM", iv: iv },
                key,
                data
            );
            const dec = new TextDecoder();
            return dec.decode(decrypted);
        } catch(e) {
            console.error("Decryption failed", e);
            return '';
        }
    }
}

/**
 * Global Configuration Cache
 */
const Config = {
    apiKey: '',
    anthropicKey: '',
    googleKey: '',
    groqKey: '',
    localOllamaUrl: 'http://localhost:11434',
    activeModel: 'llama3', // default
    targetBackend: 'ollama', // 'ollama', 'openai', 'anthropic', 'google', 'groq'
    systemPrompt: '',
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repPenalty: 1.1,
    maxTokens: 400
};

/**
 * Provider Routing Engine
 */
class Providers {
    static _buildFullPrompt(prompt, messagesHistory = []) {
        // Map abstract history messages to format
        const msgs = [];
        if (Config.systemPrompt) {
            msgs.push({ role: "system", content: Config.systemPrompt });
        }
        for (const m of messagesHistory) {
            msgs.push(m);
        }
        msgs.push({ role: "user", content: prompt });
        return msgs;
    }

    static async streamOllama(prompt, onToken, signal, messagesHistory = []) {
        let fullText = "";
        try {
            // Setup full payload
            const msgs = this._buildFullPrompt(prompt, messagesHistory);
            
            const response = await fetch(`${Config.localOllamaUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: Config.activeModel,
                    messages: msgs,
                    options: {
                        temperature: Config.temperature,
                        top_k: Config.topK,
                        top_p: Config.topP,
                        repeat_penalty: Config.repPenalty,
                        num_predict: Config.maxTokens
                    },
                    stream: true
                }),
                signal: signal
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
                    const token = data.message?.content || "";
                    fullText += token;
                    onToken(token, fullText);
                }
            }
        } catch (e) {
            if(e.name === 'AbortError') throw e;
            onToken(`\n\n**Error:** ${e.message}`, fullText);
        }
        return fullText;
    }

    static async streamOpenAILike(prompt, onToken, signal, messagesHistory = []) {
        let fullText = "";
        try {
            const msgs = this._buildFullPrompt(prompt, messagesHistory);
            const response = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${Config.apiKey}`
                },
                body: JSON.stringify({
                    model: Config.activeModel,
                    messages: msgs,
                    temperature: Config.temperature,
                    top_p: Config.topP,
                    max_tokens: Config.maxTokens,
                    frequency_penalty: Config.repPenalty > 1.0 ? (Config.repPenalty - 1.0) : 0,
                    stream: true
                }),
                signal: signal
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
            if(e.name === 'AbortError') throw e;
            onToken(`\n\n**Error:** ${e.message}`, fullText);
        }
        return fullText;
    }

    static async streamGroq(prompt, onToken, signal, messagesHistory = []) {
        let fullText = "";
        try {
            const msgs = this._buildFullPrompt(prompt, messagesHistory);
            const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${Config.groqKey}`
                },
                body: JSON.stringify({
                    model: Config.activeModel || "llama3-8b-8192",
                    messages: msgs,
                    temperature: Config.temperature,
                    top_p: Config.topP,
                    max_tokens: Config.maxTokens,
                    stream: true
                }),
                signal: signal
            });

            if (!response.ok) throw new Error("Groq API request failed. Check your API Keys.");
            
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
            if(e.name === 'AbortError') throw e;
            onToken(`\n\n**Error:** ${e.message}`, fullText);
        }
        return fullText;
    }

    static async streamAnthropic(prompt, onToken, signal, messagesHistory = []) {
        let fullText = "";
        try {
            const msgs = [];
            for (const m of messagesHistory) {
                msgs.push(m);
            }
            msgs.push({ role: "user", content: prompt });
            
            const response = await fetch('https://api.anthropic.com/v1/messages', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'x-api-key': Config.anthropicKey,
                    'anthropic-version': '2023-06-01',
                    'anthropic-dangerous-direct-browser-access': 'true' // For frontend demo only
                },
                body: JSON.stringify({
                    model: Config.activeModel || "claude-3-haiku-20240307",
                    system: Config.systemPrompt || undefined,
                    messages: msgs,
                    temperature: Config.temperature,
                    max_tokens: Config.maxTokens,
                    stream: true
                }),
                signal: signal
            });

            if (!response.ok) throw new Error("Anthropic API request failed. Check your API Key.");
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                let lines = buffer.split('\n');
                buffer = lines.pop(); // keep last incomplete line
                
                for(let line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '').trim();
                        if(!dataStr) continue;
                        const data = JSON.parse(dataStr);
                        if (data.type === 'content_block_delta') {
                            const token = data.delta.text || "";
                            fullText += token;
                            onToken(token, fullText);
                        }
                    }
                }
            }
        } catch (e) {
            if(e.name === 'AbortError') throw e;
            onToken(`\n\n**Error:** ${e.message}`, fullText);
        }
        return fullText;
    }

    static async streamGoogle(prompt, onToken, signal, messagesHistory = []) {
        let fullText = "";
        try {
            // Convert to Gemini format
            const contents = [];
            if (Config.systemPrompt) {
                contents.push({ role: "user", parts: [{ text: "System Prompt: " + Config.systemPrompt }]});
                contents.push({ role: "model", parts: [{ text: "Understood." }]});
            }
            
            for (const m of messagesHistory) {
                contents.push({
                    role: m.role === 'assistant' ? 'model' : m.role,
                    parts: [{ text: m.content }]
                });
            }
            contents.push({ role: "user", parts: [{ text: prompt }]});
            
            const modelName = Config.activeModel || "gemini-1.5-flash-latest";
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:streamGenerateContent?alt=sse&key=${Config.googleKey}`;

            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: contents,
                    generationConfig: {
                        temperature: Config.temperature,
                        topK: Config.topK,
                        topP: Config.topP,
                        maxOutputTokens: Config.maxTokens
                    }
                }),
                signal: signal
            });

            if (!response.ok) throw new Error("Google API request failed. Check your API Key.");
            
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
                    if(!dataStr) continue;
                    
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.candidates && data.candidates[0].content && data.candidates[0].content.parts) {
                            const token = data.candidates[0].content.parts[0].text || "";
                            fullText += token;
                            onToken(token, fullText);
                        }
                    } catch(e) {}
                }
            }
        } catch (e) {
            if(e.name === 'AbortError') throw e;
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
        Config.apiKey = await CryptoVault.decrypt(await DBManager.getSetting('apiKey'));
        Config.anthropicKey = await CryptoVault.decrypt(await DBManager.getSetting('anthropicKey'));
        Config.googleKey = await CryptoVault.decrypt(await DBManager.getSetting('googleKey'));
        Config.groqKey = await CryptoVault.decrypt(await DBManager.getSetting('groqKey'));
        
        Config.targetBackend = await DBManager.getSetting('targetBackend', 'ollama');
        Config.localOllamaUrl = await DBManager.getSetting('localOllamaUrl', 'http://localhost:11434');
        Config.activeModel = await DBManager.getSetting('activeModel', 'llama3');
        Config.systemPrompt = await DBManager.getSetting('systemPrompt', '');
        Config.temperature = await DBManager.getSetting('temperature', 0.7);
        Config.topK = await DBManager.getSetting('topK', 40);
        Config.topP = await DBManager.getSetting('topP', 0.9);
        Config.repPenalty = await DBManager.getSetting('repPenalty', 1.1);
        Config.maxTokens = await DBManager.getSetting('maxTokens', 400);
    }

    static async generateStream(prompt, onTokenCallback, signal, messagesHistory = []) {
        try {
            if (Config.targetBackend === 'ollama') {
                return await Providers.streamOllama(prompt, onTokenCallback, signal, messagesHistory);
            } else if (Config.targetBackend === 'openai') {
                if(!Config.apiKey) throw new Error("No OpenAI API Key configured.");
                return await Providers.streamOpenAILike(prompt, onTokenCallback, signal, messagesHistory);
            } else if (Config.targetBackend === 'groq') {
                if(!Config.groqKey) throw new Error("No Groq API Key configured.");
                return await Providers.streamGroq(prompt, onTokenCallback, signal, messagesHistory);
            } else if (Config.targetBackend === 'anthropic') {
                if(!Config.anthropicKey) throw new Error("No Anthropic API Key configured.");
                return await Providers.streamAnthropic(prompt, onTokenCallback, signal, messagesHistory);
            } else if (Config.targetBackend === 'google') {
                if(!Config.googleKey) throw new Error("No Google API Key configured.");
                return await Providers.streamGoogle(prompt, onTokenCallback, signal, messagesHistory);
            } else {
                throw new Error("Unknown provider: " + Config.targetBackend);
            }
        } catch(e) {
            if(e.name === 'AbortError') return;
            onTokenCallback(`\n\n**Error:** ${e.message}`, "");
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
window.CryptoVault = CryptoVault;
