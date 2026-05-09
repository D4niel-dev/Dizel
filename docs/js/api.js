/**
 * docs/js/api.js
 * API Client for Dizel Serverless backend
 */

const API_BASE_URL = "http://localhost:8000/api";

const ApiClient = {
    async getHealth() {
        try {
            const res = await fetch(`${API_BASE_URL}/health`);
            return await res.json();
        } catch (e) {
            console.error("Health check failed:", e);
            return null;
        }
    },

    async getConfig() {
        try {
            const res = await fetch(`${API_BASE_URL}/config/`);
            return await res.json();
        } catch (e) {
            console.error("Failed to get config:", e);
            return null;
        }
    },

    async updateConfig(updates) {
        try {
            const res = await fetch(`${API_BASE_URL}/config/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(updates)
            });
            return await res.json();
        } catch (e) {
            console.error("Failed to update config:", e);
            return null;
        }
    },

    async listSessions() {
        try {
            const res = await fetch(`${API_BASE_URL}/session/list`);
            return await res.json();
        } catch (e) {
            console.error("Failed to list sessions:", e);
            return [];
        }
    },

    async getSession(sessionId) {
        try {
            const res = await fetch(`${API_BASE_URL}/session/${sessionId}`);
            if (!res.ok) throw new Error("Session not found");
            return await res.json();
        } catch (e) {
            console.error("Failed to get session:", e);
            return null;
        }
    },

    async switchProvider() {
        try {
            const res = await fetch(`${API_BASE_URL}/chat/switch_provider`, {
                method: "POST"
            });
            return await res.json();
        } catch (e) {
            console.error("Failed to switch provider:", e);
            return null;
        }
    },

    async applyProfile(modelName, modeName) {
        try {
            const res = await fetch(`${API_BASE_URL}/chat/apply_profile`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: modelName, mode_name: modeName })
            });
            return await res.json();
        } catch (e) {
            console.error("Failed to apply profile:", e);
            return null;
        }
    }
};

window.ApiClient = ApiClient;
