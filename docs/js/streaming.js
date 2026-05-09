/**
 * docs/js/streaming.js
 * Handles SSE streaming from the backend for chat responses.
 */

const StreamingClient = {
    activeSource: null,

    /**
     * Start a streaming chat request.
     * @param {string} userText 
     * @param {string} sessionId 
     * @param {Function} onToken Callback for each token received
     * @param {Function} onDone Callback when generation completes
     * @param {Function} onError Callback on error
     */
    streamChat(userText, sessionId, onToken, onDone, onError) {
        if (this.activeSource) {
            this.activeSource.close();
            this.activeSource = null;
        }

        const url = `${API_BASE_URL}/chat/stream`;
        
        // We use fetch directly since SSE via POST is not natively supported by EventSource without workarounds
        // Or we use a polyfill. The backend expects a POST to /stream.
        
        fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            body: JSON.stringify({
                user_text: userText,
                session_id: sessionId || null,
                attachments: []
            })
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            
            let buffer = "";

            function processStream({ done, value }) {
                if (done) {
                    onDone();
                    return;
                }
                
                buffer += decoder.decode(value, { stream: true });
                let lines = buffer.split('\n\n');
                
                // Keep the last incomplete part in the buffer
                buffer = lines.pop();
                
                for (let line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.substring(6);
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.error) {
                                onError(data.error);
                            } else if (data.token) {
                                onToken(data.token);
                            } else if (data.done) {
                                onDone();
                            }
                        } catch (e) {
                            console.error("Failed to parse SSE data:", dataStr);
                        }
                    }
                }
                
                return reader.read().then(processStream);
            }
            
            return reader.read().then(processStream);
        }).catch(err => {
            onError(err.message);
        });
    },
    
    stop() {
        // Implementation for aborting the fetch request if needed
    }
};

window.StreamingClient = StreamingClient;
