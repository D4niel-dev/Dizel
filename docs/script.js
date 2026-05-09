// Initialize Lucide Icons
lucide.createIcons();

// --- DOM References ---
const chatInput = document.getElementById('chat-input');
const chatStream = document.getElementById('chat-stream');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const welcomeScreen = document.querySelector('.welcome-screen');

// --- Global Tooltip Engine ---
window.addEventListener('DOMContentLoaded', () => {
    // Demo Splash Screen Check
    if(!localStorage.getItem('dizel_demo_seen')) {
        setTimeout(() => window.openModal('welcome-modal'), 250);
    }
});

window.acknowledgeDemo = function() {
    localStorage.setItem('dizel_demo_seen', 'true');
    closeAllModals();
};

const globalTooltip = document.getElementById('global-tooltip');

document.addEventListener('mouseover', (e) => {
    const target = e.target.closest('.tooltip-target');
    if (target && target.dataset.tooltip) {
        const text = target.dataset.tooltip;
        globalTooltip.textContent = text;
        const rect = target.getBoundingClientRect();
        const tooltipX = rect.left + (rect.width / 2);
        const tooltipY = rect.top - 8;
        
        globalTooltip.style.left = tooltipX + 'px';
        globalTooltip.style.top = tooltipY + 'px';
        globalTooltip.style.transform = 'translate(-50%, -100%)';
        globalTooltip.classList.remove('hidden');
        
        setTimeout(() => globalTooltip.classList.add('visible'), 10);
    }
});

document.addEventListener('mouseout', (e) => {
    const target = e.target.closest('.tooltip-target');
    if (target) {
        globalTooltip.classList.remove('visible');
        setTimeout(() => globalTooltip.classList.add('hidden'), 200);
    }
});

// --- Title Rotator ---
const titles = [
    "Beyond expectations, into reality.",
    "Dream big, build bigger.",
    "Innovate. Iterate. Implement.",
    "Let's architect the future.",
    "Your ideas. My execution.",
    "Ready to code."
];
const titleEl = document.getElementById('hero-title-text');
if (titleEl) {
    titleEl.textContent = titles[Math.floor(Math.random() * titles.length)];
}

// --- Action Pill Carousel Scroll ---
const pillsContainer = document.getElementById('pills-container');
document.getElementById('scroll-left')?.addEventListener('click', () => {
    pillsContainer.scrollBy({ left: -200, behavior: 'smooth' });
});
document.getElementById('scroll-right')?.addEventListener('click', () => {
    pillsContainer.scrollBy({ left: 200, behavior: 'smooth' });
});

// --- Popover Engine ---
window.toggleCustomPopover = function(popoverId) {
    const popover = document.getElementById(popoverId);
    if (!popover) return;
    
    document.querySelectorAll('.popover-menu').forEach(el => {
        if (el.id !== popoverId) el.classList.add('hidden');
    });
    popover.classList.toggle('hidden');
};

// Handle swapping Model Provider
window.selectModel = function(modelName) {
    const btn = document.querySelector('.model-select-btn');
    if (btn) {
        btn.innerHTML = `<i data-lucide="chevron-up"></i> ${modelName}`;
        lucide.createIcons({ root: btn });
    }
    document.getElementById('model-popover')?.classList.add('hidden');
    if(modelName === 'Ollama') {
        closeAllModals(); // Close modal on local select
    }
};

document.addEventListener('click', (e) => {
    if (!e.target.closest('.popover-wrapper')) {
        document.querySelectorAll('.popover-menu').forEach(el => el.classList.add('hidden'));
    }
});

// --- Attach Document Mock ---
window.attachDocumentMock = function(type = 'file') {
    const panel = document.querySelector('.input-panel');
    const existing = document.querySelector(`.attachment-pill[data-tool="${type}"]`);
    if (!existing) {
        const pill = document.createElement('div');
        pill.className = 'attachment-pill';
        pill.setAttribute('data-tool', type);
        let icon = 'file-code'; let label = 'project_data.json';
        if (type === 'web-search') { icon = 'globe'; label = 'Web Search'; }
        if (type === 'deep-think') { icon = 'brain'; label = 'Deep Think'; }
        if (type === 'parse') { icon = 'folder'; label = 'File Parser'; }
        if (type === 'image') { icon = 'image'; label = 'Image Generation'; }

        pill.innerHTML = `<i data-lucide="${icon}"></i> <span>${label}</span> <button class="icon-btn xs" onclick="this.parentElement.remove()" style="margin-left:4px"><i data-lucide="x"></i></button>`;
        panel.insertBefore(pill, chatInput);
        lucide.createIcons({ root: pill });
    }
    document.getElementById('attach-popover')?.classList.add('hidden');
};

// --- Web Speech API (Voice Engine Replacement) ---
// --- Web Speech API (Voice Engine Replacement) ---
const novaPill = document.getElementById('nova-pill');
const novaCancel = document.getElementById('nova-cancel');
const novaDone = document.getElementById('nova-done');
let recognition = null;

if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    
    recognition.onstart = function() {
        novaPill.classList.remove('hidden');
        micBtn.classList.add('recording-active'); // Add a glowing ring effect CSS
    };
    
    recognition.onresult = function(event) {
        let finalTranscripts = '';
        let interimTranscripts = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) finalTranscripts += transcript;
            else interimTranscripts += transcript;
        }
        chatInput.value = finalTranscripts + interimTranscripts;
        chatInput.style.height = 'auto'; 
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    };
    
    recognition.onerror = function(event) {
        console.error("Speech recognition error", event.error);
        stopDictation();
    };
    
    recognition.onend = function() {
        stopDictation();
    };
}

function stopDictation() {
    novaPill.classList.add('hidden');
    micBtn.classList.remove('recording-active');
    if(recognition) recognition.stop();
}

micBtn?.addEventListener('click', () => {
    if(!recognition) { alert("Speech Recognition not supported in this browser."); return; }
    if(novaPill.classList.contains('hidden')) recognition.start();
    else stopDictation();
});

novaCancel?.addEventListener('click', () => {
    chatInput.value = "";
    stopDictation();
});
novaDone?.addEventListener('click', () => {
    stopDictation();
    sendMessage();
});

// --- Chat Engine ---
chatInput.addEventListener('input', function() {
    this.style.height = 'auto'; 
    this.style.height = (this.scrollHeight) + 'px';
});

chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

let currentSessionId = null;
let currentSessionMessages = [];

function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
        chatStream.style.justifyContent = 'flex-start';
    }

    createBubble('user', text);
    currentSessionMessages.push({ role: 'user', content: text });
    
    // Clear attachments mock
    const pill = document.querySelector('.attachment-pill');
    if (pill) pill.remove();

    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatStream.scrollTop = chatStream.scrollHeight;

    simulateAIResponse(text);
}

function createBubble(role, content) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${role}`;
    const isUser = role === 'user';
    const roleLabel = isUser ? 'You' : 'Dizel';
    const avatarImg = isUser 
        ? "https://ui-avatars.com/api/?name=User&background=333&color=fff&rounded=true"
        : "assets/app/Dizel.png";

    bubble.innerHTML = `
        <div class="bubble-container">
            ${!isUser ? `
            <div class="role-header">
                <img src="${avatarImg}" alt="${roleLabel}" style="width: 20px; height: 20px; border-radius: 50%;"> 
                ${roleLabel}
            </div>` : ''}
            <div class="message-body">${content}</div>
            ${!isUser ? `
            <div class="bubble-action-bar">
                <button class="icon-btn xs tooltip-target" data-tooltip="Copy" onclick="copyMock(this)"><i data-lucide="copy"></i></button>
                <button class="icon-btn xs tooltip-target" data-tooltip="Good response"><i data-lucide="thumbs-up"></i></button>
                <button class="icon-btn xs tooltip-target" data-tooltip="Bad response"><i data-lucide="thumbs-down"></i></button>
                <button class="icon-btn xs tooltip-target" data-tooltip="Regenerate"><i data-lucide="refresh-cw"></i></button>
            </div>` : ''}
        </div>
    `;
    
    chatStream.appendChild(bubble);
    lucide.createIcons({ root: bubble });
    return bubble;
}

window.copyMock = function(btn) {
    const icon = btn.querySelector('i');
    icon.setAttribute('data-lucide', 'check');
    lucide.createIcons({ root: btn });
    setTimeout(() => {
        icon.setAttribute('data-lucide', 'copy');
        lucide.createIcons({ root: btn });
    }, 2000);
}

function parseAndFormatThought(rawContent) {
    const thinkRegex = /<think>([\s\S]*?)(<\/think>|$)/;
    const match = rawContent.match(thinkRegex);
    if (match) {
        const thoughtContent = match[1].trim();
        const mainContent = rawContent.replace(thinkRegex, '').trim();
        return `
            <div class="thought-widget">
                <button class="thought-toggle" onclick="toggleThought(this)">
                    <i data-lucide="cpu" style="width: 14px; height: 14px;"></i> 
                    <span>Thought process</span>
                </button>
                <div class="thought-content hidden">${thoughtContent.replace(/\n/g, '<br>')}</div>
            </div>
            <div class="main-text">${mainContent.replace(/\n/g, '<br>')}</div>
        `;
    }
    return `<div class="main-text">${rawContent.replace(/\n/g, '<br>')}</div>`;
}

window.toggleThought = function(btn) {
    btn.nextElementSibling.classList.toggle('hidden');
    chatStream.scrollTop = chatStream.scrollHeight;
};

let currentAbortController = null;

function setGeneratingState(isGenerating) {
    const sendBtn = document.getElementById('send-btn');
    if (!sendBtn) return;
    const icon = sendBtn.querySelector('i');
    if (isGenerating) {
        icon.setAttribute('data-lucide', 'square');
        sendBtn.onclick = stopGeneration;
        sendBtn.style.color = 'var(--error)';
    } else {
        icon.setAttribute('data-lucide', 'arrow-up');
        sendBtn.onclick = sendMessage;
        sendBtn.style.color = '';
    }
    lucide.createIcons({ root: sendBtn });
}

function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
}

// Initial setup
document.getElementById('send-btn').onclick = sendMessage;

function simulateAIResponse(userQuery) {
    const loadingBubble = createBubble('assistant', `<div class="typing-dots"><span></span><span></span><span></span></div>`);
    chatStream.scrollTop = chatStream.scrollHeight;
    
    let actualBubble = null;
    let bodyNode = null;
    let isFirstToken = true;

    setGeneratingState(true);

    StreamingClient.streamChat(
        userQuery, 
        currentSessionId,
        (token) => {
            if (isFirstToken) {
                isFirstToken = false;
                loadingBubble.remove();
                actualBubble = createBubble('assistant', '');
                bodyNode = actualBubble.querySelector('.message-body');
            }
            // Append token to the UI
            bodyNode.innerHTML += token.replace(/\\n/g, '<br>');
            chatStream.scrollTop = chatStream.scrollHeight;
        },
        async () => {
            setGeneratingState(false);
            const finalText = bodyNode ? bodyNode.innerText : "*Empty Response*";
            currentSessionMessages.push({ role: 'assistant', content: finalText });

            if (!currentSessionId) {
                // Actually the backend might have created a new session ID if we passed null,
                // but for now we'll just refresh history. 
                // A better approach is to get the session_id from the backend, 
                // but the backend saves it internally. We'll reload sessions later.
                const pad = (n) => n.toString().padStart(2, '0');
                const d = new Date();
                currentSessionId = `session_${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
            }
            refreshHistoryUI();
        },
        (errorMsg) => {
            setGeneratingState(false);
            if(loadingBubble.parentElement) loadingBubble.remove();
            createBubble('assistant', `<span style="color:var(--error);"><i data-lucide="alert-triangle"></i> Error generating response: ${errorMsg}</span>`);
        }
    );
}

// History UI Engine
async function refreshHistoryUI(query = '') {
    const list = document.getElementById('history-list');
    if(!list) return;
    
    let sessions = await ApiClient.listSessions();
    if(query) {
        sessions = sessions.filter(s => (s.title || '').toLowerCase().includes(query.toLowerCase()));
    }
    
    list.innerHTML = '';
    
    if(sessions.length === 0) {
        list.innerHTML = `<div class="dim-text" style="text-align:center; padding: 20px;">No chats found.</div>`;
        return;
    }
    
    sessions.forEach(s => {
        const item = document.createElement('div');
        item.className = 'info-card history-item hover-bg-transition';
        item.style.cursor = 'pointer';
        item.style.border = currentSessionId === s.id ? '1px solid var(--border-focus)' : '1px solid var(--border)';
        
        const pinColor = s.pinned ? 'color: var(--accent);' : 'color: var(--text-dim);';
        
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                <h4 style="font-size: 13px; font-weight: 500; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;">${s.title}</h4>
                <div style="display:flex; gap: 4px;">
                    <button class="icon-btn xs" onclick="event.stopPropagation(); togglePin('${s.id}')"><i data-lucide="pin" style="${pinColor}"></i></button>
                    <button class="icon-btn xs" onclick="event.stopPropagation(); deleteSession('${s.id}')"><i data-lucide="trash-2"></i></button>
                </div>
            </div>
            <div class="dim-text" style="font-size: 11px;">${new Date(s.created).toLocaleDateString()} ${new Date(s.created).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
        `;
        item.onclick = () => loadSession(s.id);
        list.appendChild(item);
    });
    lucide.createIcons({ root: list });
}

window.newChat = function() {
    currentSessionId = null;
    currentSessionMessages = [];
    document.querySelectorAll('.message-bubble').forEach(b => b.remove());
    if (welcomeScreen) {
        welcomeScreen.style.display = 'flex';
        chatStream.style.justifyContent = 'center';
    }
    refreshHistoryUI();
};

window.loadSession = async function(id) {
    const session = await ApiClient.getSession(id);
    if(!session) return;
    
    currentSessionId = session.id;
    currentSessionMessages = session.messages || [];
    
    document.querySelectorAll('.message-bubble').forEach(b => b.remove());
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    chatStream.style.justifyContent = 'flex-start';
    
    for(const m of currentSessionMessages) {
        if(m.role === 'user') createBubble('user', m.content);
        else {
            const b = createBubble('assistant', '');
            const body = b.querySelector('.message-body');
            body.innerHTML = parseAndFormatThought(m.content);
            lucide.createIcons({ root: body }); 
        }
    }
    chatStream.scrollTop = chatStream.scrollHeight;
    refreshHistoryUI();
    if(window.innerWidth < 900) toggleSecondarySidebar(); // auto close on mobile
};

window.deleteSession = async function(id) {
    if(confirm("Are you sure you want to delete this chat?")) {
        await fetch(`${API_BASE_URL}/session/${id}`, { method: 'DELETE' });
        if(currentSessionId === id) window.newChat();
        else refreshHistoryUI();
    }
};

window.togglePin = async function(id) {
    await fetch(`${API_BASE_URL}/session/${id}/pin`, { method: 'POST' });
    refreshHistoryUI();
};

document.getElementById('history-search')?.addEventListener('input', (e) => {
    refreshHistoryUI(e.target.value);
});

// Run refresh on load
window.addEventListener('DOMContentLoaded', () => { setTimeout(() => refreshHistoryUI(), 500); });


// --- Global Modal Engine & Ctrl+K Palette ---

const backdrop = document.getElementById('modal-backdrop');

window.openModal = function(modalId) {
    // Hide all internal panels first
    document.querySelectorAll('.modal-box').forEach(box => box.classList.add('hidden'));
    
    // Show target
    const target = document.getElementById(modalId);
    if (target) {
        target.classList.remove('hidden');
        backdrop.classList.remove('hidden');
        
        // Auto-focus logic for command palette
        if(modalId === 'command-palette') {
            document.getElementById('cmd-input')?.focus();
        } else if (modalId === 'settings-modal') {
            hydrateSettingsModal();
        }
    }
};

window.closeAllModals = function() {
    backdrop.classList.add('hidden');
};

window.toggleCommandPalette = function() {
    if(backdrop.classList.contains('hidden')){
        openModal('command-palette');
    } else {
        closeAllModals();
    }
}

// Global Hotkeys Listener
document.addEventListener('keydown', (e) => {
    // Escape to close modals
    if (e.key === 'Escape') {
        closeAllModals();
        document.getElementById('secondary-sidebar').classList.remove('open');
    }
    
    // Command Palette: Ctrl+K or Cmd+K
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        toggleCommandPalette();
    }
});

// --- Command Execution ---
window.executeCommand = function(cmd) {
    closeAllModals();
    if(cmd === 'clear') {
        // Clear chat, reshow welcome screen
        document.querySelectorAll('.message-bubble').forEach(b => b.remove());
        if(welcomeScreen) {
            welcomeScreen.style.display = 'flex';
            chatStream.style.justifyContent = 'center';
        }
    } else if(cmd === 'settings') {
        openModal('settings-modal');
    } else if(cmd === 'theme') {
        alert("Simulating standard Theme Toggle!");
    }
};

window.switchThemeTab = function(tabId, btnElement) {
    // Un-active tabs
    document.querySelectorAll('.settings-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
    btnElement.classList.add('active');
    
    // Hide all panes
    document.querySelectorAll('.settings-panes .pane').forEach(p => p.classList.add('hidden'));
    
    // Show specific pane
    document.getElementById(tabId).classList.remove('hidden');
};

// --- Secondary Sidebar Engine ---
window.togglePrimarySidebar = function() {
    document.getElementById('sidebar').classList.toggle('expanded');
};

window.toggleSecondarySidebar = function() {
    const sidebar = document.getElementById('secondary-sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    const isOpen = sidebar.classList.toggle('open');
    
    // Toggle backdrop only on mobile (when window width <= 768)
    if (window.innerWidth <= 768 && backdrop) {
        if (isOpen) {
            backdrop.classList.remove('hidden');
            // Small delay to allow display block to apply before opacity transition
            setTimeout(() => backdrop.classList.add('visible'), 10);
        } else {
            backdrop.classList.remove('visible');
            setTimeout(() => backdrop.classList.add('hidden'), 300);
        }
    }
};

// Also close sidebar on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAllModals();
        const sidebar = document.getElementById('secondary-sidebar');
        const backdrop = document.getElementById('sidebar-backdrop');
        if (sidebar && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
            if (backdrop) {
                backdrop.classList.remove('visible');
                setTimeout(() => backdrop.classList.add('hidden'), 300);
            }
        }
    }
});

// --- Settings Bindings ---
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const topkSlider = document.getElementById('topk-slider');
const topkVal = document.getElementById('topk-val');
const toppSlider = document.getElementById('topp-slider');
const toppVal = document.getElementById('topp-val');
const repSlider = document.getElementById('rep-slider');
const repVal = document.getElementById('rep-val');
const tokensSlider = document.getElementById('tokens-slider');
const tokensVal = document.getElementById('tokens-val');

if (tempSlider && tempVal) tempSlider.addEventListener('input', (e) => tempVal.innerText = (e.target.value / 100).toFixed(2));
if (topkSlider && topkVal) topkSlider.addEventListener('input', (e) => topkVal.innerText = e.target.value);
if (toppSlider && toppVal) toppSlider.addEventListener('input', (e) => toppVal.innerText = (e.target.value / 100).toFixed(2));
if (repSlider && repVal) repSlider.addEventListener('input', (e) => repVal.innerText = (e.target.value / 100).toFixed(2));
if (tokensSlider && tokensVal) tokensSlider.addEventListener('input', (e) => tokensVal.innerText = e.target.value);

let selectedProvider = 'ollama';

window.selectProviderSettings = function(provider) {
    selectedProvider = provider;
    document.querySelectorAll('.provider-card').forEach(c => c.classList.remove('active'));
    document.querySelector(`.provider-card[data-provider="${provider}"]`)?.classList.add('active');
    
    // UI logic
    const apiKeyRow = document.getElementById('api-key-row');
    const ollamaUrlRow = document.getElementById('ollama-url-row');
    
    if (provider === 'ollama') {
        apiKeyRow.style.display = 'none';
        ollamaUrlRow.style.display = 'flex';
    } else {
        apiKeyRow.style.display = 'flex';
        ollamaUrlRow.style.display = 'none';
    }
};

window.testConnection = async function() {
    alert("Connection verified via fetch! (Mocked)");
};

async function hydrateSettingsModal() {
    const config = await ApiClient.getConfig() || {};
    
    document.getElementById('system-prompt').value = config.system_prompt || '';
    
    // API router settings
    const api_router = config.api_router || {};
    document.getElementById('model-name').value = api_router.model || '';
    const backend = api_router.provider || 'local';
    selectProviderSettings(backend);
    
    document.getElementById('ollama-url').value = api_router.ollama_url || '';
    document.getElementById('api-key').value = api_router.api_key || ''; // Note: normally wouldn't send key back to client in plain text, but this is a demo.

    // Sampling
    const sampling = config.sampling || {};
    if (tempSlider) { tempSlider.value = (sampling.temperature || 0.7) * 100; tempVal.innerText = (sampling.temperature || 0.7).toFixed(2); }
    if (topkSlider) { topkSlider.value = sampling.top_k || 40; topkVal.innerText = sampling.top_k || 40; }
    if (toppSlider) { toppSlider.value = (sampling.top_p || 0.9) * 100; toppVal.innerText = (sampling.top_p || 0.9).toFixed(2); }
    if (repSlider) { repSlider.value = (sampling.repetition_penalty || 1.1) * 100; repVal.innerText = (sampling.repetition_penalty || 1.1).toFixed(2); }
    if (tokensSlider) { tokensSlider.value = sampling.max_new_tokens || 400; tokensVal.innerText = sampling.max_new_tokens || 400; }
}

window.saveSettings = async function() {
    const updates = {
        system_prompt: document.getElementById('system-prompt')?.value || '',
        api_router: {
            provider: selectedProvider,
            model: document.getElementById('model-name')?.value || ''
        },
        sampling: {
            temperature: tempSlider ? parseInt(tempSlider.value) / 100 : 0.7,
            top_k: topkSlider ? parseInt(topkSlider.value) : 40,
            top_p: toppSlider ? parseInt(toppSlider.value) / 100 : 0.9,
            repetition_penalty: repSlider ? parseInt(repSlider.value) / 100 : 1.1,
            max_new_tokens: tokensSlider ? parseInt(tokensSlider.value) : 400
        }
    };
    
    if (selectedProvider === 'ollama') {
        updates.api_router.ollama_url = document.getElementById('ollama-url').value;
    } else if (selectedProvider !== 'local') {
        updates.api_router.api_key = document.getElementById('api-key').value;
    }

    await ApiClient.updateConfig(updates);
    await ApiClient.switchProvider(); // Apply the new provider
    
    closeAllModals();
};
