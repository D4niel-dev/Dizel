// Initialize Lucide Icons
lucide.createIcons();

// --- DOM References ---
const chatInput = document.getElementById('chat-input');
const chatStream = document.getElementById('chat-stream');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const welcomeScreen = document.querySelector('.welcome-screen');

// --- Global Tooltip Engine ---
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
window.togglePopover = function(popoverId) {
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
};

document.addEventListener('click', (e) => {
    if (!e.target.closest('.popover-wrapper')) {
        document.querySelectorAll('.popover-menu').forEach(el => el.classList.add('hidden'));
    }
});

// --- Attach Document Mock ---
window.attachDocumentMock = function() {
    const panel = document.querySelector('.input-panel');
    const existing = document.querySelector('.attachment-pill');
    if (!existing) {
        const pill = document.createElement('div');
        pill.className = 'attachment-pill';
        pill.innerHTML = `<i data-lucide="file-code"></i> <span>project_data.json</span> <button class="icon-btn xs" onclick="this.parentElement.remove()" style="margin-left:4px"><i data-lucide="x"></i></button>`;
        panel.insertBefore(pill, chatInput);
        lucide.createIcons({ root: pill });
    }
    document.getElementById('attach-popover')?.classList.add('hidden');
};

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

function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
        chatStream.style.justifyContent = 'flex-start';
    }

    createBubble('user', text);
    
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

function simulateAIResponse(userQuery) {
    const loadingBubble = createBubble('assistant', `<div class="typing-dots"><span></span><span></span><span></span></div>`);
    chatStream.scrollTop = chatStream.scrollHeight;
    setTimeout(() => {
        loadingBubble.remove();
        const actualBubble = createBubble('assistant', '...');
        const body = actualBubble.querySelector('.message-body');
        
        // Command execution intercepts (e.g. running a clear)
        let fakeStream = `<think>\nAnalyzing the requested input...\nI am simulating a response in the standalone Web SPA.</think>\nSure, I can help you with that!`;
        
        let currentIndex = 0; let accumulatedText = "";
        const interval = setInterval(() => {
            if (currentIndex < fakeStream.length) {
                accumulatedText += fakeStream.charAt(currentIndex);
                body.innerHTML = parseAndFormatThought(accumulatedText);
                lucide.createIcons({ root: body }); 
                chatStream.scrollTop = chatStream.scrollHeight;
                currentIndex++;
            } else {
                clearInterval(interval);
            }
        }, 15);
    }, 1200);
}


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
window.toggleSecondarySidebar = function() {
    const sidebar = document.getElementById('secondary-sidebar');
    const mainView = document.getElementById('main-view');
    
    sidebar.classList.toggle('open');
    if(sidebar.classList.contains('open')) {
        // Shift main view slightly on desktop
        if(window.innerWidth > 900) {
            mainView.style.marginRight = '320px';
        }
    } else {
        mainView.style.marginRight = '0';
    }
};
