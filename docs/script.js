// Initialize Lucide Icons
lucide.createIcons();

// --- DOM References ---
const chatInput = document.getElementById('chat-input');
const chatStream = document.getElementById('chat-stream');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const novaPill = document.getElementById('nova-pill');
const welcomeScreen = document.querySelector('.welcome-screen');

// --- Global Tooltip Engine ---
const globalTooltip = document.getElementById('global-tooltip');

document.addEventListener('mouseover', (e) => {
    const target = e.target.closest('.tooltip-target');
    if (target && target.dataset.tooltip) {
        const text = target.dataset.tooltip;
        globalTooltip.textContent = text;
        const rect = target.getBoundingClientRect();
        
        // Position tooltip centered above the element
        const tooltipX = rect.left + (rect.width / 2);
        const tooltipY = rect.top - 8;
        
        globalTooltip.style.left = tooltipX + 'px';
        globalTooltip.style.top = tooltipY + 'px';
        globalTooltip.style.transform = 'translate(-50%, -100%)';
        globalTooltip.classList.remove('hidden');
        
        // Slight delay for smooth fade in
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
    // Pick random on page load
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
    
    // Auto-close others
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
    document.getElementById('model-popover').classList.add('hidden');
    
    // Simulate Opening API Key modal if selected
    if (modelName === 'API Keys') {
        setTimeout(() => alert("Mocking the Dizel API Key Modal!"), 100);
    }
};

// Close popovers on click outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.popover-wrapper')) {
        document.querySelectorAll('.popover-menu').forEach(el => el.classList.add('hidden'));
    }
});

// Auto-resizing textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto'; 
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle 'Enter' to send
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
    
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatStream.scrollTop = chatStream.scrollHeight;

    // Simulate AI with Typing Indicator latency
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

    // Incorporating the Action Bar on assistant messages!
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
                    <span>Thought process (completed)</span>
                </button>
                <div class="thought-content hidden">
                    ${thoughtContent.replace(/\n/g, '<br>')}
                </div>
            </div>
            <div class="main-text">${mainContent.replace(/\n/g, '<br>')}</div>
        `;
    }
    return `<div class="main-text">${rawContent.replace(/\n/g, '<br>')}</div>`;
}

window.toggleThought = function(btn) {
    const container = btn.nextElementSibling;
    container.classList.toggle('hidden');
    chatStream.scrollTop = chatStream.scrollHeight;
};

// Deepseek Simulation Logic with Typing Dots!
function simulateAIResponse(userQuery) {
    // Inject Typing Indicator Bubble First
    const loadingBubble = createBubble('assistant', `
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
    `);
    
    chatStream.scrollTop = chatStream.scrollHeight;

    // Simulate Server Latency
    setTimeout(() => {
        // Remove typing dots bubble
        loadingBubble.remove();
        
        // Spawn actual streaming bubble
        const actualBubble = createBubble('assistant', '...');
        const body = actualBubble.querySelector('.message-body');
        
        let fakeStream = `<think>\nAnalyzing the user's intent: "${userQuery}"\n1. Process request\n2. Determine parameters\n3. Construct response\nI will supply a formatted response with code.</think>\nHere is the synthesized response to your request exactly as programmed!`;
        
        let currentIndex = 0;
        let accumulatedText = "";
        
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
    }, 1200); // 1.2 second fake latency
}

// Dictation "Nova" Simulation
micBtn.addEventListener('click', () => {
    micBtn.classList.toggle('recording');
    document.getElementById('nova-pill').classList.toggle('hidden');
});

document.getElementById('nova-cancel').addEventListener('click', () => {
    micBtn.classList.remove('recording');
    document.getElementById('nova-pill').classList.add('hidden');
});

document.getElementById('nova-done').addEventListener('click', () => {
    micBtn.classList.remove('recording');
    document.getElementById('nova-pill').classList.add('hidden');
    chatInput.value = "Dictated text from Nova system...";
    chatInput.style.height = 'auto'; 
});
