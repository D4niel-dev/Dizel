// Initialize Lucide Icons
lucide.createIcons();

// --- DOM References ---
const chatInput = document.getElementById('chat-input');
const chatStream = document.getElementById('chat-stream');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const novaPill = document.getElementById('nova-pill');
const novaCancel = document.getElementById('nova-cancel');
const novaDone = document.getElementById('nova-done');
const welcomeScreen = document.querySelector('.welcome-screen');

// Auto-resizing textarea to match _AutoResizingTextEdit
chatInput.addEventListener('input', function() {
    this.style.height = 'auto'; // Reset height
    this.style.height = (this.scrollHeight) + 'px'; // Set to content height
});

// Handle 'Enter' to send, 'Shift+Enter' for newline
chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Mock sending a message
function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Remove welcome screen on first message
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
        chatStream.style.justifyContent = 'flex-start';
    }

    createBubble('user', text);
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatStream.scrollTop = chatStream.scrollHeight;

    // Simulate AI Mock Response 
    setTimeout(() => simulateAIResponse(text), 500);
}

// Create a generic message bubble inside DOM
function createBubble(role, content) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${role}`;
    
    // Generate inner HTML structure mapping to pyside6 message_bubble.py layout
    const isUser = role === 'user';
    const roleLabel = isUser ? 'You' : 'Dizel';
    const avatarImg = isUser 
        ? "https://ui-avatars.com/api/?name=User&background=333&color=fff&rounded=true"
        : "https://ui-avatars.com/api/?name=Dizel&background=AD1535&color=fff&rounded=true";

    bubble.innerHTML = `
        <div class="bubble-container">
            ${!isUser ? `
            <div class="role-header">
                <img src="${avatarImg}" alt="${roleLabel}" style="width: 20px; height: 20px; border-radius: 50%;"> 
                ${roleLabel}
            </div>` : ''}
            <div class="message-body">${content}</div>
        </div>
    `;
    
    chatStream.appendChild(bubble);
    lucide.createIcons();
    return bubble;
}

// Logic to parse Chain of Thought (<think>...</think>) from a raw string
function parseAndFormatThought(rawContent) {
    const thinkRegex = /<think>([\s\S]*?)(<\/think>|$)/;
    const match = rawContent.match(thinkRegex);
    
    if (match) {
        const thoughtContent = match[1].trim();
        const mainContent = rawContent.replace(thinkRegex, '').trim();
        
        // This mirrors our recent ThoughtWidget addition!
        const html = `
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
        return html;
    }
    
    return `<div class="main-text">${rawContent.replace(/\n/g, '<br>')}</div>`;
}

// Make toggle globally accessible
window.toggleThought = function(btn) {
    const container = btn.nextElementSibling;
    if (container.classList.contains('hidden')) {
        container.classList.remove('hidden');
    } else {
        container.classList.add('hidden');
    }
    chatStream.scrollTop = chatStream.scrollHeight;
};

// Simulate Fake AI Generation (Streaming style)
function simulateAIResponse(userQuery) {
    const bubble = createBubble('assistant', '...');
    const body = bubble.querySelector('.message-body');
    
    // Fake the latency + Deepseek CoT format
    let fakeStream = `<think>\nAnalyzing the user's intent: "${userQuery}"\n1. Process request\n2. Determine parameters\n3. Construct response\nI will supply a formatted response with code.</think>\nHere is the synthesized response to your request!`;
    
    let currentIndex = 0;
    let accumulatedText = "";
    
    const interval = setInterval(() => {
        if (currentIndex < fakeStream.length) {
            accumulatedText += fakeStream.charAt(currentIndex);
            
            // Re-parse dynamically (just like MessageBubble _parse_and_update in PySide6)
            body.innerHTML = parseAndFormatThought(accumulatedText);
            lucide.createIcons({ root: body }); // re-init icons for the CPU toggle
            
            chatStream.scrollTop = chatStream.scrollHeight;
            currentIndex++;
        } else {
            clearInterval(interval);
        }
    }, 15); // Print speed
}

sendBtn.addEventListener('click', sendMessage);

// Dictation "Nova" Simulation
micBtn.addEventListener('click', () => {
    micBtn.classList.toggle('recording');
    if (micBtn.classList.contains('recording')) {
        novaPill.classList.remove('hidden');
    } else {
        novaPill.classList.add('hidden');
    }
});

novaCancel.addEventListener('click', () => {
    micBtn.classList.remove('recording');
    novaPill.classList.add('hidden');
});

novaDone.addEventListener('click', () => {
    micBtn.classList.remove('recording');
    novaPill.classList.add('hidden');
    chatInput.value = "Dictated text from Nova system...";
    chatInput.style.height = 'auto'; // trigger adjust
});
