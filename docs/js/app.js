// docs/js/app.js
document.addEventListener('DOMContentLoaded', () => {
    // Observer for scroll animation triggers
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-slide-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const checkElements = document.querySelectorAll('.scroll-observe');
    checkElements.forEach(el => observer.observe(el));

    // Copy to clipboard logic
    const copyBtn = document.getElementById('copy-btn');
    const installCmd = document.getElementById('install-cmd');
    
    if (copyBtn && installCmd) {
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(installCmd.innerText);
                const originalText = copyBtn.innerText;
                copyBtn.innerText = 'Copied!';
                copyBtn.style.color = 'var(--accent)';
                copyBtn.style.borderColor = 'var(--accent)';
                setTimeout(() => {
                    copyBtn.innerText = originalText;
                    copyBtn.style.color = '';
                    copyBtn.style.borderColor = '';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    }

    // Interactive Hover Glow Effect
    const glowCards = document.querySelectorAll('.glow-card');
    glowCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    });

    // Performance Bar filling
    const perfObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const fillBars = entry.target.querySelectorAll('.fill-target');
                fillBars.forEach(bar => {
                    const w = bar.getAttribute('data-width');
                    bar.style.width = w;
                });
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const perfGrids = document.querySelectorAll('.performance-grid');
    perfGrids.forEach(grid => perfObserver.observe(grid));

    // FAQ Accordion
    const faqQuestions = document.querySelectorAll('.faq-question');
    faqQuestions.forEach(btn => {
        btn.addEventListener('click', () => {
            const item = btn.parentElement;
            const answer = item.querySelector('.faq-answer');
            
            // Toggle active state
            const isActive = item.classList.contains('active');
            
            // Close all other accordions
            document.querySelectorAll('.faq-item').forEach(otherItem => {
                otherItem.classList.remove('active');
                otherItem.querySelector('.faq-answer').style.maxHeight = null;
            });

            if (!isActive) {
                item.classList.add('active');
                answer.style.maxHeight = answer.scrollHeight + "px";
            }
        });
    });

    // Use Cases Tab Switching
    const tabBtns = document.querySelectorAll('.usecases-tab');
    const tabContents = document.querySelectorAll('.usecases-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all
            tabBtns.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    // Ctrl+K Command Palette
    const cmdPalette = document.getElementById('cmd-palette');
    const cmdInput = document.getElementById('cmd-input');
    const cmdTrigger = document.getElementById('cmd-k-trigger');

    const togglePalette = () => {
        const isActive = cmdPalette.classList.contains('active');
        if (isActive) {
            cmdPalette.classList.remove('active');
            cmdPalette.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = '';
        } else {
            cmdPalette.classList.add('active');
            cmdPalette.setAttribute('aria-hidden', 'false');
            document.body.style.overflow = 'hidden';
            setTimeout(() => cmdInput.focus(), 100);
        }
    };

    if (cmdTrigger) cmdTrigger.addEventListener('click', togglePalette);
    
    // Close on background click
    if (cmdPalette) {
        cmdPalette.addEventListener('click', (e) => {
            if (e.target === cmdPalette) togglePalette();
        });
    }

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (cmdPalette) togglePalette();
        }
        if (e.key === 'Escape' && cmdPalette && cmdPalette.classList.contains('active')) {
            togglePalette();
        }
    });

    // Realtime Terminal Logs Simulator
    const terminalLogs = document.getElementById('terminal-logs');
    if (terminalLogs) {
        const fakeLogs = [
            '<span class="log-inf">[sys]</span> Scanning VRAM availability... <span class="log-ok">24.5GB Verified</span>',
            '<span class="log-inf">[api]</span> Local proxy endpoint starting on 127.0.0.1:8100',
            '<span class="log-inf">[loader]</span> Mounting <span style="color:#fff">deepseek-coder-v2</span> into Workspace::Coding',
            '<span class="log-inf">[cuBLAS]</span> Hardware acceleration hooks attached.',
            '<span class="log-warn">[metrics]</span> Context window expanding to 32,768 tokens.',
            '<span class="log-inf">[sys]</span> Tensor runtime operational.',
            '<span class="log-ok">[ready]</span> Dizel AI engine is awaiting instructions. Latency: 0.2ms'
        ];
        
        let logIndex = 0;
        const printLog = () => {
            if (logIndex >= fakeLogs.length) {
                // Occasionally drop a ping
                if (Math.random() > 0.7) {
                    const el = document.createElement('div');
                    el.className = 'log-line';
                    el.innerHTML = '<span class="log-ok">[ping]</span> Idle worker check-in... OK';
                    terminalLogs.appendChild(el);
                }
            } else {
                const el = document.createElement('div');
                el.className = 'log-line';
                el.innerHTML = fakeLogs[logIndex];
                terminalLogs.appendChild(el);
                logIndex++;
            }
            
            // Auto trim old logs
            while (terminalLogs.children.length > 8) {
                terminalLogs.removeChild(terminalLogs.firstChild);
            }
            
            // Random interval for realism
            setTimeout(printLog, Math.random() * 2000 + 500);
        };
        
        // Start sequence
        setTimeout(printLog, 1500);
    }
});
