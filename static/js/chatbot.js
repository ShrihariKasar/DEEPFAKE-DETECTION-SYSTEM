/**
 * DeepGuard AI Forensic Assistant - Client Controller
 * Provides interactive chat UI, markdown rendering, suggestion chips,
 * Web Audio synthesised chimes, and session persistence.
 */

(function () {
    "use strict";

    // DOM Elements Cache
    let triggerBtn = null;
    let chatWidget = null;
    let chatBody = null;
    let chatForm = null;
    let chatInput = null;
    let sendBtn = null;
    let suggestionsContainer = null;
    let typingIndicator = null;
    let minimizeBtn = null;
    let expandBtn = null;
    let closeBtn = null;
    let clearBtn = null;

    // Chat State
    const STORAGE_KEY = "deepguard_chat_history_v1";
    const WIDGET_STATE_KEY = "deepguard_chat_widget_state";
    let isWaitingResponse = false;
    let chatHistory = [];

    // Synthesized Web Audio chime (no external mp3 file required)
    function playNotificationChime() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
            osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.12); // A5

            gain.gain.setValueAtTime(0.05, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start();
            osc.stop(ctx.currentTime + 0.25);
        } catch (e) {
            // Audio context not allowed before user interaction
        }
    }

    // Initialize UI on DOM Ready
    document.addEventListener("DOMContentLoaded", function () {
        initChatElements();
        loadStoredChat();
        bindEvents();
    });

    function initChatElements() {
        triggerBtn = document.getElementById("deepguard-chat-trigger");
        chatWidget = document.getElementById("deepguard-chat-widget");
        chatBody = document.getElementById("chat-body-messages");
        chatForm = document.getElementById("chat-input-form");
        chatInput = document.getElementById("chat-input-field");
        sendBtn = document.getElementById("chat-send-btn");
        suggestionsContainer = document.getElementById("chat-suggestions-container");
        typingIndicator = document.getElementById("chat-typing-indicator");
        minimizeBtn = document.getElementById("btn-minimize-chat");
        expandBtn = document.getElementById("btn-expand-chat");
        closeBtn = document.getElementById("btn-close-chat");
        clearBtn = document.getElementById("btn-clear-chat");
    }

    function bindEvents() {
        if (triggerBtn) {
            triggerBtn.addEventListener("click", function () {
                toggleChatWidget(true);
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener("click", function () {
                toggleChatWidget(false);
            });
        }

        if (minimizeBtn) {
            minimizeBtn.addEventListener("click", function () {
                if (!chatWidget) return;
                chatWidget.classList.toggle("minimized");
                const icon = minimizeBtn.querySelector("i");
                if (icon) {
                    if (chatWidget.classList.contains("minimized")) {
                        icon.className = "bi bi-chevron-up";
                    } else {
                        icon.className = "bi bi-dash-lg";
                    }
                }
            });
        }

        if (expandBtn) {
            expandBtn.addEventListener("click", function () {
                if (!chatWidget) return;
                chatWidget.classList.toggle("expanded");
                const icon = expandBtn.querySelector("i");
                if (icon) {
                    if (chatWidget.classList.contains("expanded")) {
                        icon.className = "bi bi-fullscreen-exit";
                    } else {
                        icon.className = "bi bi-arrows-angle-expand";
                    }
                }
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener("click", function () {
                if (confirm("Reset conversation with DeepGuard Assistant?")) {
                    resetConversation();
                }
            });
        }

        if (chatForm) {
            chatForm.addEventListener("submit", function (e) {
                e.preventDefault();
                handleUserSubmit();
            });
        }

        if (chatInput) {
            chatInput.addEventListener("keydown", function (e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleUserSubmit();
                }
            });
        }
    }

    function toggleChatWidget(show) {
        if (!chatWidget) return;
        if (show) {
            chatWidget.classList.add("active");
            chatWidget.classList.remove("minimized");
            sessionStorage.setItem(WIDGET_STATE_KEY, "open");
            if (triggerBtn) triggerBtn.style.display = "none";
            scrollToBottom();
            setTimeout(() => chatInput && chatInput.focus(), 150);
        } else {
            chatWidget.classList.remove("active");
            sessionStorage.setItem(WIDGET_STATE_KEY, "closed");
            if (triggerBtn) triggerBtn.style.display = "flex";
        }
    }

    function handleUserSubmit() {
        if (isWaitingResponse || !chatInput) return;
        const msg = chatInput.value.trim();
        if (!msg) return;

        chatInput.value = "";
        sendMessage(msg);
    }

    function sendMessage(userText) {
        // Render user message bubble
        appendMessage("user", userText);
        saveChatHistory("user", userText);

        isWaitingResponse = true;
        if (sendBtn) sendBtn.disabled = true;
        showTyping(true);

        fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: userText,
                history: chatHistory.slice(-6)
            })
        })
            .then(res => res.json())
            .then(data => {
                showTyping(false);
                isWaitingResponse = false;
                if (sendBtn) sendBtn.disabled = false;

                if (data && data.reply) {
                    appendMessage("bot", data.reply);
                    saveChatHistory("bot", data.reply);
                    playNotificationChime();

                    if (data.suggestions && data.suggestions.length > 0) {
                        renderSuggestions(data.suggestions);
                    }
                } else {
                    appendMessage("bot", "⚠️ Received empty response. Please retry.");
                }
            })
            .catch(err => {
                showTyping(false);
                isWaitingResponse = false;
                if (sendBtn) sendBtn.disabled = false;
                appendMessage("bot", "❌ Network error communicating with DeepGuard engine. Please check connection.");
            });
    }

    function appendMessage(sender, text) {
        if (!chatBody) return;

        const row = document.createElement("div");
        row.className = `chat-message-row ${sender === "user" ? "user-row" : "bot-row"}`;

        if (sender === "bot") {
            const avatar = document.createElement("div");
            avatar.className = "message-avatar bot-avatar-icon";
            avatar.innerHTML = '<i class="bi bi-shield-check"></i>';
            row.appendChild(avatar);

            const bubble = document.createElement("div");
            bubble.className = "message-bubble";
            bubble.innerHTML = parseMarkdown(text);
            row.appendChild(bubble);
        } else {
            const bubble = document.createElement("div");
            bubble.className = "message-bubble";
            bubble.textContent = text;
            row.appendChild(bubble);

            const avatar = document.createElement("div");
            avatar.className = "message-avatar user-avatar-icon";
            avatar.innerHTML = '<i class="bi bi-person"></i>';
            row.appendChild(avatar);
        }

        chatBody.appendChild(row);
        scrollToBottom();
    }

    function renderSuggestions(suggestions) {
        if (!suggestionsContainer) return;
        suggestionsContainer.innerHTML = "";

        suggestions.forEach(item => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "suggestion-chip";
            chip.textContent = item;
            chip.addEventListener("click", function () {
                sendMessage(item);
            });
            suggestionsContainer.appendChild(chip);
        });
    }

    function showTyping(show) {
        if (!typingIndicator) return;
        if (show) {
            typingIndicator.classList.add("active");
            if (chatBody) chatBody.appendChild(typingIndicator);
            scrollToBottom();
        } else {
            typingIndicator.classList.remove("active");
        }
    }

    function scrollToBottom() {
        if (chatBody) {
            chatBody.scrollTop = chatBody.scrollHeight;
        }
    }

    function saveChatHistory(sender, text) {
        chatHistory.push({ sender, text });
        if (chatHistory.length > 25) {
            chatHistory = chatHistory.slice(-25);
        }
        try {
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory));
        } catch (e) {
            // Storage quota exceeded
        }
    }

    function loadStoredChat() {
        try {
            const stored = sessionStorage.getItem(STORAGE_KEY);
            if (stored) {
                chatHistory = JSON.parse(stored);
                if (Array.isArray(chatHistory) && chatHistory.length > 0) {
                    if (chatBody) chatBody.innerHTML = "";
                    chatHistory.forEach(item => {
                        appendMessage(item.sender, item.text);
                    });
                }
            } else {
                // Initialize with initial bot welcome
                requestInitialGreeting();
            }

            // Restore open state if user had it open
            const state = sessionStorage.getItem(WIDGET_STATE_KEY);
            if (state === "open") {
                toggleChatWidget(true);
            }
        } catch (e) {
            requestInitialGreeting();
        }
    }

    function requestInitialGreeting() {
        fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "" })
        })
            .then(res => res.json())
            .then(data => {
                if (data && data.reply) {
                    if (chatBody) chatBody.innerHTML = "";
                    appendMessage("bot", data.reply);
                    saveChatHistory("bot", data.reply);
                    if (data.suggestions) {
                        renderSuggestions(data.suggestions);
                    }
                }
            })
            .catch(() => {});
    }

    function resetConversation() {
        sessionStorage.removeItem(STORAGE_KEY);
        chatHistory = [];
        if (chatBody) chatBody.innerHTML = "";
        requestInitialGreeting();
    }

    /**
     * Safe client-side Markdown parser for structured bot responses.
     */
    function parseMarkdown(md) {
        if (!md) return "";

        // Escape HTML to prevent injection
        let html = md
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Code blocks: ```code```
        html = html.replace(/```([\s\S]*?)```/g, function (match, code) {
            return "<pre><code>" + code.trim() + "</code></pre>";
        });

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

        // Headers
        html = html.replace(/^#### (.*$)/gim, "<h6>$1</h6>");
        html = html.replace(/^### (.*$)/gim, "<h5>$1</h5>");
        html = html.replace(/^## (.*$)/gim, "<h4>$1</h4>");
        html = html.replace(/^# (.*$)/gim, "<h3>$1</h3>");

        // Horizontal rules
        html = html.replace(/^---$/gim, "<hr/>");

        // Bold & Italic
        html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

        // Clickable Links: [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (match, text, url) {
            const cleanUrl = url.replace(/&amp;/g, "&");
            return '<a href="' + cleanUrl + '" target="_self">' + text + '</a>';
        });

        // Split into lines for list processing
        const lines = html.split("\n");
        let formatted = [];
        let inList = false;

        lines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                if (!inList) {
                    formatted.push("<ul>");
                    inList = true;
                }
                formatted.push("<li>" + trimmed.substring(2) + "</li>");
            } else if (/^\d+\.\s/.test(trimmed)) {
                if (!inList) {
                    formatted.push("<ol>");
                    inList = true;
                }
                const content = trimmed.replace(/^\d+\.\s/, "");
                formatted.push("<li>" + content + "</li>");
            } else {
                if (inList) {
                    formatted.push("</ul>");
                    inList = false;
                }
                if (trimmed.length > 0) {
                    if (!trimmed.startsWith("<h") && !trimmed.startsWith("<pre") && !trimmed.startsWith("<hr")) {
                        formatted.push("<p>" + line + "</p>");
                    } else {
                        formatted.push(line);
                    }
                }
            }
        });

        if (inList) {
            formatted.push("</ul>");
        }

        return formatted.join("");
    }
})();
