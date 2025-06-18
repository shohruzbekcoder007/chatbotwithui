function scrollToBottom() {
    requestAnimationFrame(() => {
        const conversation = document.querySelector('.chat-messages');
        conversation.scrollTop = conversation.scrollHeight;
    });
}

function getChatIdFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('chat_id');
}

async function onsubmitstream(event) {
    
    event.preventDefault();

    const input = event.target.querySelector('input[name="message"]');
    const topicInput = event.target.querySelector('input[name="topic"]');
    const userText = input.value.trim();
    const topic = topicInput ? topicInput.value : 'default';
    console.log(topic, "<topic>");
    if (!userText) return;

    const chatId = getChatIdFromUrl();
    if (!chatId) {
        console.error("Chat ID not found in the URL.");
        alert("Chat ID topilmadi. Iltimos, sahifani qayta yuklang.");
        return;
    }

    addMessage('user', userText); // foydalanuvchi xabarini ko'rsatish
    input.value = "";

    // Update chat name in sidebar immediately if this is the first message
    updateChatNameInSidebar(chatId, userText);

    // To'g'ri selector ishlatish - conversation konteyneriga to'g'ridan-to'g'ri qo'shish
    const chatMessagesDiv = document.querySelector('.conversation');

    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    botMsg.innerHTML = `
    <div class="message-container">
        <div class="avatar">AI</div>
        <div class="message-content">
            <div class="message-text">
                <div class="loading">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
    </div>`;
    chatMessagesDiv.appendChild(botMsg);
    scrollToBottom();

    const messageTextSpan = botMsg.querySelector(".message-text");

    try {

        const headers = {
            'Content-Type': 'application/json'
        };
        // Token endi cookie'da saqlanadi va so'rov bilan avtomatik yuboriladi

        const response = await fetch("/chat/stream", {
            method: "POST",
            headers: headers,
            credentials: 'include', // Cookie'larni yuborish uchun
            body: JSON.stringify({
                query: userText,
                chat_id: chatId,
                language: "uz",
                device: "web",
                topic: topic
            })
        });

        if (!response.ok) {
            messageTextSpan.textContent = "Xatolik yuz berdi.";
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullText = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });

            // SSE protokol bo'yicha "data: " prefiksi bo'ladi
            chunk.split("\n").forEach(line => {
                if (line) {
                    const token = line;
                    fullText += token;
                    messageTextSpan.innerHTML = `<div class="message-text">${fullText}</div>`;
                    setTimeout(() => scrollToBottom(), 0);
                }
            });
        }
        messageTextSpan.innerHTML += `
            <div class="message-actions">
                <div class="feedback-buttons">
                    <button class="action-btn copy-btn" onclick="copyMessage(this)" title="Nusxa olish">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
                        </svg>
                    </button>
                    <button class="action-btn like-btn" onclick="giveFeedback(this, 'like')" title="Yaxshi javob">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/>
                        </svg>
                    </button>
                    <button class="action-btn dislike-btn" onclick="giveFeedback(this, 'dislike')" title="Yaxshi emas">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-7 10v2a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0-.714.211-1.412.608-2.006L17 13V4m-7 10h2"/>
                        </svg>
                    </button>
                    <button class="action-btn comment-btn" onclick="giveFeedback(this, 'comment')" title="Izoh qoldirish">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
                        </svg>
                    </button>
                </div>
            </div>
        `
    } catch (err) {
        messageTextSpan.textContent = "Ulanishda xatolik yuz berdi.";
        console.error("Streaming error:", err);
    }
}

function addMessage(sender, text) {

    let uuiddate = new Date().getTime();

    const messageHTML = `
        <div id="id${uuiddate}">
            <div class="message ${sender}">
                <div class="message-container">
                    <div class="message-content ${sender === "user" ? "text-right" : ""}">
                        <span>${text}</span>
                    </div>
                    <div class="avatar">${sender === "user" ? "Siz" : "AI"}</div>
                </div>
            </div>
        </div>`;
    document.querySelector(".conversation").insertAdjacentHTML("beforeend", messageHTML);
    scrollToBottom();
}

function sendRecommendedText(el) {
    const input = document.querySelector(".chat-input input");
    input.value = el.textContent;
    document.querySelector(".chat-input").dispatchEvent(new Event("submit"));
}