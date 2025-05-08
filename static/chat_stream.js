function scrollToBottom() {
    requestAnimationFrame(() => {
        const conversation = document.querySelector('.chat-messages');
        conversation.scrollTop = conversation.scrollHeight;
    });
}

async function onsubmitstream(event) {
    event.preventDefault();

    const input = event.target.querySelector('input');
    const userText = input.value.trim();
    if (!userText) return;

    addMessage('user', userText); // foydalanuvchi xabarini ko'rsatish
    input.value = "";

    const chatMessagesDiv = document.querySelector(".chat-messages");

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
        const response = await fetch("/chat/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                content: userText,
                context: "", // kerakli system prompt bo‘lsa shu yerga qo‘sh
                language: "uz"
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

            // SSE protokol bo‘yicha "data: " prefiksi bo‘ladi
            chunk.split("\n").forEach(line => {
                if (line.startsWith("data: ")) {
                    const token = line.replace("data: ", "");
                    fullText += token;
                    messageTextSpan.innerHTML = fullText;
                    setTimeout(() => scrollToBottom(), 0);
                }
            });
        }
    } catch (err) {
        messageTextSpan.textContent = "Ulanishda xatolik yuz berdi.";
    }
}

function addMessage(sender, text) {
    const messageHTML = `
        <div class="message ${sender}">
            <div class="message-container">
                <div class="message-content ${sender === "user" ? "text-right" : ""}">
                    <div class="message-text">${text}</div>
                </div>
                <div class="avatar">${sender === "user" ? "Siz" : "AI"}</div>
            </div>
        </div>`;
    document.querySelector(".chat-messages").insertAdjacentHTML("beforeend", messageHTML);
    scrollToBottom();
}

function sendRecommendedText(el) {
    const input = document.querySelector(".chat-input input");
    input.value = el.textContent;
    document.querySelector(".chat-input").dispatchEvent(new Event("submit"));
}