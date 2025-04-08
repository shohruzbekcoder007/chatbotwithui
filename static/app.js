function sendRecommendedText(element) {
    const text = element.textContent;
    const input = document.querySelector('.chat-input input');
    input.value = text;
    document.querySelector('.chat-input').requestSubmit();
}

// Copy message to clipboard
async function copyMessage(button) {
    const messageText = button.closest('.message-content').querySelector('.message-text').textContent;
    try {
        await navigator.clipboard.writeText(messageText.trim());
        // Show temporary success feedback
        const originalTitle = button.title;
        button.title = 'Nusxalandi!';
        setTimeout(() => {
            button.title = originalTitle;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy text:', err);
    }
}

// Handle feedback
function giveFeedback(button, type) {
    const feedbackButtons = button.closest('.feedback-buttons');
    const likeBtn = feedbackButtons.querySelector('.like-btn');
    const dislikeBtn = feedbackButtons.querySelector('.dislike-btn');

    if (type === 'like') {
        if (button.classList.contains('active')) {
            button.classList.remove('active');
        } else {
            button.classList.add('active');
            dislikeBtn.classList.remove('active');
        }
    } else {
        if (button.classList.contains('active')) {
            button.classList.remove('active');
        } else {
            button.classList.add('active');
            likeBtn.classList.remove('active');
        }
    }

    // Store the feedback type and message for the modal
    const messageText = button.closest('.message-content').querySelector('.message-text').textContent;
    const feedbackModal = document.getElementById('feedbackModal');
    feedbackModal.dataset.messageText = messageText;
    feedbackModal.dataset.feedbackType = type;
    
    // Show the modal
    openCommentModal(button);
}

function openCommentModal(button) {
    const messageText = button.closest('.message-content').querySelector('.message-text').textContent;
    const feedbackModal = document.getElementById('feedbackModal');
    feedbackModal.dataset.messageText = messageText;
    feedbackModal.classList.remove('hidden');
    document.getElementById('feedbackComment').value = '';

    // Get user's question from the previous message
    const messageDiv = button.closest('#id' + new Date().getTime());
    if (!messageDiv) {
        const allMessages = document.querySelectorAll('[id^="id"]');
        for (let msg of allMessages) {
            if (msg.contains(button)) {
                const userMessage = msg.querySelector('.user .message-content span');
                if (userMessage) {
                    feedbackModal.dataset.userQuestion = userMessage.textContent;
                }
                break;
            }
        }
    }
}

function closeModal() {
    const feedbackModal = document.getElementById('feedbackModal');
    feedbackModal.classList.add('hidden');
}

function submitFeedback() {
    const modal = document.getElementById('feedbackModal');
    const comment = document.getElementById('feedbackComment').value;
    const messageText = modal.dataset.messageText;
    const feedbackType = modal.dataset.feedbackType || '';
    const userQuestion = modal.dataset.userQuestion || '';

    // Here you can send the feedback to your backend
    console.log('Feedback:', {
        type: feedbackType,
        userQuestion: userQuestion,
        aiResponse: messageText,
        comment: comment
    });

    // Clear and close modal
    document.getElementById('feedbackComment').value = '';
    modal.classList.add('hidden');
    modal.dataset.userQuestion = '';
}

async function onsubmitnew(event) {
    event.preventDefault();

    let uuiddate = new Date().getTime();
    let conversation = document.querySelector('.conversation');
    let input = document.querySelector('.chat-input input');
    let message = input.value;
    input.value = '';

    let button = document.querySelector('.chat-input button');

    input.setAttribute('disabled', true);
    button.setAttribute('disabled', true);

    document.querySelector('.conversation').innerHTML += `
        <div id="id${uuiddate}">
            <div class="message user">
                <div class="message-container">
                    <div class="message-content text-right">
                        <span>${message}</span>
                    </div>
                    <div class="avatar">U</div>
                </div>
            </div>
            <div class="message bot">
                <div class="message-container">
                    <div class="avatar">AI</div>
                    <div class="message-content">
                        <div class="loading">
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Auto scroll to bottom after sending message
    const chatMessages = document.querySelector('.chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: message })
    })
        .then(response => response.json())
        .then(data => {
            const messageContent = document.querySelector(`#id${uuiddate} .bot .message-content`);
            messageContent.innerHTML = `
                <div class="message-text">${data.response}</div>
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
                        <button class="action-btn comment-btn" onclick="openCommentModal(this)" title="Izoh qoldirish">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
                            </svg>
                        </button>
                    </div>
                </div>
            `;
            input.removeAttribute('disabled');
            button.removeAttribute('disabled');
            
            // Auto scroll to bottom after response
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            document.querySelector(`#id${uuiddate} .bot .message-content`).innerHTML = 'Kechirasiz, so\'rovingizni qayta ishlashda xatolik yuz berdi.';
            input.removeAttribute('disabled');
            button.removeAttribute('disabled');
        });
}
