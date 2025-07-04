function sendRecommendedText(element) {
    const text = element.textContent;
    const input = document.querySelector('.chat-input input[name="message"]');
    input.value = text;
    
    // Create and dispatch a submit event to trigger onsubmitstream
    const form = document.querySelector('.chat-input');
    const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
    form.dispatchEvent(submitEvent);
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
let currentFeedbackButton = null;

function giveFeedback(button, type) {
    currentFeedbackButton = button;

    // Get the message text
    const messageContainer = button.closest('.message-container');
    const messageText = messageContainer.querySelector('.message-text').textContent;

    // Store the feedback type and message for the modal
    const feedbackModal = document.getElementById('feedbackModal');
    feedbackModal.dataset.messageText = messageText; // chatbot's answer
    feedbackModal.dataset.feedbackType = type;

    // Clear previous comment
    document.getElementById('feedbackComment').value = '';

    // Show the modal
    feedbackModal.classList.remove('hidden');
}

function closeModal() {
    const feedbackModal = document.getElementById('feedbackModal');
    feedbackModal.classList.add('hidden');
}

async function submitFeedback() {
    if (!currentFeedbackButton) return;

    const modal = document.getElementById('feedbackModal');
    const messageText = modal.dataset.messageText;
    const feedbackType = modal.dataset.feedbackType;
    const comment = document.getElementById('feedbackComment').value;

    try {
        // Send feedback to server
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message_text: messageText,
                feedback_type: feedbackType === 'comment' ? 'comment' : feedbackType,
                comment: comment,
                user_id: localStorage.getItem('user_id')
            })
        });

        const data = await response.json();

        if (!data.success) {
            console.error('Feedback error:', data.error);
            showToast('Xatolik yuz berdi', true);
            return;
        }

        // Show success message using toast
        showToast('Fikringiz uchun rahmat! 👍');

        // Clear form and close modal
        document.getElementById('feedbackComment').value = '';
        modal.dataset.messageText = '';
        modal.dataset.feedbackType = '';
        currentFeedbackButton = null;
        closeModal();

    } catch (error) {
        console.error('Error sending feedback:', error);
        showToast('Xatolik yuz berdi', true);
    }
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
                    <div class="avatar">Siz</div>
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

    // Token olish (faqat yangi token)
    const token = localStorage.getItem('token');
    const headers = {
        'Content-Type': 'application/json'
    };

    // Agar token mavjud bo'lsa, headerga qo'shish
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    // Prepare request data
    const requestData = { query: message };

    // Add chat_id from URL only
    const chatId = getChatIdFromUrl();
    if (chatId) {
        requestData.chat_id = chatId;
    }

    fetch('/chat', {
        method: 'POST',
        headers: headers,
        credentials: 'include', // Cookie'larni yuborish uchun
        body: JSON.stringify(requestData)
    })
        .then(response => response.json())
        .then(data => {
            const messageContent = document.querySelector(`#id${uuiddate} .bot .message-content`);
            messageContent.innerHTML = `
                <div class="message-content">
                    <div class="message-text">${data.response}</div>
                    <div class="user-query" style="display: none;">${message}</div>
                </div>
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
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/>
                            </svg>
                        </button>
                    </div>
                </div>
            `;
            input.removeAttribute('disabled');
            button.removeAttribute('disabled');

            // Auto scroll to bottom after response
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Update chat name in sidebar if this is the first message
            updateChatNameInSidebar(chatId, message);
        })
        .catch(error => {
            console.error('Error:', error);
            document.querySelector(`#id${uuiddate} .bot .message-content`).innerHTML = 'Kechirasiz, so\'rovingizni qayta ishlashda xatolik yuz berdi.';
            input.removeAttribute('disabled');
            button.removeAttribute('disabled');
        });
}

// Function to extract chat_id from URL query parameters
function getChatIdFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('chat_id');
}

// Function to redirect for a new chat - just go to root URL
async function redirectWithNewChat() {
    // Check if current chat is empty before creating a new one
    const currentChatId = getChatIdFromUrl();
    
    if (currentChatId) {
        // Check if there are any messages in the current conversation
        const existingMessages = document.querySelectorAll('.conversation .message');
        
        // If current chat is empty (no user messages), don't create a new chat
        if (existingMessages.length === 0) {
            console.log('Current chat is empty, not creating a new chat');
            return; // Don't redirect, stay in current empty chat
        }
    }
    
    // Server will auto-generate chat_id and redirect
    window.location.href = '/';
}

// Function to check if this is the first message in a chat and update sidebar name
async function updateChatNameInSidebar(chatId, firstMessage) {
    try {
        // Check if there are any existing messages in the conversation
        const existingMessages = document.querySelectorAll('.conversation .message');
        
        // If this is the first message (no existing messages), update the sidebar
        if (existingMessages.length <= 2) { // 2 because we just added user and bot message
            // Database da to'liq nomni saqlash, sidebar da qisqartirib ko'rsatish
            const fullChatName = firstMessage; // To'liq nom
            const displayChatName = firstMessage.length > 30 ? firstMessage.substring(0, 30) + "..." : firstMessage; // Sidebar uchun qisqartirilgan
            
            // Find the current chat in sidebar and update its name
            const currentChatLink = document.querySelector(`a[href*="chat_id=${chatId}"]`);
            if (currentChatLink) {
                const chatNameSpan = currentChatLink.querySelector('.chat-name');
                if (chatNameSpan) {
                    // Sidebar da qisqartirilgan nomni ko'rsatish
                    chatNameSpan.textContent = displayChatName;
                    // Tooltip uchun to'liq nomni saqlash
                    chatNameSpan.setAttribute('title', fullChatName);
                }
            }
        }
    } catch (error) {
        console.error('Error updating chat name in sidebar:', error);
    }
}

// Load chat history when page loads
document.addEventListener('DOMContentLoaded', async function () {
    // Get chat ID from URL query parameter
    const chatId = getChatIdFromUrl();

    if (chatId) {
        // Load chat list sidebar
        await loadChatHistory(chatId);
        
        // Load the actual chat messages for this chat
        await loadChatMessages(chatId);
    }
    // No else needed - server will handle missing chat_id

    // Setup new chat button
    const newChatBtn = document.querySelector('.new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function () {
            redirectWithNewChat();
        });
    }
    
    // Setup chat link click handlers
    setupChatLinkHandlers();
});

// Function to setup click handlers for chat links
function setupChatLinkHandlers() {
    // Use event delegation for chat links (works for dynamically added elements too)
    document.querySelector('.chat-history').addEventListener('click', function(event) {
        // Find if a chat link was clicked
        const chatLink = event.target.closest('.chat-link');
        if (chatLink) {
            event.preventDefault(); // Prevent default navigation
            
            // Extract chat_id from href
            const url = new URL(chatLink.href);
            const chatId = url.searchParams.get('chat_id');
            
            if (chatId) {
                // Update URL without page reload
                window.history.pushState({}, '', `/?chat_id=${chatId}`);
                
                // Mark this chat as active
                document.querySelectorAll('.chat-history-item').forEach(item => {
                    item.classList.remove('active');
                });
                chatLink.querySelector('.chat-history-item').classList.add('active');
                
                // Load chat messages
                loadChatMessages(chatId);
            }
        }
        
        // Handle settings button clicks
        const settingsBtn = event.target.closest('.settings-btn');
        if (settingsBtn) {
            event.preventDefault();
            event.stopPropagation();
            
            // Close all other open dropdowns
            document.querySelectorAll('.settings-dropdown.show').forEach(dropdown => {
                dropdown.classList.remove('show');
            });
            
            // Toggle current dropdown
            const dropdown = settingsBtn.nextElementSibling;
            dropdown.classList.toggle('show');
        }
        
        // Handle dropdown item clicks
        const dropdownItem = event.target.closest('.dropdown-item');
        if (dropdownItem) {
            event.preventDefault();
            event.stopPropagation();
            
            const chatId = dropdownItem.dataset.chatId;
            
            if (dropdownItem.classList.contains('edit')) {
                const chatName = dropdownItem.dataset.chatName;
                renameChatPrompt(chatId, chatName);
            } else if (dropdownItem.classList.contains('delete')) {
                deleteChatPrompt(chatId);
            }
            
            // Close dropdown after action
            dropdownItem.closest('.settings-dropdown').classList.remove('show');
        }
    });
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.chat-settings')) {
            document.querySelectorAll('.settings-dropdown.show').forEach(dropdown => {
                dropdown.classList.remove('show');
            });
        }
    });
}

// Pagination state for chat list
let chatListPagination = {
    offset: 0,
    limit: 15, 
    loadMoreSize: 5,
    hasMore: true,
    isLoading: false,
    total: 0,
    isExpanded: false 
};

// Function to load chat history
async function loadChatHistory(chatId, resetPagination = true) {
    if (!chatId) {
        chatId = getChatIdFromUrl();
        if (!chatId) {
            console.error('No chat ID available to load history');
            return;
        }
    }

    // Reset pagination if needed
    if (resetPagination) {
        chatListPagination = {
            offset: 0,
            limit: 15, 
            loadMoreSize: 5,
            hasMore: true,
            isLoading: false,
            total: 0,
            isExpanded: false
        };
    }

    // If already loading, don't make another request
    if (chatListPagination.isLoading) {
        return;
    }

    chatListPagination.isLoading = true;

    const headers = {
        'Content-Type': 'application/json'
    };

    // So'rovni yuborish with pagination parameters
    const response = await fetch(`/api/user-chats?limit=${chatListPagination.limit}&offset=${chatListPagination.offset}`, {
        method: 'GET',
        headers: headers,
        credentials: 'include'
    });

    if (!response.ok) {
        chatListPagination.isLoading = false;
        throw new Error('Failed to load chat history');
    } else {
        const data = await response.json();
        if (data.success) {
            let chatHistory = document.querySelector('.chat-history');
            
            // Update pagination state
            chatListPagination.hasMore = data.pagination.has_more;
            chatListPagination.total = data.pagination.total;
            chatListPagination.offset = data.pagination.offset;
            
            // Only clear the container if this is the first page
            if (chatListPagination.offset === 0) {
                chatHistory.innerHTML = '';
            }

            // Add new chats to the list
            data?.chats?.forEach?.(chat => {
                // Chat nomini 30 belgi + 3 nuqta formatida ko'rsatish
                const fullName = chat?.name || "Yangi suhbat";
                const displayName = fullName.length > 30 ? fullName.substring(0, 30) + "..." : fullName;
                
                chatHistory.innerHTML += `
                    <div class="chat-item-container">
                        <a href="/?chat_id=${chat?.chat_id}" class="chat-link">
                            <div class="chat-history-item ${chat?.chat_id === chatId ? 'active' : ''}">
                                <svg stroke="currentColor" fill="none" viewBox="0 0 24 24" width="16" height="16">
                                    <path d="M20 2a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h16z" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                                <span class="chat-name" title="${fullName}">${displayName}</span>
                            </div>
                        </a>
                        <div class="chat-settings">
                            <button class="settings-btn" type="button">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="12" cy="12" r="1"></circle>
                                    <circle cx="19" cy="12" r="1"></circle>
                                    <circle cx="5" cy="12" r="1"></circle>
                                </svg>
                            </button>
                            <div class="settings-dropdown">
                                <button class="dropdown-item edit" data-chat-id="${chat?.chat_id}" data-chat-name="${fullName}">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                                    </svg>
                                    Tahrirlash
                                </button>
                                <button class="dropdown-item delete" data-chat-id="${chat?.chat_id}">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M3 6h18"></path>
                                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                    </svg>
                                    O'chirish
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });

            // Agar chatlar bo'lmasa va birinchi sahifa bo'lsa
            if ((!data.chats || data.chats.length === 0) && chatListPagination.offset === 0) {
                chatHistory.innerHTML = '<div class="no-chats-message">Hozircha chatlar yo\'q</div>';
            }
            
            // Smart button logic
            updateLoadMoreButton(chatHistory);
            
            // Setup scroll event for chat history container
            setupChatHistoryScroll();
        }
        
        chatListPagination.isLoading = false;
    }
}

// Function to update load more button based on current state
function updateLoadMoreButton(chatHistory) {
    // Remove existing button if any
    const existingBtn = document.querySelector('.load-more-chats');
    if (existingBtn) {
        existingBtn.remove();
    }

    // Don't show button if no chats at all
    if (chatListPagination.total === 0) {
        return;
    }

    // Calculate how many chats are currently shown
    const currentlyShown = chatListPagination.offset + chatListPagination.limit;
    
    // Show button if there are more chats to load OR if expanded and can collapse
    if (chatListPagination.hasMore || (chatListPagination.isExpanded && currentlyShown > 15)) {
        const loadMoreBtn = document.createElement('div');
        loadMoreBtn.className = 'load-more-chats';
        
        // Determine button text and action
        if (chatListPagination.isExpanded && !chatListPagination.hasMore) {
            // All chats are shown, show "Yopish" button
            loadMoreBtn.textContent = 'Yopish';
            loadMoreBtn.onclick = collapseChats;
        } else {
            // Show "Ko'proq ko'rish" button
            loadMoreBtn.textContent = 'Ko\'proq ko\'rish';
            loadMoreBtn.onclick = loadMoreChats;
        }
        
        chatHistory.appendChild(loadMoreBtn);
    }
}

// Function to load more chats (5 at a time after initial 15)
async function loadMoreChats() {
    if (chatListPagination.isLoading) {
        return;
    }

    chatListPagination.isLoading = true;
    
    // Calculate next batch size
    const nextLimit = chatListPagination.loadMoreSize; // 5 ta
    const nextOffset = chatListPagination.offset + chatListPagination.limit;
    
    const headers = {
        'Content-Type': 'application/json'
    };

    try {
        const response = await fetch(`/api/user-chats?limit=${nextLimit}&offset=${nextOffset}`, {
            method: 'GET',
            headers: headers,
            credentials: 'include'
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                const chatHistory = document.querySelector('.chat-history');
                const chatId = getChatIdFromUrl();
                
                // Add new chats to the list
                data?.chats?.forEach?.(chat => {
                    const fullName = chat?.name || "Yangi suhbat";
                    const displayName = fullName.length > 30 ? fullName.substring(0, 30) + "..." : fullName;
                    
                    chatHistory.insertAdjacentHTML('beforeend', `
                        <div class="chat-item-container">
                            <a href="/?chat_id=${chat?.chat_id}" class="chat-link">
                                <div class="chat-history-item ${chat?.chat_id === chatId ? 'active' : ''}">
                                    <svg stroke="currentColor" fill="none" viewBox="0 0 24 24" width="16" height="16">
                                        <path d="M20 2a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h16z" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    </svg>
                                    <span class="chat-name" title="${fullName}">${displayName}</span>
                                </div>
                            </a>
                            <div class="chat-settings">
                                <button class="settings-btn" type="button">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <circle cx="12" cy="12" r="1"></circle>
                                        <circle cx="19" cy="12" r="1"></circle>
                                        <circle cx="5" cy="12" r="1"></circle>
                                    </svg>
                                </button>
                                <div class="settings-dropdown">
                                    <button class="dropdown-item edit" data-chat-id="${chat?.chat_id}" data-chat-name="${fullName}">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                                        </svg>
                                        Tahrirlash
                                    </button>
                                    <button class="dropdown-item delete" data-chat-id="${chat?.chat_id}">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                            <path d="M3 6h18"></path>
                                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                        </svg>
                                        O'chirish
                                    </button>
                                </div>
                            </div>
                        </div>
                    `);
                });
                
                // Update pagination state
                chatListPagination.limit += nextLimit; // 5 ta qo'shildi
                chatListPagination.hasMore = data.pagination.has_more;
                chatListPagination.isExpanded = true;
                
                // Update button
                updateLoadMoreButton(chatHistory);
            }
        }
    } catch (error) {
        console.error('Error loading more chats:', error);
    }
    
    chatListPagination.isLoading = false;
}

// Function to collapse chats back to default 15
function collapseChats() {
    // Reset to default state
    chatListPagination.isExpanded = false;
    chatListPagination.limit = 15;
    chatListPagination.offset = 0;
    
    // Reload chat history with default pagination
    loadChatHistory(getChatIdFromUrl(), true);
}

async function loadChatMessages(chatId) {
    try {
        // Get token for authorization
        const token = localStorage.getItem('token');
        const headers = {
            'Content-Type': 'application/json'
        };

        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        // Fetch chat history
        const response = await fetch(`/api/chat-history/${chatId}`, {
            method: 'GET',
            headers: headers,
            credentials: 'include' // Cookie'larni yuborish uchun
        });

        if (!response.ok) {
            throw new Error('Failed to load chat history');
        }

        const data = await response.json();

        if (data.success && data.messages && data.messages.length > 0) {
            console.log('Chat history loaded:', data.messages.length, 'messages');

            // Clear existing conversation
            document.querySelector('.conversation').innerHTML = '';

            // Add messages to conversation
            data.messages.forEach(msg => {
                const timestamp = new Date().getTime(); // Just using as unique ID
                document.querySelector('.conversation').innerHTML += `
                    <div id="id${timestamp}">
                        <div class="message user">
                            <div class="message-container">
                                <div class="message-content text-right">
                                    <span>${msg.message}</span>
                                </div>
                                <div class="avatar">Siz</div>
                            </div>
                        </div>
                        <div class="message bot">
                            <div class="message-container">
                                <div class="avatar">AI</div>
                                <div class="message-content">
                                    <div class="message-content">
                                        <div class="message-text">${msg.response}</div>
                                        <div class="user-query" style="display: none;">${msg.message}</div>
                                    </div>
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
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"/>
                                                </svg>
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            // Auto scroll to bottom
            const chatMessages = document.querySelector('.chat-messages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Chat nomini o'zgartirish uchun funksiya
function renameChatPrompt(chatId, currentName) {
    // Modal elementlarini olish
    const modal = document.getElementById('renameChatModal');
    const nameInput = document.getElementById('newChatName');
    const chatIdInput = document.getElementById('currentChatId');
    const cancelBtn = document.getElementById('cancelRenameBtn');
    const confirmBtn = document.getElementById('confirmRenameBtn');

    // Modal ma'lumotlarini to'ldirish
    nameInput.value = currentName;
    chatIdInput.value = chatId;

    // Modalni ko'rsatish
    modal.classList.remove('hidden');

    // Bekor qilish tugmasi uchun event listener
    cancelBtn.onclick = function () {
        modal.classList.add('hidden');
    };

    // Saqlash tugmasi uchun event listener
    confirmBtn.onclick = function () {
        const newName = nameInput.value.trim();
        if (newName !== '') {
            // Chat nomini yangilash uchun so'rov yuborish
            fetch(`/api/chat/${chatId}/rename`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include', // Cookie'larni yuborish uchun
                body: JSON.stringify({ name: newName })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Muvaffaqiyatli xabar ko'rsatish
                        showToast('Chat nomi muvaffaqiyatli o\'zgartirildi!');
                        // Chatlar ro'yxatini yangilash
                        loadChatHistory(chatId);
                    } else {
                        // Xatolik xabarini ko'rsatish
                        showToast('Chat nomini o\'zgartirishda xatolik yuz berdi: ' + (data.error || 'Noma\'lum xatolik'), true);
                    }
                    // Modalni yopish
                    modal.classList.add('hidden');
                })
                .catch(error => {
                    showToast('Chat nomini o\'zgartirishda xatolik yuz berdi', true);
                    console.error('Chat nomini o\'zgartirishda xatolik yuz berdi:', error);
                    modal.classList.add('hidden');
                });
        } else {
            showToast('Chat nomi bo\'sh bo\'lishi mumkin emas', true);
        }
    };
}

// Chatni o'chirish uchun funksiya
function deleteChatPrompt(chatId) {
    // Modal elementlarini olish
    const modal = document.getElementById('deleteChatModal');
    const chatIdInput = document.getElementById('deleteChatId');
    const cancelBtn = document.getElementById('cancelDeleteBtn');
    const confirmBtn = document.getElementById('confirmDeleteBtn');

    // Modal ma'lumotlarini to'ldirish
    chatIdInput.value = chatId;

    // Modalni ko'rsatish
    modal.classList.remove('hidden');

    // Bekor qilish tugmasi uchun event listener
    cancelBtn.onclick = function () {
        modal.classList.add('hidden');
    };

    // O'chirish tugmasi uchun event listener
    confirmBtn.onclick = function () {
        // Chatni o'chirish uchun so'rov yuborish
        fetch(`/api/chat/${chatId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include' // Cookie'larni yuborish uchun
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Muvaffaqiyatli xabar ko'rsatish
                    showToast('Chat muvaffaqiyatli o\'chirildi!');
                    // Bosh sahifaga qaytish yoki chatlar ro'yxatini yangilash
                    if (window.location.search.includes(`chat_id=${chatId}`)) {
                        window.location.href = '/';
                    } else {
                        loadChatHistory();
                    }
                } else {
                    // Xatolik xabarini ko'rsatish
                    showToast('Chatni o\'chirishda xatolik yuz berdi: ' + (data.error || 'Noma\'lum xatolik'), true);
                }
                // Modalni yopish
                modal.classList.add('hidden');
            })
            .catch(error => {
                showToast('Chatni o\'chirishda xatolik yuz berdi', true);
                console.error('Chatni o\'chirishda xatolik yuz berdi:', error);
                modal.classList.add('hidden');
            });
    };
}

// Toast xabarlarini ko'rsatish uchun funksiya
function showToast(message, isError = false) {
    console.log('Toast function called:', message, isError);
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');

    if (!toast || !toastMessage) {
        console.error('Toast elements not found');
        return;
    }

    // Agar oldingi toast hali ko'rsatilayotgan bo'lsa, uni tozalash
    if (toast.timeout) {
        clearTimeout(toast.timeout);
    }
    
    // Toast matnini o'rnatish
    toastMessage.textContent = message;

    // Xatolik bo'lsa, qizil rang berish
    if (isError) {
        toast.classList.add('bg-red-500');
        toast.classList.remove('bg-gray-800');
    } else {
        toast.classList.add('bg-gray-800');
        toast.classList.remove('bg-red-500');
    }

    // Toastni ko'rsatish
    toast.classList.add('visible');
    console.log('Toast should be visible now');

    // 3 sekunddan keyin toastni yashirish
    toast.timeout = setTimeout(() => {
        toast.classList.remove('visible');
        console.log('Toast hidden after timeout');
    }, 3000);
}

// Sahifa yuklanganda modal tugmalarini ishga tushirish
document.addEventListener('DOMContentLoaded', function () {
    // Esc tugmasi bosilganda modalni yopish
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            document.getElementById('renameChatModal').classList.add('hidden');
            document.getElementById('deleteChatModal').classList.add('hidden');
        }
    });

    // Modal tashqarisiga bosilganda modalni yopish
    const modals = document.querySelectorAll('#renameChatModal, #deleteChatModal');
    modals.forEach(modal => {
        modal.addEventListener('click', function (event) {
            if (event.target === modal) {
                modal.classList.add('hidden');
            }
        });
    });
    
    // Toast elementini yashirish (sahifa yuklanganda)
    const toast = document.getElementById('toast');
    if (toast) {
        toast.classList.add('translate-y-full');
    }
});

// Setup scroll event for chat history container (disabled for new pagination logic)
function setupChatHistoryScroll() {
    // Scroll-based loading disabled - using manual "Ko'proq ko'rish" button instead
    return;
}

// Handle scroll event for chat history container (disabled for new pagination logic)
function handleChatHistoryScroll() {
    // Scroll-based loading disabled - using manual "Ko'proq ko'rish" button instead
    return;
}
