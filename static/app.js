function sendRecommendedText(element) {
    const text = element.textContent;
    const input = document.querySelector('.chat-input input');
    input.value = text;
    document.querySelector('.chat-input').requestSubmit();
}

function onsubmitnew(event) {
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

    // conversation.scrollTop = conversation.scrollHeight;

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query:message })
    })
    .then(response => response.json())
    .then(data => {
        document.querySelector(`#id${uuiddate} .bot .message-content`).innerHTML = data.response;
        input.removeAttribute('disabled');
        button.removeAttribute('disabled');
        // conversation.scrollTop = conversation.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        document.querySelector(`#id${uuiddate} .bot .message-content`).innerHTML = 'Kechirasiz, so\'rovingizni qayta ishlashda xatolik yuz berdi.';
        input.removeAttribute('disabled');
        button.removeAttribute('disabled');
    });
}
