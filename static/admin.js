// Handle login
async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch('/api/admin/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('adminToken', data.token);
            document.getElementById('loginForm').classList.add('hidden');
            document.getElementById('dashboard').classList.remove('hidden');
            loadFeedbacks();
        } else {
            alert('Noto\'g\'ri login yoki parol!');
        }
    } catch (error) {
        console.error('Login error:', error);
        alert('Xatolik yuz berdi');
    }
}

// Handle logout
function handleLogout() {
    document.getElementById('loginForm').classList.remove('hidden');
    document.getElementById('dashboard').classList.add('hidden');
    localStorage.removeItem('adminToken');
}

// Load feedbacks from server
async function loadFeedbacks() {
    try {
        const token = localStorage.getItem('adminToken');
        if (!token) {
            handleLogout();
            return;
        }

        const response = await fetch('/api/admin/feedbacks', {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            }
        });
        const data = await response.json();
        
        if (data.success) {
            displayFeedbacks(data.feedbacks);
        } else {
            console.error('Error loading feedbacks:', data.error);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Display feedbacks in table
function displayFeedbacks(feedbacks) {
    const tbody = document.getElementById('feedbackTable');
    tbody.innerHTML = '';

    feedbacks.forEach(feedback => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${new Date(feedback.created_at).toLocaleString()}
            </td>
            <td class="px-6 py-4 text-sm text-gray-500">
                ${feedback.message_text}
            </td>
            <td class="px-6 py-4 text-sm text-gray-500">
                ${feedback.answer_text}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm">
                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                    ${feedback.feedback_type === 'like' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                    ${feedback.feedback_type}
                </span>
            </td>
            <td class="px-6 py-4 text-sm text-gray-500">
                ${feedback.comment || '-'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${feedback.user_id || 'Anonim'}
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Check if admin is already logged in
window.onload = function() {
    if (localStorage.getItem('adminToken')) {
        document.getElementById('loginForm').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');
        loadFeedbacks();
    }
};
