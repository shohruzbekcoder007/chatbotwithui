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

// Global variables for pagination
let currentPage = 1;
let totalPages = 1;
const pageLimit = 10; // Har bir sahifada 10 ta element

// Load feedbacks from server with pagination
async function loadFeedbacks(page = 1) {
    try {
        const token = localStorage.getItem('adminToken');
        if (!token) {
            handleLogout();
            return;
        }

        // Pagination parametrlarini qo'shish
        const response = await fetch(`/api/admin/feedbacks?page=${page}&limit=${pageLimit}`, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            }
        });
        const data = await response.json();
        
        if (data.success) {
            // Pagination ma'lumotlarini saqlash
            currentPage = data.pagination.page;
            totalPages = data.pagination.total_pages;
            
            // Feedbacklarni ko'rsatish
            displayFeedbacks(data.feedbacks);
            
            // Pagination elementlarini yangilash
            updatePagination();
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

    if (feedbacks.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="6" class="px-6 py-4 text-center text-sm text-gray-500">
                Ma'lumotlar topilmadi
            </td>
        `;
        tbody.appendChild(row);
        return;
    }

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

// Update pagination controls
function updatePagination() {
    const paginationContainer = document.getElementById('pagination');
    if (!paginationContainer) return;
    
    paginationContainer.innerHTML = '';
    
    // Pagination elementlarini yaratish
    // Oldingi sahifa tugmasi
    const prevButton = document.createElement('button');
    prevButton.innerHTML = '&laquo; Oldingi';
    prevButton.className = `px-3 py-1 rounded-md ${currentPage === 1 ? 'bg-gray-200 text-gray-500 cursor-not-allowed' : 'bg-blue-500 text-white hover:bg-blue-600'}`;
    prevButton.disabled = currentPage === 1;
    prevButton.onclick = () => {
        if (currentPage > 1) {
            loadFeedbacks(currentPage - 1);
        }
    };
    paginationContainer.appendChild(prevButton);
    
    // Sahifa raqamlari
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    for (let i = startPage; i <= endPage; i++) {
        const pageButton = document.createElement('button');
        pageButton.innerText = i;
        pageButton.className = `mx-1 px-3 py-1 rounded-md ${i === currentPage ? 'bg-blue-600 text-white' : 'bg-gray-200 hover:bg-gray-300'}`;
        pageButton.onclick = () => loadFeedbacks(i);
        paginationContainer.appendChild(pageButton);
    }
    
    // Keyingi sahifa tugmasi
    const nextButton = document.createElement('button');
    nextButton.innerHTML = 'Keyingi &raquo;';
    nextButton.className = `px-3 py-1 rounded-md ${currentPage === totalPages ? 'bg-gray-200 text-gray-500 cursor-not-allowed' : 'bg-blue-500 text-white hover:bg-blue-600'}`;
    nextButton.disabled = currentPage === totalPages;
    nextButton.onclick = () => {
        if (currentPage < totalPages) {
            loadFeedbacks(currentPage + 1);
        }
    };
    paginationContainer.appendChild(nextButton);
}

// Check if admin is already logged in
window.onload = function() {
    if (localStorage.getItem('adminToken')) {
        document.getElementById('loginForm').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');
        loadFeedbacks(1); // Birinchi sahifani yuklash
    }
};

