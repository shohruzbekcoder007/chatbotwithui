// Foydalanuvchi autentifikatsiyasini tekshirish
async function checkAuthentication() {
    try {
        // Foydalanuvchi ma'lumotlarini olish uchun /auth/me endpoint ga so'rov yuborish
        const response = await fetch('/auth/me', {
            credentials: 'include' // Cookie-dagi tokenni yuborish uchun
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.authenticated) {
                return {
                    authenticated: true,
                    userName: data.name
                };
            }
        }
        return {
            authenticated: false,
            userName: null
        };
    } catch (error) {
        console.error('Foydalanuvchi ma\'lumotlarini olishda xatolik:', error);
        return {
            authenticated: false,
            userName: null
        };
    }
}

// Tizimdan chiqish
async function logout() {
    try {
        // Tizimdan chiqish uchun /auth/logout endpoint ga so'rov yuborish
        await fetch('/auth/logout', {
            method: 'POST',
            credentials: 'include' // Cookie-dagi tokenni yuborish uchun
        });
        
        // localStorage-dan tokenni o'chirish (agar bo'lsa)
        localStorage.removeItem('token');
        
        // Bosh sahifaga qaytish
        window.location.href = '/';
    } catch (error) {
        console.error('Tizimdan chiqishda xatolik:', error);
        // Xatolik bo'lsa ham bosh sahifaga qaytish
        window.location.href = '/';
    }
}
