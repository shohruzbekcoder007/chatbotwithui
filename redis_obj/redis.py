import redis
import json
import os
from typing import Optional

class RedisSession:
    """Redis'da foydalanuvchi sessiyasini saqlash"""
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost") or host,
                port=int(os.getenv("REDIS_PORT", 6379)) or port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.connected = True
        except redis.ConnectionError:
            print("Warning: Could not connect to Redis. Running without session storage.")
            self.connected = False
    
        self.ttl = ttl

    def set_user_session(self, user_id, data):
        """Foydalanuvchi sessiyasini Redis'ga yozish (oxirgi 5 ta yozuvni saqlaydi)"""
        if not self.connected:
            return
        try:
            key = f"session:{user_id}"
            # Add new data to the beginning of the list
            self.redis_client.lpush(key, json.dumps(data))
            # Keep only the last 5 items
            self.redis_client.ltrim(key, 0, 4)
            # Set expiration time for the list
            self.redis_client.expire(key, self.ttl)
        except redis.ConnectionError:
            print("Warning: Redis connection failed while setting session")

    def get_user_session(self, user_id, count: int = 1) -> Optional[list]:
        """Foydalanuvchi sessiyasini Redis'dan olish
        
        Args:
            user_id: Foydalanuvchi ID
            count: Qaytariladigan yozuvlar soni (default=1, max=5)
        Returns:
            List of session data or None if no data exists
        """
        if not self.connected:
            return None
        try:
            count = min(max(1, count), 5)  # Ensure count is between 1 and 5
            key = f"session:{user_id}"
            data = self.redis_client.lrange(key, 0, count - 1)
            return [json.loads(item) for item in data] if data else None
        except redis.ConnectionError:
            print("Warning: Redis connection failed while getting session")
            return None

    def delete_user_session(self, user_id):
        """Foydalanuvchi sessiyasini Redis'dan o'chirish"""
        if not self.connected:
            return
        try:
            self.redis_client.delete(f"session:{user_id}")
        except redis.ConnectionError:
            print("Warning: Redis connection failed while deleting session")

    def delete_all_sessions(self):
        """Redis'da barcha sessiyalarni o'chirish"""
        if not self.connected:
            return
        try:
            self.redis_client.flushdb()
        except redis.ConnectionError:
            print("Warning: Redis connection failed while flushing database")

# Namuna ishlatish
redis_session = RedisSession()
# Bir nechta so'rovlarni saqlash
# redis_session.set_user_session("user123", {"last_query": "O'zbekiston aholi soni"})
# redis_session.set_user_session("user123", {"last_query": "Toshkent ob-havosi"})
# print(redis_session.get_user_session("user123", 2))  # Oxirgi 2 ta yozuvni olish
