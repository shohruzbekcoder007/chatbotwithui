import redis
import json
import os
from typing import Optional

class RedisSession:
    """Redis'da foydalanuvchi sessiyasini saqlash"""
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        try:
            self.redis_client = redis.Redis(
                host,
                port,
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
        """Foydalanuvchi sessiyasini Redis'ga yozish"""
        if not self.connected:
            return
        try:
            self.redis_client.setex(f"session:{user_id}", self.ttl, json.dumps(data))
        except redis.ConnectionError:
            print("Warning: Redis connection failed while setting session")

    def get_user_session(self, user_id) -> Optional[dict]:
        """Foydalanuvchi sessiyasini Redis'dan olish"""
        if not self.connected:
            return None
        try:
            data = self.redis_client.get(f"session:{user_id}")
            return json.loads(data) if data else None
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
# redis_session.set_user_session("user123", {"last_query": "O‘zbekiston aholi soni"})
# print(redis_session.get_user_session("user123"))  # {'last_query': 'O‘zbekiston aholi soni'}
