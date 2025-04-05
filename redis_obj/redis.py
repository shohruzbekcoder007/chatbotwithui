import redis
import json

class RedisSession:
    """Redis'da foydalanuvchi sessiyasini saqlash"""

    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl

    def set_user_session(self, user_id, data):
        """Foydalanuvchi sessiyasini Redis'ga yozish"""
        self.redis_client.setex(f"session:{user_id}", self.ttl, json.dumps(data))

    def get_user_session(self, user_id):
        """Foydalanuvchi sessiyasini Redis'dan olish"""
        data = self.redis_client.get(f"session:{user_id}")
        return json.loads(data) if data else None

    def delete_user_session(self, user_id):
        """Foydalanuvchi sessiyasini Redis'dan o'chirish"""
        self.redis_client.delete(f"session:{user_id}")

    def delete_all_sessions(self):
        """Redis'da barcha sessiyalarni o'chirish"""
        self.redis_client.flushdb()

# Namuna ishlatish
redis_session = RedisSession()
# redis_session.set_user_session("user123", {"last_query": "O‘zbekiston aholi soni"})
# print(redis_session.get_user_session("user123"))  # {'last_query': 'O‘zbekiston aholi soni'}
