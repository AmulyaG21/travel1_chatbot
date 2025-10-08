import redis

try:
    r = redis.Redis(host='localhost', port=6379)
    print(r.ping())  # Should print True if Redis is running
except redis.ConnectionError:
    print("Redis is not running or not installed.")
