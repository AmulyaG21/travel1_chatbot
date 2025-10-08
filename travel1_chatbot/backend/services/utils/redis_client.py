import redis
from services.config import settings

def get_redis_client():
    if settings.REDIS_URL:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    else:
        client = redis.StrictRedis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            username=settings.REDIS_USERNAME,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True

        )

    try:
        client.ping()
        print("✅ Connected to Redis successfully!")
    except redis.exceptions.ConnectionError as e:
        print("❌ Redis connection failed:", e)
    return client

redis_client = get_redis_client()

