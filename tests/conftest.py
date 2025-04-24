# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db_pool

@pytest.fixture(scope="module")
async def db_pool():
    # 初始化数据库连接池
    pool = await get_db_pool()
    yield pool
    # 测试结束后关闭连接池
    await pool.close()

# tests/conftest.py
@pytest.fixture
async def test_user(db_pool):
    async with db_pool.acquire() as conn:
        # 直接返回用户ID（不是协程）
        user_id = await conn.fetchval(
            "INSERT INTO users (username) VALUES ('test_user') RETURNING id"
        )
        return user_id

    

@pytest.fixture
def client(db_pool, test_user):
    # 注意：这里需要同步函数
    app.state.db_pool = db_pool
    with TestClient(app) as c:
        yield c

