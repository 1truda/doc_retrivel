from typing import AsyncIterator

import asyncpg
from fastapi import Request, HTTPException


async def get_db_pool():
    """创建数据库连接池（应用启动时调用）"""
    return await asyncpg.create_pool(
        user="postgres",
        password="123456",
        database="doc_search_db",
        host="localhost",
        min_size=5,
        max_size=20
    )

async def get_connection(request: Request) -> AsyncIterator[asyncpg.Connection]:
    """
    获取数据库连接的依赖项
    用法：
    @router.get("/items")
    async def read_items(conn: asyncpg.Connection = Depends(get_connection)):
        ...
    """
    try:
        async with request.app.state.db_pool.acquire() as conn:
            yield conn
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=500,
            detail=f"数据库连接错误: {str(e)}"
        )