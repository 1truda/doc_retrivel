import os

import pytest
from fastapi.testclient import TestClient
from app.main import app  # 导入你的 FastAPI 应用
from app.db import get_db_pool  # 导入连接池创建函数


# # 覆盖 startup 事件，确保测试时初始化 db_pool
# @app.on_event("startup")
# async def mock_startup():
#     app.state.db_pool = await get_db_pool()  # 或用模拟对象替代


# def test_upload_pdf(client):  # 自动注入 fixture
#     test_file = os.path.join(os.path.dirname(__file__), "test_files", "test.pdf")
#     with open(test_file, "rb") as f:
#         response = client.post("/documents/upload", files={"file": ("test.pdf", f)}, data={"user_id": "1"})
#     assert response.status_code == 200
@pytest.mark.asyncio
async def test_upload_pdf(client, test_user):
    test_file = os.path.join(os.path.dirname(__file__), "test_files", "test.pdf")

    # 确保测试文件存在
    if not os.path.exists(test_file):
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.4 minimal pdf file content")

    with open(test_file, "rb") as f:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", f)},
            data={"user_id": str(await test_user)}  # 等待协程并转换
        )

    assert response.status_code == 200