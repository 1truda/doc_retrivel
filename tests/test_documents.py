from fastapi.testclient import TestClient
from app.main import app
import os
import pytest

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_uploads():
    """测试后清理上传文件"""
    yield
    upload_dir = os.path.join(os.path.dirname(__file__), "../app/uploads")
    for filename in os.listdir(upload_dir):
        if filename.startswith(("test_", "1-")):
            os.remove(os.path.join(upload_dir, filename))

def test_upload_real_pdf():
    """测试 PDF 文件上传"""
    test_file_path = os.path.join(os.path.dirname(__file__), "test_files/sample.pdf")
    with open(test_file_path, "rb") as f:
        response = client.post(
            "/upload",  # 注意路径是否带前缀（如 /documents/upload）
            files={"file": ("real.pdf", f, "application/pdf")},
            data={"user_id": 1, "metadata": None}  # user_id 改为整数，metadata 可选
        )
    print(response.json())  # 调试用
    print(app.routes)
    assert response.status_code == 201