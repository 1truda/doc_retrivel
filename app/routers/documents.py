# app/routers/documents.py
import pathlib
import shutil
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
import asyncpg
import os
import hashlib
from datetime import datetime
from typing import Optional
import json

from starlette import status

from app.db import get_connection

router = APIRouter(prefix="/documents", tags=["documents"])

BASE_DIR = pathlib.Path(__file__).parent.parent  # 定位到项目根目录
UPLOAD_DIR = BASE_DIR / "uploads"  # 存储到项目根目录下的uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 安全辅助函数
def secure_filename(filename: str) -> str:
    return filename.replace("../", "").replace("/", "_").strip()

async def is_pdf(file: UploadFile) -> bool:
    header = await file.read(4)
    await file.seek(0)
    return header == b'%PDF'

async def save_file_async(path: str, file: UploadFile):
    async with aiofiles.open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

async def check_storage_quota(conn: asyncpg.Connection, user_id: int, new_size: int):
    used = await conn.fetchval(
        "SELECT COALESCE(SUM(file_size), 0) FROM documents WHERE user_id = $1",
        user_id
    )
    quota = await conn.fetchval(
        "SELECT storage_quota FROM users WHERE id = $1",
        user_id
    )
    if used + new_size > quota:
        raise HTTPException(403, "存储空间不足")




@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    metadata: str = Form(None),
    conn: asyncpg.Connection = Depends(get_connection)
):
    try:
        # 1. 验证用户存在
        user_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)",
            user_id
        )
        if not user_exists:
            raise HTTPException(404, "用户不存在")

        # 2. 验证文件类型
        if not (file.content_type == "application/pdf" and
                file.filename.lower().endswith('.pdf') and
                await is_pdf(file)):
            raise HTTPException(400, "仅支持有效的PDF文件")

        # 3. 准备存储路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = secure_filename(file.filename)
        unique_filename = f"{user_id}_{timestamp}_{safe_filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # 4. 保存文件并计算哈希
        sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                sha256.update(chunk)
                await f.write(chunk)
        file_hash = sha256.hexdigest()
        file_size = os.path.getsize(file_path)

        if file_size == 0:
            raise HTTPException(400, "文件不能为空")

        # 5. 解析元数据
        meta = {}
        if metadata:
            try:
                meta = json.loads(metadata)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"元数据格式错误: {str(e)}")

        # 6. 插入数据库
        doc_id = await conn.fetchval(
            """
            INSERT INTO documents (
                user_id, original_filename, file_extension,
                storage_path, file_size, sha256_hash, metadata,
                document_type
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            RETURNING id
            """,
            user_id,
            safe_filename,
            os.path.splitext(safe_filename)[1][1:].lower(),
            file_path,
            file_size,
            file_hash,
            json.dumps(meta),
            "pdf"
        )

        return JSONResponse(
            status_code=201,
            content={"document_id": doc_id, "sha256_hash": file_hash}
        )

    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "文件已存在")
    except asyncpg.ForeignKeyViolationError:
        raise HTTPException(404, "用户不存在")
    except Exception as e:
        if 'file_path' in locals():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        raise HTTPException(500, f"上传失败: {str(e)}")