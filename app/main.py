import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.db import get_db_pool
from app.routers import documents, search  # 导入子路由
from app.routers.documents import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    app.state.db_pool = await get_db_pool()
    yield
    # 关闭时
    await app.state.db_pool.close()

app = FastAPI(lifespan=lifespan)
app.include_router(router)

# 包含子路由
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(search.router, prefix="/search", tags=["search"])






# import json
# import os
# import shutil
# import hashlib
# from datetime import datetime
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
# from fastapi.responses import JSONResponse
# from typing import Optional
# import asyncpg
# from pydantic import BaseModel
#
# app = FastAPI()
#
# # 配置
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
#
#
# # 数据库连接池
# async def get_db_conn():
#     return await asyncpg.connect(
#         user="postgres",
#         password="123456",
#         database="doc_search_db",
#         host="localhost"
#     )
#
#
# # 请求模型
# class DocumentMetadata(BaseModel):
#     description: Optional[str] = None
#     tags: Optional[list[str]] = None
#
#
# @app.post("/upload")
# async def upload_file(
#         file: UploadFile = File(...),
#         user_id: int = Form(...),
#         metadata: str = Form(None)  # JSON字符串
# ):
#     """处理文件上传并保存到数据库"""
#     try:
#         # 1. 验证文件类型
#         if not file.filename.lower().endswith('.pdf'):
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="仅支持PDF文件"
#             )
#
#         # 2. 准备存储路径
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         safe_filename = file.filename.replace("/", "_")  # 防止路径遍历
#         unique_filename = f"{user_id}_{timestamp}_{safe_filename}"
#         file_path = os.path.join(UPLOAD_DIR, unique_filename)
#
#         # 3. 保存文件
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#
#         # 4. 计算文件哈希
#         with open(file_path, "rb") as f:
#             file_hash = hashlib.sha256(f.read()).hexdigest()
#
#         # 5. 获取文件信息
#         file_size = os.path.getsize(file_path)
#         file_ext = os.path.splitext(file.filename)[1][1:].lower()
#
#         # 6. 解析元数据
#         meta = {}
#         if metadata:
#             try:
#                 meta = json.loads(metadata)
#             except json.JSONDecodeError as e:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"元数据格式错误: {str(e)}"
#                 )
#
#         # 7. 保存到数据库
#         conn = await get_db_conn()
#         try:
#             doc_id = await conn.fetchval(
#                 """
#                 INSERT INTO documents (
#                     user_id, original_filename, file_extension,
#                     storage_path, file_size, sha256_hash, metadata
#                 )
#                 VALUES ($1, $2, $3, $4, $5, $6, $7)
#                 RETURNING id
#                 """,
#                 user_id, file.filename, file_ext,
#                 file_path, file_size, file_hash, meta
#             )
#         except asyncpg.UniqueViolationError:
#             raise HTTPException(
#                 status_code=status.HTTP_409_CONFLICT,
#                 detail="相同文件已存在"
#             )
#         finally:
#             await conn.close()
#
#         return JSONResponse(
#             status_code=status.HTTP_201_CREATED,
#             content={
#                 "document_id": doc_id,
#                 "filename": file.filename,
#                 "size": file_size,
#                 "hash": file_hash
#             }
#         )
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         # 清理失败的文件
#         if 'file_path' in locals() and os.path.exists(file_path):
#             os.remove(file_path)
#
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"上传失败: {str(e)}"
#         )
#
#
# @app.on_event("startup")
# async def startup():
#     """创建数据库连接池"""
#     app.state.db_pool = await asyncpg.create_pool(
#         user="postgres",
#         password="123456",
#         database="doc_search_db",
#         host="localhost",
#         min_size=5,
#         max_size=20
#     )
#
#
# @app.on_event("shutdown")
# async def shutdown():
#     """关闭连接池"""
#     await app.state.db_pool.close()
#
# # 文档检索API
# # @app.get("/search")
# # async def search_documents(query: str, user_id: int, top_k: int = 5):
# #     # 1. 向量化查询文本
# #     # 2. 在向量数据库/PostgreSQL中搜索
# #     # 3. 返回匹配段落
# #     return {"results": [...]}
# #
# # # 文档管理API
# # @app.get("/documents")
# # async def list_documents(user_id: int):
# #     # 获取用户文档列表
# #     return {"documents": [...]}