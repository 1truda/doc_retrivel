-- 启用必要扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- 用于生成唯一ID
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- 用于密码加密
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- 用于文本相似度搜索

-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    storage_quota BIGINT DEFAULT 1073741824, -- 默认1GB存储
    used_storage BIGINT DEFAULT 0,
    settings JSONB DEFAULT '{"dark_mode": false, "default_search_type": "vector"}'::jsonb
);

-- 用户会话表（用于登录状态管理）
-- CREATE TABLE user_sessions (
--     session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
--     ip_address INET,
--     user_agent TEXT,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
--     expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
--     is_revoked BOOLEAN DEFAULT FALSE
-- );

-- 文档表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename VARCHAR(512) NOT NULL,
    file_extension VARCHAR(20) NOT NULL,
    storage_path VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    upload_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE,
    processed BOOLEAN DEFAULT FALSE,
    processing_status VARCHAR(20) DEFAULT 'pending' 
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    metadata JSONB,
    sha256_hash VARCHAR(64) UNIQUE,
    document_type VARCHAR(50)  -- 可扩展为 'pdf', 'docx'等
);

-- 文档处理日志表
CREATE TABLE document_processing_logs (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL,
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER
);

-- 文档段落表（核心表）
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    start_char INTEGER,
    end_char INTEGER,
    vector_id VARCHAR(255),  -- 向量数据库引用ID
    embedding_model VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    -- 全文搜索字段
    search_vector TSVECTOR
);

-- 搜索历史表
CREATE TABLE search_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    search_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    results_count INTEGER NOT NULL,
    search_type VARCHAR(20) NOT NULL CHECK (search_type IN ('vector', 'keyword', 'hybrid')),
    session_id VARCHAR(255),
    search_params JSONB  -- 存储搜索参数如top_k等
);

-- 书签/收藏表
CREATE TABLE document_bookmarks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    tags VARCHAR(255)[] DEFAULT '{}'
);

-- 创建索引
CREATE INDEX idx_documents_user ON documents(user_id);
CREATE INDEX idx_documents_processed ON documents(processing_status) WHERE processed = FALSE;
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_vector_id ON document_chunks(vector_id) WHERE vector_id IS NOT NULL;
CREATE INDEX idx_search_history_user ON search_history(user_id);
CREATE INDEX idx_search_history_time ON search_history(search_time DESC);

-- 全文搜索索引
CREATE INDEX idx_chunks_search_vector ON document_chunks USING GIN(search_vector);

-- 触发器函数：自动更新全文搜索向量
CREATE OR REPLACE FUNCTION update_chunk_search_vector() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector = to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
CREATE TRIGGER trg_chunk_search_vector
BEFORE INSERT OR UPDATE OF chunk_text ON document_chunks
FOR EACH ROW EXECUTE FUNCTION update_chunk_search_vector();

-- 添加注释（可选）
COMMENT ON TABLE users IS '系统用户账户信息';
COMMENT ON TABLE documents IS '用户上传的文档元数据';
COMMENT ON TABLE document_chunks IS '文档分块内容，用于检索';
COMMENT ON COLUMN document_chunks.vector_id IS '对应向量数据库中的ID，如Milvus/Pinecone中的ID';