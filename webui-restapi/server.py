#!/usr/bin/env python3
"""
Audio2Face REST API Web Interface Backend
提供文件上传和下载功能,配合前端使用Audio2Face REST API
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

# 配置
current_dir = Path(__file__).parent
UPLOAD_DIR = current_dir / "tmp"
LOG_FILE = UPLOAD_DIR / "server.log"  # 日志文件放在 tmp 目录

# Audio2Face REST API 地址
A2F_API_BASE = "http://127.0.0.1:8011"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Audio2Face Web Interface",
    description="文件上传下载服务,配合Audio2Face REST API使用",
    version="1.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
current_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(current_dir)), name="static")


@app.get("/")
async def root():
    """重定向到前端页面"""
    return FileResponse(current_dir / "index.html")


@app.get("/api/config")
async def get_config():
    """返回服务器配置信息"""
    return JSONResponse({
        "usd_file": str(current_dir / "claire_solved_arkit_offline.usd"),
        "export_dir": str(UPLOAD_DIR),
    })


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    上传音频文件到服务器
    
    Args:
        file: 音频文件
        
    Returns:
        文件路径和名称
    """
    # 验证文件类型
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。允许的类型: {', '.join(allowed_extensions)}"
        )
    
    # 保存文件
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存文件失败: {str(e)}"
        )
    
    return {
        "success": True,
        "filename": file.filename,
        "path": str(file_path),
        "size": file_path.stat().st_size
    }


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    下载结果文件
    
    Args:
        filename: 文件名
        
    Returns:
        文件下载响应
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {filename}"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/api/files")
async def list_files():
    """列出所有可用文件"""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "path": str(file_path)
            })
    return {"files": files}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除文件"""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {filename}"
        )
    
    try:
        file_path.unlink()
        return {"success": True, "message": f"文件已删除: {filename}"}
    except Exception as e:
        r


@app.api_route("/api/a2f/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_a2f_api(path: str, request: Request):
    """
    代理 Audio2Face REST API 请求,解决 CORS 问题
    
    所有 /api/a2f/* 的请求都会被转发到 Audio2Face REST API
    """
    # 构建目标 URL
    target_url = f"{A2F_API_BASE}/{path}"
    
    # 获取请求体
    body = None
    if request.method in ["POST", "PUT"]:
        body = await request.body()
        # 调试日志
        if body:
            print(f"[代理] {request.method} /{path}")
            print(f"[请求体] {body.decode('utf-8')}")
    
    # 获取查询参数
    query_params = dict(request.query_params)
    
    # 转发请求
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.request(
                method=request.method,
                url=target_url,
                params=query_params,
                content=body,
                headers={k: v for k, v in request.headers.items() 
                        if k.lower() not in ['host', 'connection']}
            )
            
            # 记录错误响应
            if response.status_code >= 400:
                print(f"[错误响应] {response.status_code}: {response.text}")
            
            # 返回响应
            return JSONResponse(
                content=response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                status_code=response.status_code
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="无法连接到 Audio2Face REST API,请确保服务正在运行"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"代理请求失败: {str(e)}"
            )


if __name__ == "__main__":
    print("=" * 60)
    print("Audio2Face Web Interface")
    print("=" * 60)
    print(f"上传目录: {UPLOAD_DIR}")
    print(f"日志文件: {LOG_FILE}")
    print(f"前端访问地址: http://127.0.0.1:8000")
    print(f"确保Audio2Face REST API运行在: http://127.0.0.1:8011")
    print("=" * 60)
    
    # 配置日志输出到控制台和文件
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s:     %(name)s - %(message)s",
                },
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(LOG_FILE),
                    "formatter": "default",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "file"],
            },
        },
    )
