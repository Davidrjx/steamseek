FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements_update.txt .
RUN pip install --no-cache-dir -r requirements_update.txt

# 复制应用代码（.dockerignore 会自动排除 venv 等文件）
COPY . .

# 创建必要的目录
RUN mkdir -p data logs templates static

# 暴露 Flask 端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False
ENV PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 启动应用
CMD ["python", "app.py"]
