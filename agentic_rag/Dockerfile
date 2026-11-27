# 1. Base Image: Python 3.12 Slim (Lightweight & Secure)
FROM python:3.12-slim

# 2. 시스템 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 환경 변수 기본값 (Cloud Run 환경 호환)
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 7. 포트 노출
EXPOSE 8080

# 8. Uvicorn 서버 실행 (Cloud Run의 $PORT 변수 바인딩)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
