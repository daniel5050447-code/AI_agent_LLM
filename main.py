"""
FastAPI Backend Server for Advanced RAG
========================================
- FAISS IndexFlatIP + BM25 Hybrid Search
- 5-Way Multi-Query Expansion Layer
- Reciprocal Rank Fusion (RRF) Re-ranking Layer
- StreamHandler for Real-time Token & Status Streaming
- Google Apps Script (GAS) & Cloud Run Ready
"""

import os
import io
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader

from rag_pipeline import RAGPipeline

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Advanced RAG Engine (FAISS + MultiQuery + Hybrid + RRF)",
    description="FastAPI backend supporting Hybrid Search, 5-Way Multi-Query Expansion, and RRF Re-ranking",
    version="1.0.0"
)

# CORS 설정 (Google Apps Script 및 외부 웹앱 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 RAG 파이프라인 인스턴스 (인메모리 FAISS & BM25)
rag_pipeline = RAGPipeline()


# Pydantic 데이터 모델
class QueryRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5


class Base64PDFRequest(BaseModel):
    filename: str
    base64_data: str  # Base64로 인코딩된 PDF 바이너리


@app.get("/")
def root():
    """서버 상태 및 기본 안내"""
    return {
        "service": "Advanced RAG Server",
        "status": "online",
        "indexed_chunks_count": len(rag_pipeline.all_chunks),
        "faiss_index_type": rag_pipeline.faiss_manager.index_type,
        "faiss_flat_db_path": rag_pipeline.faiss_manager.index_file_path,
        "supported_features": [
            "FAISS Dense Vector Indexing (IndexFlatL2 / IndexHNSWFlat)",
            "BM25 Keyword Search",
            "5-Way Multi-Query Expansion",
            "Reciprocal Rank Fusion (RRF)",
            "Real-time Streaming via SSE"
        ],
        "endpoints": {
            "health": "/health",
            "upload_pdf": "/api/rag/upload-pdf (multipart or base64)",
            "query": "/api/rag/query (JSON)",
            "stream": "/api/rag/stream (SSE Streaming)"
        }
    }


@app.get("/health")
def health_check():
    """Cloud Run 헬스체크 및 연동 상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "indexed_chunks": len(rag_pipeline.all_chunks),
        "faiss_index_type": rag_pipeline.faiss_manager.index_type,
        "faiss_flat_db_file": os.path.basename(rag_pipeline.faiss_manager.index_file_path),
        "faiss_flat_db_exists": os.path.exists(rag_pipeline.faiss_manager.index_file_path),
        "faiss_dimension": rag_pipeline.embedding_service.dimension
    }


@app.post("/api/rag/upload-pdf")
async def upload_pdf(
    file: Optional[UploadFile] = File(None),
    payload: Optional[Base64PDFRequest] = None
):
    """
    PDF 파일 업로드 및 FAISS / BM25 인덱싱
    - multipart/form-data 또는 Base64 JSON 형식 모두 지원 (GAS 호환성)
    """
    try:
        pdf_bytes = None
        filename = "uploaded_document.pdf"

        if file:
            filename = file.filename or filename
            pdf_bytes = await file.read()
        elif payload:
            import base64
            filename = payload.filename or filename
            # data:application/pdf;base64,... 프리픽스 제거
            b64_str = payload.base64_data.split(",")[-1]
            pdf_bytes = base64.b64decode(b64_str)
        else:
            raise HTTPException(status_code=400, detail="업로드할 PDF 파일이 제공되지 않았습니다.")

        # PDF 텍스트 추출 (페이지별)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text: List[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append((i, text))

        if not pages_text:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다. 스캔본 이미지이거나 암호화된 파일일 수 있습니다.")

        # RAG 파이프라인에 적재 및 인덱싱 수행
        index_result = rag_pipeline.index_pdf_text(
            filename=filename,
            pages_text=pages_text,
            chunk_size=500,
            chunk_overlap=100
        )

        return JSONResponse(content={
            "success": True,
            "message": f"'{filename}' 문서가 성공적으로 FAISS 및 BM25 인덱스에 적재되었습니다.",
            "data": index_result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 인덱싱 중 오류 발생: {str(e)}")


@app.post("/api/rag/query")
async def execute_rag_query(req: QueryRequest):
    """
    단일 JSON 응답을 위한 RAG 질의 엔드포인트
    - GAS UrlFetchApp 연동에 최적화
    - Multi-Query -> Hybrid Search -> RRF Re-ranking -> LLM 답변
    """
    if not rag_pipeline.all_chunks:
        raise HTTPException(status_code=400, detail="먼저 PDF 문서를 업로드하여 벡터 DB를 생성해 주세요.")

    try:
        # 1. 하이브리드 검색 및 RRF 순위 융합
        search_res = rag_pipeline.search_hybrid_rrf(req.query, top_n=req.top_n)

        # 2. LLM 전체 답변 취합 생성
        answer_parts = []
        for token in rag_pipeline.generate_answer_stream(req.query, search_res["ranked_documents"]):
            answer_parts.append(token)
        final_answer = "".join(answer_parts)

        return {
            "success": True,
            "original_query": search_res["original_query"],
            "expanded_queries": search_res["expanded_queries"],
            "ranked_documents": search_res["ranked_documents"],
            "answer": final_answer,
            "timings": search_res["timings"],
            "complexity": search_res["complexity"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 질의 처리 실패: {str(e)}")


@app.get("/api/rag/stream")
@app.post("/api/rag/stream")
async def execute_rag_stream(query: str = Query(...), top_n: int = Query(5)):
    """
    실시간 스트리밍 핸들러 (StreamHandler) 엔드포인트
    - Server-Sent Events (SSE) 프로토콜
    - 1단계: 5가지 Multi-Query 확장 상태 및 결과
    - 2단계: FAISS + BM25 하이브리드 검색 상태
    - 3단계: RRF 상호 순위 융합 결과 및 출처 메타데이터
    - 4단계: LLM 토큰 단위 실시간 스트리밍
    """
    if not rag_pipeline.all_chunks:
        async def empty_stream():
            yield f"event: error\ndata: {json.dumps({'error': '인덱스된 문서가 없습니다. 먼저 PDF를 등록해 주세요.'})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    async def sse_event_generator():
        try:
            # 1. 시작 이벤트
            yield f"event: status\ndata: {json.dumps({'step': 1, 'message': '1/4. 다중 쿼리 확장 중 (5가지 관점)...'})}\n\n"
            await asyncio.sleep(0.05)

            # 2. 검색 및 RRF 융합 수행
            search_res = rag_pipeline.search_hybrid_rrf(query, top_n=top_n)

            yield f"event: multi_queries\ndata: {json.dumps({'queries': search_res['expanded_queries']})}\n\n"
            yield f"event: status\ndata: {json.dumps({'step': 2, 'message': '2/4. FAISS (Dense) & BM25 (Sparse) 하이브리드 검색 완료'})}\n\n"
            await asyncio.sleep(0.05)

            yield f"event: status\ndata: {json.dumps({'step': 3, 'message': '3/4. Reciprocal Rank Fusion(RRF) 상호 순위 융합 완료'})}\n\n"
            yield f"event: contexts\ndata: {json.dumps({'documents': search_res['ranked_documents'], 'timings': search_res['timings'], 'complexity': search_res['complexity']})}\n\n"
            await asyncio.sleep(0.05)

            # 3. LLM 스트리밍 생성
            yield f"event: status\ndata: {json.dumps({'step': 4, 'message': '4/4. 답변 생성 중...'})}\n\n"
            
            for token in rag_pipeline.generate_answer_stream(query, search_res["ranked_documents"]):
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                await asyncio.sleep(0.01)

            # 4. 완료 이벤트
            yield f"event: done\ndata: {json.dumps({'status': 'completed', 'timings': search_res['timings']})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"[*] Starting Advanced RAG Server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
