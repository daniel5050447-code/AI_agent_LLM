/**
 * ===================================================================
 * Google Apps Script (GAS) Backend: Code.gs
 * ===================================================================
 * 
 * [역할 및 책임]
 * 1. Google Cloud Run에 배포된 FastAPI 백엔드 서버와의 안전한 통신 중계 (UrlFetchApp)
 * 2. Script Properties(환경변수)를 활용한 서버 엔드포인트 URL 동적 관리
 * 3. 클라이언트(index.html) 웹앱 서빙 (doGet)
 * 4. PDF 업로드 및 벡터 DB 생성 요청 중계
 * 5. Multi-Query + Hybrid Search + RRF 검색 및 스트리밍 연동 지원
 */

/**
 * 기본 Cloud Run 서버 URL (스크립트 속성에 등록되지 않은 경우 대체 기본값)
 */
var DEFAULT_SERVER_URL = "http://localhost:8080";

/**
 * 1. 웹 앱 진입점: index.html 템플릿 서빙
 */
function doGet(e) {
  var template = HtmlService.createTemplateFromFile('index');
  // 초기 렌더링 시 서버 URL 주입
  template.serverUrl = getServerUrl();
  
  return template.evaluate()
    .setTitle("Advanced RAG - FAISS & RAG Fusion 실습")
    .addMetaTag('viewport', 'width=device-width, initial-scale=1')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

/**
 * 2. 스크립트 속성(환경 변수)에서 Cloud Run 서버 URL 조회
 * PropertiesService를 사용하여 소스 코드 내 하드코딩 방지
 */
function getServerUrl() {
  var properties = PropertiesService.getScriptProperties();
  var url = properties.getProperty('SERVER_URL');
  if (!url) {
    url = DEFAULT_SERVER_URL;
  }
  // 끝 슬래시 제거
  return url.replace(/\/+$/, "");
}

/**
 * 스크립트 속성에 Cloud Run 서버 URL 저장/갱신
 * (GAS 에디터에서 직접 실행하거나 UI 설정 모달에서 호출 가능)
 */
function setServerUrl(newUrl) {
  if (!newUrl) {
    throw new Error("유효한 URL을 입력해 주세요.");
  }
  var sanitizedUrl = newUrl.trim().replace(/\/+$/, "");
  PropertiesService.getScriptProperties().setProperty('SERVER_URL', sanitizedUrl);
  return { success: true, url: sanitizedUrl };
}

/**
 * 3. 서버 헬스체크 및 연동 상태 확인
 * UrlFetchApp을 통해 백엔드의 /health 엔드포인트 호출
 */
function checkServerHealth() {
  var serverUrl = getServerUrl();
  var healthEndpoint = serverUrl + "/health";
  
  try {
    var response = UrlFetchApp.fetch(healthEndpoint, {
      method: "get",
      muteHttpExceptions: true,
      headers: { "Accept": "application/json" }
    });
    
    var statusCode = response.getResponseCode();
    var content = response.getContentText();
    
    if (statusCode === 200) {
      var data = JSON.parse(content);
      return {
        online: true,
        statusCode: statusCode,
        serverUrl: serverUrl,
        indexedChunks: data.indexed_chunks || 0,
        dimension: data.faiss_dimension || 768
      };
    } else {
      return {
        online: false,
        statusCode: statusCode,
        serverUrl: serverUrl,
        error: "서버 응답 오류 (HTTP " + statusCode + "): " + content
      };
    }
  } catch (err) {
    return {
      online: false,
      serverUrl: serverUrl,
      error: "서버 연결 불가: " + err.toString()
    };
  }
}

/**
 * 4. PDF 업로드 및 벡터 DB 인덱싱 요청
 * PDF 파일의 Base64 데이터를 수신하여 Cloud Run 백엔드 /api/rag/upload-pdf 로 전달
 * 
 * @param {Object} filePayload { filename: string, base64: string }
 */
function uploadPdfToRAG(filePayload) {
  if (!filePayload || !filePayload.base64) {
    throw new Error("업로드할 PDF 데이터가 올바르지 않습니다.");
  }

  var serverUrl = getServerUrl();
  var uploadEndpoint = serverUrl + "/api/rag/upload-pdf";
  
  var requestBody = {
    filename: filePayload.filename || "document.pdf",
    base64_data: filePayload.base64
  };

  try {
    var options = {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify(requestBody),
      muteHttpExceptions: true
    };

    var response = UrlFetchApp.fetch(uploadEndpoint, options);
    var statusCode = response.getResponseCode();
    var resultText = response.getContentText();

    if (statusCode >= 200 && statusCode < 300) {
      return JSON.parse(resultText);
    } else {
      throw new Error("서버 인덱싱 실패 (HTTP " + statusCode + "): " + resultText);
    }
  } catch (e) {
    throw new Error("PDF 업로드 중 예외 발생: " + e.message);
  }
}

/**
 * 5. RAG 질의 실행 (UrlFetchApp 기반 단일 취합 응답)
 * Multi-Query(5개) -> FAISS + BM25 Hybrid Search -> RRF Re-ranking -> LLM 답변
 * 
 * @param {string} query 사용자 질문
 * @param {number} topN 가져올 상위 문서 수 (기본: 5)
 */
function queryRAG(query, topN) {
  if (!query || query.trim() === "") {
    throw new Error("질의 내용이 비어 있습니다.");
  }

  var serverUrl = getServerUrl();
  var queryEndpoint = serverUrl + "/api/rag/query";
  var topCount = topN || 5;

  var requestBody = {
    query: query.trim(),
    top_n: topCount
  };

  try {
    var options = {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify(requestBody),
      muteHttpExceptions: true
    };

    var response = UrlFetchApp.fetch(queryEndpoint, options);
    var statusCode = response.getResponseCode();
    var resultText = response.getContentText();

    if (statusCode >= 200 && statusCode < 300) {
      return JSON.parse(resultText);
    } else {
      throw new Error("RAG 질의 실패 (HTTP " + statusCode + "): " + resultText);
    }
  } catch (e) {
    throw new Error("질의 통신 중 예외 발생: " + e.message);
  }
}

/**
 * 6. 브라우저 실시간 스트리밍 핸들러(StreamHandler)용 엔드포인트 URL 반환
 * 클라이언트 index.html에서 EventSource (SSE)를 직접 연결할 때 사용
 */
function getStreamEndpointUrl(query, topN) {
  var serverUrl = getServerUrl();
  var count = topN || 5;
  return serverUrl + "/api/rag/stream?query=" + encodeURIComponent(query) + "&top_n=" + count;
}
