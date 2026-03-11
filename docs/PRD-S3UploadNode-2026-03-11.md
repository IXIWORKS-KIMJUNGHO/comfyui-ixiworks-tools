# PRD: ComfyUI S3 Upload Node

## 관련 문서

| 관계 | 문서 |
|------|------|
| 프로젝트 | comfyui-ixiworks-tools |

| 항목 | 내용 |
|------|------|
| Author | Kim Jungho |
| Date | 2026-03-11 |
| Version | 1.0 |
| Status | Draft |

---

## 📋 Executive Summary

> [!abstract] 프로젝트 개요
> ComfyUI 워크플로우에서 생성된 최종 결과물(이미지, 텍스트)을 AWS S3 버킷에 직접 업로드하는 커스텀 노드를 개발한다.
>
> 현재 `comfyui-ixiworks-tools` 패키지의 `Save Text Image` 노드는 로컬 디스크에만 저장이 가능하다. 생성 결과물을 외부 서비스나 팀과 공유하려면 별도의 수동 업로드 과정이 필요하며, 이는 자동화 파이프라인에서 병목이 된다.
>
> `Upload to S3` 노드를 추가하여 워크플로우 끝단에서 결과물이 자동으로 S3에 업로드되도록 한다. 기존 `Save Text Image` 노드와 병렬로 사용하여 로컬 저장 + 클라우드 업로드를 동시에 처리한다.

---

## 🔍 Background & Context

### 현재 상태
- `SaveFileNode` (Save Text Image)로 로컬 `output/` 디렉토리에 이미지/텍스트 저장
- 클라우드 스토리지 연동 기능 없음
- S3 업로드가 필요한 경우 워크플로우 외부에서 수동 처리

### 필요성
- 생성 결과물의 클라우드 백업 및 공유 자동화
- 외부 서비스(웹앱, API 등)에서 생성 결과물에 즉시 접근
- 배치 생성 파이프라인에서 결과물 자동 수집

### 기술 기반
- 기존 `SaveFileNode`의 입력 패턴(IMAGE + STRING, INPUT_IS_LIST) 재활용
- AWS SDK `boto3`를 통한 S3 연동
- 인증 정보는 설정 파일로 분리하여 보안 유지

---

## 🎯 Goals & Objectives

### Primary Goals

> [!success] 목표
> 1. ComfyUI 워크플로우 내에서 S3 업로드를 원스텝으로 처리
> 2. 기존 `Save Text Image` 노드와 동일한 입력 인터페이스 제공
> 3. AWS 인증 정보를 안전하게 관리 (워크플로우 JSON에 노출 방지)

### Success Metrics

> [!success] 성공 지표
> - 노드 연결 후 실행 시 S3에 파일이 정상 업로드됨
> - 이미지(PNG) + 텍스트(TXT) 동시 업로드 지원
> - 인증 실패/네트워크 오류 시 명확한 에러 메시지 제공

---

## 👤 User Stories & Use Cases

### Persona: ComfyUI 워크플로우 사용자

> [!example] User Stories
> | ID | User Story | Priority |
> |----|-----------|----------|
> | US-1 | 사용자로서, 생성된 이미지를 S3에 자동 업로드하고 싶다. 수동 업로드 과정을 없애기 위해. | Must |
> | US-2 | 사용자로서, 이미지와 함께 프롬프트 텍스트도 S3에 같이 올리고 싶다. 나중에 어떤 프롬프트로 생성했는지 추적하기 위해. | Must |
> | US-3 | 사용자로서, AWS 키를 설정 파일에 한 번만 넣어두고 싶다. 매번 노드에 입력하거나 워크플로우에 노출되지 않도록. | Must |
> | US-4 | 사용자로서, 업로드 실패 시 무엇이 잘못되었는지 알고 싶다. 빠르게 문제를 해결하기 위해. | Must |
> | US-5 | 사용자로서, S3의 원하는 경로에 파일을 정리하고 싶다. 버킷/접두어/파일명을 지정하여. | Should |

### Use Case: 기본 워크플로우

```
1. 사용자가 이미지 생성 워크플로우를 구성
2. VAEDecode 출력을 Save Text Image + Upload to S3에 병렬 연결
3. Upload to S3 노드에 bucket, path_prefix 설정
4. 워크플로우 실행
5. 로컬에 저장됨과 동시에 S3에 업로드됨
```

---

## 📐 Requirements

### Functional Requirements

> [!danger] Must-Have `P0`
> | ID | 요구사항 | 상세 |
> |----|---------|------|
> | FR-1 | 이미지 업로드 | IMAGE tensor를 PNG로 변환하여 S3에 업로드 |
> | FR-2 | 텍스트 업로드 | STRING을 TXT 파일로 S3에 업로드 |
> | FR-3 | 설정 파일 인증 | `s3_config.json`에서 AWS 인증 정보 로드 |
> | FR-4 | 환경변수 fallback | 설정 파일 없을 시 환경변수(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)로 fallback |
> | FR-5 | 파일명 생성 | 타임스탬프 기반 고유 파일명: `{prefix}_{YYYYMMDD_HHmmss}.{ext}` |
> | FR-6 | 에러 처리 | 인증 실패, 버킷 오류, 네트워크 오류 시 명확한 메시지 |
> | FR-7 | 배치 지원 | INPUT_IS_LIST로 다중 이미지/텍스트 동시 업로드 |

> [!warning] Should-Have `P1`
> | ID | 요구사항 | 상세 |
> |----|---------|------|
> | FR-8 | 리전 설정 | `s3_config.json`에서 region 지정 가능 |
> | FR-9 | Content-Type 설정 | 이미지는 `image/png`, 텍스트는 `text/plain; charset=utf-8` |

> [!info] Could-Have `P2`
> | ID | 요구사항 | 상세 |
> |----|---------|------|
> | FR-10 | 업로드 결과 로그 | 업로드된 S3 URI를 콘솔에 출력 |

### Non-Functional Requirements

| 항목 | 요구사항 |
|------|---------|
| 보안 | 인증 정보가 워크플로우 JSON에 포함되지 않을 것 |
| 보안 | `s3_config.json`은 `.gitignore`에 추가 |
| 성능 | 이미지 1장 업로드 시 S3 API 호출 외 추가 지연 없을 것 |
| 호환성 | 기존 `SaveFileNode`과 동일한 IMAGE/STRING 입력 수용 |
| 의존성 | `boto3` 패키지 추가 (requirements.txt) |

> [!caution] Technical Constraints
> | 항목 | 제약 |
> |------|------|
> | 패키지 | 기존 `comfyui-ixiworks-tools`에 추가 |
> | 파일 | `s3_nodes.py` 신규 생성 |
> | 등록 | `__init__.py`에서 import 및 NODE_CLASS_MAPPINGS 병합 |
> | 노드 카테고리 | `IXIWORKS/Utils` |
> | Python | 3.10+ |
> | IMAGE 형식 | `torch.Tensor[B, H, W, C]`, float32, 0.0~1.0 |

---

## 🎨 Design Considerations

### 노드 인터페이스

> [!tip] INPUT_TYPES 설계
> ```python
> # INPUT_TYPES
> {
>     "required": {
>         "bucket": ("STRING", {"default": ""}),
>         "path_prefix": ("STRING", {"default": ""}),
>         "filename_prefix": ("STRING", {"default": "file"}),
>     },
>     "optional": {
>         "text": ("STRING", {"forceInput": True}),
>         "image": ("IMAGE",),
>     }
> }
>
> # Node attributes
> INPUT_IS_LIST = True
> RETURN_TYPES = ()
> OUTPUT_NODE = True
> FUNCTION = "upload"
> CATEGORY = "IXIWORKS/Utils"
> ```

### 인증 설정 파일 형식

```json
{
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "...",
    "region": "ap-northeast-2"
}
```

### 파일명 패턴

```
{filename_prefix}_{YYYYMMDD_HHmmss}_{index}.png
{filename_prefix}_{YYYYMMDD_HHmmss}_{index}.txt
```

- 타임스탬프로 고유성 보장 (S3 ListObjects 불필요)
- 배치 내 순서는 `_{index}` 로 구분 (0부터)
- 예: `render_20260311_143052_0.png`, `render_20260311_143052_0.txt`

### S3 Key 구조

```
{path_prefix}/{filename_prefix}_{timestamp}_{index}.{ext}

예시:
outputs/comfyui/render_20260311_143052_0.png
outputs/comfyui/render_20260311_143052_0.txt
```

---

## 🏗️ Implementation Approach

### 파일 구조 변경

```
comfyui-ixiworks-tools/
├── __init__.py              ← S3 노드 import 추가
├── s3_nodes.py              ← 신규: S3UploadNode 클래스
├── s3_config.json           ← 사용자 생성 (gitignore 대상)
├── .gitignore               ← s3_config.json 추가
└── requirements.txt         ← boto3 추가
```

### 구현 단계

| 단계 | 작업 | 산출물 |
|------|------|--------|
| 1 | S3 설정 로드 함수 구현 | `_load_s3_config()` |
| 2 | S3UploadNode 클래스 구현 | `s3_nodes.py` |
| 3 | 노드 등록 | `__init__.py` 수정 |
| 4 | 의존성 추가 | `requirements.txt` 수정 |
| 5 | gitignore 추가 | `.gitignore` 수정 |

### 핵심 로직 흐름

```
upload() 호출
  ├─ s3_config.json 로드 (없으면 환경변수)
  ├─ boto3.client('s3') 생성
  ├─ 타임스탬프 생성
  ├─ 이미지 처리 (있는 경우)
  │   ├─ List[Tensor[B,H,W,C]] → List[Tensor[H,W,C]] flatten
  │   ├─ 각 이미지: tensor → numpy → PIL → BytesIO (PNG)
  │   └─ s3.put_object(Bucket, Key, Body, ContentType)
  ├─ 텍스트 처리 (있는 경우)
  │   └─ s3.put_object(Bucket, Key, Body, ContentType)
  └─ 결과 로그 출력
```

---

## ⚠️ Risks & Mitigation

> [!warning] 리스크 분석
> | 리스크 | 영향 | 완화 방안 |
> |--------|------|----------|
> | AWS 키 유출 | 보안 침해 | 설정 파일 분리 + gitignore |
> | boto3 미설치 | 노드 로드 실패 | lazy import + 명확한 설치 안내 메시지 |
> | 네트워크 불안정 | 업로드 실패 | 1회 재시도 + 에러 메시지 |
> | 대용량 이미지 배치 | 업로드 지연 | 순차 업로드, 진행 로그 출력 |

---

## ❓ Open Questions

> [!question] Q1: UI 표시
> 업로드 성공/실패 결과를 UI에 표시할 필요가 있는가?
> - **상태**: Open

> [!question] Q2: 다른 클라우드 스토리지
> S3 외 다른 클라우드 스토리지(GCS, Azure Blob) 지원 필요성?
> - **상태**: 현재 불필요

> [!question] Q3: URL 출력
> 업로드 후 S3 URL을 STRING으로 출력하여 후속 노드에 전달할 필요?
> - **상태**: 현재 불필요 (OUTPUT_NODE)

---

## 📎 Appendix

### 참고 코드: SaveFileNode 패턴

```python
class SaveFileNode:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subfolder": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "file"}),
            },
            "optional": {
                "text": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "IXIWORKS/Utils"
```

### 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| boto3 | latest | AWS S3 SDK |
