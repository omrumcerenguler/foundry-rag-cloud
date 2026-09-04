# Microsoft Foundry & Local AI Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?logo=streamlit&logoColor=white)](https://ceren-azure-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-gpt--4.1--mini-0078D4)](https://azure.microsoft.com/products/ai-services/openai-service)
![SQLite Vector](https://img.shields.io/badge/SQLite-Vector%20Retrieval-003B57?logo=sqlite&logoColor=white)
[![Pytest](https://img.shields.io/badge/Pytest-69%2F69%20Passed-2EA44F?logo=pytest&logoColor=white)](tests/)
[![MyPy](https://img.shields.io/badge/MyPy-Strict-1674B1)](https://mypy.readthedocs.io/)
[![Ruff](https://img.shields.io/badge/Ruff-Clean-D7FF64?logo=ruff&logoColor=111827)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/License-MIT-0B101B)](LICENSE)

**[Open the live application](https://ceren-azure-ai.streamlit.app)**

An enterprise-grade Grounded Retrieval-Augmented Generation (RAG) assistant for Microsoft Foundry and local AI systems engineering. It retrieves from a curated technical knowledge base, enforces citation-backed answers with safe fallbacks, and exposes real-time retrieval telemetry.

## English

### Technology Stack

| Area                      | Implementation                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language and runtime      | Python 3.10+; Python 3.13 is the tested local, CI, and container path.                                                                                |
| Frontend and UX           | Streamlit with a dark glassmorphism layout, focused conversation flow, and a native JavaScript clipboard bridge.                                      |
| Vector engine and storage | SQLite persistent storage, JSON-serialized vector embeddings, provenance metadata, and local cosine-similarity ranking.                               |
| Cloud AI and models       | Azure OpenAI `text-embedding-3-small` for 1536-dimensional embeddings and `gpt-4.1-mini` for grounded answer generation.                              |
| HTTP and configuration    | `httpx` API client, `python-dotenv` environment loading, and a Streamlit Secrets bridge for hosted deployments.                                       |
| Providers and API         | Foundry Local and Azure OpenAI adapters; FastAPI with Pydantic v2 contracts, API-key authentication, CORS policy controls, and structured errors.     |
| Observability and caching | Query latency, similarity confidence, match count, model identity, and in-memory cache HIT/MISS telemetry.                                            |
| Security and reliability  | Eight-query session quota, four-second cooldown, bounded context, validation, atomic ingestion replacement, and curated ingestion boundaries.         |
| Quality and DevSecOps     | 69 unit and integration tests; `pytest-cov` with an 85% coverage gate enforced in GitHub Actions CI; strict MyPy, Ruff, Flake8, Bandit, `pip-audit`, and Docker. |

### Architecture and Data Flow

```text
INGESTION
Curated TXT/Markdown documents
  -> Recursive document chunker
  -> Azure OpenAI embedding API / Foundry Local provider
  -> SQLite vector table with provenance metadata

RETRIEVAL AND GENERATION
User query
  -> Query embedding
  -> Local cosine-similarity ranking
  -> Top-K filtering and confidence threshold
  -> Strict grounding prompt with bounded context
  -> Azure OpenAI GPT generation / Foundry Local provider
  -> Citation-backed answer: docX.txt, chunk Y + telemetry
```

The bundled corpus is an internal engineering knowledge base covering Microsoft Foundry, offline inference, Python environments, RAG ingestion, SQLite vector retrieval, Apple Silicon compatibility, and a local RAG delivery plan.

### Key Engineering Capabilities

**Grounded retrieval and provenance**

- Documents are chunked deterministically with source file, chunk index, offset, and corpus metadata retained through retrieval.
- Responses are constrained to retrieved context. Empty, low-confidence, or uncited results return a controlled fallback rather than an unsupported answer.
- The citation inspector presents retrieved source IDs, similarity scores, relevance indicators, and matching passage text.

**Embedded vector retrieval**

- SQLite keeps the deployment footprint small while providing durable local storage for embeddings and metadata.
- A custom cosine-similarity engine ranks candidates locally, removing the need for a managed external vector database for this use case.
- Embeddings are validated for numerical integrity: NaN, Infinity, dimension-mismatched, and zero vectors are rejected before storage or retrieval.
- The architecture supports Azure OpenAI cloud models and the optional offline Foundry Local provider through stable provider ports.

**Clean architecture and extensibility**

- The RAG service uses constructor-based dependency injection over abstract embedding, chat, and vector-store ports, decoupling core orchestration from runtime implementations.
- `factory.py` applies the Factory pattern to assemble a complete LOCAL or AZURE_CLOUD service without embedding provider-specific logic in the core workflow.

**Observability and efficiency**

- Every response exposes latency, confidence, match count, model identity, and cache HIT/MISS state.
- The per-session cache keys normalized prompts and confidence thresholds, avoiding redundant provider calls for equivalent requests.
- A fixed 12,000-character context budget limits generation input while preserving the highest-ranked retrieved evidence.

**Application resilience**

- Eight questions per browser session and a four-second request cooldown protect the public demo from accidental burst traffic.
- Curated, pre-indexed documents create a controlled data boundary that reduces untrusted-content and prompt-injection exposure.
- Ingestion skips binary, null-byte, unreadable, and unsupported files and replaces the index atomically, preserving the previous usable index on failure.
- SQLite uses WAL journaling and a 30-second busy timeout for concurrent file access. Within one Python process, an `RLock` serializes access; multi-process coordination relies on SQLite file-level locking. Equal scores are resolved deterministically.
- Azure OpenAI requests parse 429 `Retry-After` headers, cap the delay at 10 seconds, and retry eligible 429/5xx failures with bounded exponential backoff.
- Container images run as a non-root user. Docker Compose applies read-only service filesystems, restricted `tmpfs` mounts, health-based startup, and CPU/memory limits.

### Evaluation and Operations

- `evaluate.py` benchmarks grounded retrieval with Precision@3, Mean Reciprocal Rank (MRR), citation grounding, and latency measurements.
- `main.py` provides a unified CLI for ingestion, single-query retrieval, interactive chat, index health checks, and evaluation runs.
- `healthcheck.py` supplies the container readiness probe, while `export_openapi.py` generates the OpenAPI contract artifact.

### Focused User Experience

- The dark Streamlit interface includes a five-category engineering question library, bilingual sidebar guidebook, conversation download, and source inspection.
- The onboarding banner and suggested questions disappear after the first interaction so the conversation becomes the primary workspace.
- The copy action runs in a client-side JavaScript component without a Streamlit rerun, uses Clipboard API and `execCommand` fallback paths, and reports `Copied!` feedback.

### Repository Structure

```text
.
├── core/                    # Domain models, provider/store ports, chunking, grounded RAG service
├── providers/               # Azure OpenAI HTTP adapter and optional Foundry Local adapter
├── stores/                  # SQLite vector persistence, provenance metadata, cosine-similarity engine
├── tests/                   # 69 unit, integration, and hardening test cases across nine suites
├── api.py                   # FastAPI contract, authentication, CORS, and lifecycle management
├── app.py                   # Streamlit chat experience, telemetry, caching, and clipboard component
├── config.py                # Validated environment settings and Streamlit Secrets bridge
├── factory.py               # Provider, vector store, and RAG service assembly
├── ingestion.py             # Recursive corpus validation, chunking, embedding, and atomic replacement
├── main.py                  # Unified CLI for ingest, query, chat, health, and evaluation
├── evaluate.py              # Retrieval and citation-quality evaluation engine
├── healthcheck.py           # Container readiness probe
├── export_openapi.py        # OpenAPI specification exporter
├── Dockerfile               # Production API container image
├── Dockerfile.streamlit     # Standalone Streamlit container image
└── docker-compose.yml       # API/UI deployment, storage, health, and resource policies
```

### Quickstart

Prerequisites: Python 3.10+, an Azure OpenAI endpoint and API key for `AZURE_CLOUD` mode, or a configured Foundry Local runtime for `LOCAL` mode.

```sh
git clone git@github.com:omrumcerenguler/foundry-rag-cloud.git
cd foundry-rag-cloud
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` for Azure OpenAI:

```dotenv
RAG_MODE=AZURE_CLOUD
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=replace-with-your-secret
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_EMBEDDING_DIMENSION=1536
RAG_DATABASE_PATH=stores/rag.db
# Leave RAG_API_URL unset for direct local Streamlit mode.
```

Index the bundled corpus, then launch the UI:

```sh
python main.py ingest
streamlit run app.py
```

For FastAPI mode, start `uvicorn api:app --host 127.0.0.1 --port 8000` and set `RAG_API_URL=http://127.0.0.1:8000` before starting Streamlit.

### Testing and Quality Assurance

Run the local verification suite from an activated virtual environment:

```sh
pytest -v
mypy --strict app.py
ruff check .
```

The suite contains 69 passing tests across API behavior, Azure provider handling, chunking, CLI commands, evaluation, security hardening, ingestion, grounded generation, and SQLite storage.

### Test Suite Matrix

| Suite | Explicit verification |
| --- | --- |
| `test_api.py` | Health/readiness states, grounded query payloads, structured 422 errors, API-key enforcement, 429 mapping, atomic ingest summaries, and degraded index health. |
| `test_azure_provider.py` | Endpoint normalization, `Retry-After` handling, 429 exhaustion, and bounded 5xx retries. |
| `test_chunker.py` | Deterministic chunk boundaries and metadata plus empty, binary, and invalid-text handling. |
| `test_cli.py` | CLI flag aliases and user-friendly interactive-chat interruption handling. |
| `test_evaluate.py` | Structured evaluation behavior when a provider mode is unavailable. |
| `test_hardening.py` | Streamlit Secrets, malformed input rejection, safe citation fallback, NaN/Inf/zero-vector rejection, corrupted SQLite data, provider failure sanitization, and evaluation edge cases. |
| `test_ingestion.py` | Null-byte/binary skipping, recursive relative provenance, and preservation of the existing index after provider or vector-dimension failure. |
| `test_service.py` | Grounded generation, confidence fallback, empty-query rejection, uncited-answer fallback, and context-budget enforcement. |
| `test_store.py` | Cosine ranking, atomic rollback, zero-norm and dimension validation, WAL/busy-timeout configuration, deterministic tie-breaking, metadata health, and missing provenance handling. |

---

## 🇹🇷 Türkçe Dokümantasyon

Microsoft Foundry & Local AI Assistant, Microsoft Foundry ve yerel yapay zeka sistemleri mühendisliği için geliştirilmiş, kurumsal nitelikte bir Grounded RAG uygulamasıdır. Düzenlenmiş teknik bilgi tabanından kaynak getirir, kaynakla desteklenmeyen yanıtlar için güvenli geri dönüş uygular ve sorgu sürecini gerçek zamanlı telemetri ile görünür kılar.

### Teknoloji Altyapısı

| Alan                           | Uygulama                                                                                                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dil ve çalışma ortamı          | Python 3.10+; yerel geliştirme, CI ve container ortamında doğrulanan sürüm Python 3.13'tür.                                                                                  |
| Arayüz ve kullanıcı deneyimi   | Koyu cam efektli Streamlit arayüzü, sohbet odaklı akış ve tarayıcı tarafında çalışan JavaScript pano kopyalama köprüsü.                                                      |
| Vektör motoru ve depolama      | Kalıcı SQLite depolama, JSON olarak serileştirilmiş vektör embedding'leri, provenance metadatası ve yerel cosine similarity sıralaması.                                      |
| Bulut yapay zekası ve modeller | 1536 boyutlu embedding'ler için Azure OpenAI `text-embedding-3-small`; kaynaklı yanıt üretimi için `gpt-4.1-mini`.                                                           |
| HTTP ve yapılandırma           | `httpx` API istemcisi, `python-dotenv` ile ortam değişkeni yükleme ve barındırılan kurulumlar için Streamlit Secrets köprüsü.                                                |
| Sağlayıcılar ve API            | Foundry Local ve Azure OpenAI adaptörleri; Pydantic v2 sözleşmeleri, API anahtarı doğrulaması, CORS politikaları ve yapılandırılmış hata yanıtlarıyla FastAPI servis sınırı. |
| Gözlemlenebilirlik ve önbellek | Sorgu gecikmesi, benzerlik güveni, eşleşme sayısı, model bilgisi ve bellek içi Cache HIT/MISS telemetrisi.                                                                   |
| Güvenlik ve dayanıklılık       | Sekiz sorguluk oturum kotası, dört saniyelik bekleme, sınırlı bağlam, girdi doğrulama, atomik indeks güncellemesi ve kontrollü veri sınırı.                                  |
| Kalite ve DevSecOps            | 69 birim ve entegrasyon testi; GitHub Actions CI içinde zorlanan `%85` kod kapsamı eşiğiyle `pytest-cov`, strict MyPy, Ruff, Flake8, Bandit, `pip-audit` ve Docker.          |

### Mimari ve Veri Akışı

```text
İÇERİ AKTARMA
Düzenlenmiş TXT/Markdown belgeleri
  -> Özyinelemeli belge parçalayıcı
  -> Azure OpenAI embedding API / Foundry Local sağlayıcısı
  -> Provenance metadatası içeren SQLite vektör tablosu

GETİRME VE YANIT ÜRETİMİ
Kullanıcı sorgusu
  -> Sorgu embedding'i
  -> Yerel cosine similarity sıralaması
  -> Top-K filtreleme ve güven eşiği
  -> Sınırlı bağlamlı, katı kaynaklama istemi
  -> Azure OpenAI GPT üretimi / Foundry Local sağlayıcısı
  -> Kaynaklı yanıt: docX.txt, chunk Y + telemetri
```

Birlikte gelen bilgi tabanı Microsoft Foundry, çevrim dışı çıkarım, Python ortamları, RAG içerik alma, SQLite vektör getirme, Apple Silicon uyumluluğu ve yerel RAG teslim planı konularını kapsar.

### Temel Mühendislik Yetenekleri

**Kaynağa dayalı getirme ve izlenebilirlik**

- Belgeler; kaynak dosya, parça indeksi, karakter konumu ve corpus metadatası korunarak deterministik şekilde parçalanır.
- Yanıtlar getirilen bağlamla sınırlandırılır. Boş, düşük güvenli veya kaynak içermeyen sonuçlarda desteklenmeyen yanıt üretmek yerine kontrollü geri dönüş yapılır.
- Kaynak görünümü; kaynak kimliğini, benzerlik skorunu, uygunluk göstergesini ve ilgili metin parçasını gösterir.

**Gömülü vektör getirme**

- SQLite, embedding ve metadata için kalıcı yerel depolama sağlarken kurulum yükünü düşük tutar.
- Özel cosine similarity motoru adayları yerelde sıralar; bu kullanım senaryosunda yönetilen harici bir vektör veritabanına ihtiyaç duyulmaz.
- Embedding'ler sayısal bütünlük açısından doğrulanır; NaN, sonsuz, boyutu uyumsuz ve sıfır vektörler depolama veya getirme aşamasına ulaşmadan reddedilir.
- Sağlayıcı soyutlaması, Azure OpenAI bulut modelleri ile isteğe bağlı Foundry Local çalışma zamanını destekler.

**Temiz mimari ve genişletilebilirlik**

- RAG servisi, soyut embedding, sohbet ve vektör deposu portları üzerinde yapıcı tabanlı bağımlılık enjeksiyonu kullanır; böylece çekirdek orkestrasyon somut çalışma zamanı uygulamalarından ayrılır.
- `factory.py`, sağlayıcıya özgü mantığı çekirdek iş akışına taşımadan eksiksiz bir LOCAL veya AZURE_CLOUD servisi oluşturmak için Factory desenini uygular.

**Gözlemlenebilirlik ve verimlilik**

- Her yanıtta gecikme, güven değeri, eşleşme sayısı, model kimliği ve Cache HIT/MISS durumu sunulur.
- Oturum içi önbellek, normalize edilmiş istem ve güven eşiğini anahtar olarak kullanarak eşdeğer isteklerde sağlayıcı çağrılarını önler.
- 12.000 karakterlik sabit bağlam bütçesi, en yüksek sıralı kanıtı korurken üretim girdisini sınırlar.

**Uygulama dayanıklılığı**

- Tarayıcı oturumu başına sekiz sorgu ve dört saniyelik istek bekleme süresi, genel kullanıma açık demoyu ani istek yoğunluğuna karşı korur.
- Önceden indekslenen düzenlenmiş belgeler, güvenilmeyen içerik ve prompt injection riskini azaltan kontrollü bir veri sınırı oluşturur.
- İçeri aktarma süreci ikili, null byte içeren, okunamayan ve desteklenmeyen dosyaları atlar; hata durumunda önceki kullanılabilir indeksi koruyacak şekilde atomik güncelleme yapar.
- SQLite, eş zamanlı dosya erişimi için WAL günlük modu ve 30 saniyelik yoğunluk zaman aşımı kullanır. Tek Python işlemi içinde `RLock` erişimi serileştirir; çok işlemli eşgüdüm SQLite'ın dosya düzeyindeki kilitlemesine dayanır. Eşit skorlar deterministik biçimde sıralanır.
- Azure OpenAI istekleri, 429 yanıtlarındaki `Retry-After` başlığını işler, gecikmeyi en fazla 10 saniye ile sınırlar ve uygun 429/5xx hatalarını sınırlı üstel geri çekilmeyle yeniden dener.
- Container imajları root olmayan kullanıcıyla çalışır. Docker Compose; salt okunur servis dosya sistemleri, kısıtlı `tmpfs` bağlamaları, sağlık kontrolüne bağlı başlatma ve CPU/bellek sınırları uygular.

### Değerlendirme ve Operasyonlar

- `evaluate.py`, kaynaklı getirme kalitesini Precision@3, Ortalama Karşılıklı Sıra (MRR), kaynaklama doğruluğu ve gecikme ölçümleriyle değerlendirir.
- `main.py`, içeri aktarma, tekil sorgu, etkileşimli sohbet, indeks sağlık kontrolü ve değerlendirme çalıştırmaları için birleşik bir CLI sunar.
- `healthcheck.py` container hazır olma denetimini sağlar; `export_openapi.py` ise OpenAPI sözleşmesi çıktısını üretir.

### Odaklı Kullanıcı Deneyimi

- Koyu temalı Streamlit arayüzü; beş kategorili mühendislik soru kütüphanesi, iki dilli yan panel rehberi, sohbet indirme ve kaynak inceleme işlevleri içerir.
- İlk etkileşimden sonra karşılama alanı ve önerilen sorular gizlenir; böylece sohbet çalışma alanı ön plana çıkar.
- Kopyalama işlemi Streamlit yeniden çalıştırması oluşturmadan tarayıcı tarafındaki JavaScript bileşeninde yürür; Clipboard API ve `execCommand` geri dönüşünü kullanır, `Copied!` bildirimi gösterir.

### Dizin Yapısı

```text
.
├── core/                    # Alan modelleri, sağlayıcı/depo portları, parçalama ve kaynaklı RAG servisi
├── providers/               # Azure OpenAI HTTP adaptörü ve isteğe bağlı Foundry Local adaptörü
├── stores/                  # SQLite vektör kalıcılığı, provenance metadatası ve cosine similarity motoru
├── tests/                   # Dokuz pakete dağıtılmış 69 birim, entegrasyon ve sertleştirme testi
├── api.py                   # FastAPI sözleşmesi, kimlik doğrulama, CORS ve yaşam döngüsü yönetimi
├── app.py                   # Streamlit sohbet arayüzü, telemetri, önbellek ve pano kopyalama bileşeni
├── config.py                # Doğrulanmış ortam ayarları ve Streamlit Secrets köprüsü
├── factory.py               # Sağlayıcı, vektör deposu ve RAG servisi oluşturma katmanı
├── ingestion.py             # Özyinelemeli corpus doğrulama, parçalama, embedding ve atomik güncelleme
├── main.py                  # İçeri aktarma, sorgu, sohbet, sağlık ve değerlendirme için birleşik CLI
├── evaluate.py              # Getirme ve kaynak kalitesi değerlendirme motoru
├── healthcheck.py           # Container hazır olma denetimi
├── export_openapi.py        # OpenAPI sözleşmesi dışa aktarıcısı
├── Dockerfile               # Üretim API container imajı
├── Dockerfile.streamlit     # Bağımsız Streamlit container imajı
└── docker-compose.yml       # API/UI dağıtımı, depolama, sağlık ve kaynak politikaları
```

### Hızlı Başlangıç

Gereksinimler: Python 3.10+, `AZURE_CLOUD` modu için Azure OpenAI endpoint'i ve API anahtarı veya `LOCAL` modu için yapılandırılmış Foundry Local çalışma zamanı.

```sh
git clone git@github.com:omrumcerenguler/foundry-rag-cloud.git
cd foundry-rag-cloud
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` dosyasını Azure OpenAI için yapılandırın:

```dotenv
RAG_MODE=AZURE_CLOUD
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=replace-with-your-secret
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_EMBEDDING_DIMENSION=1536
RAG_DATABASE_PATH=stores/rag.db
# Doğrudan yerel Streamlit modu için RAG_API_URL tanımlamayın.
```

Birlikte gelen corpus'u indeksleyin ve arayüzü başlatın:

```sh
python main.py ingest
streamlit run app.py
```

FastAPI modunda önce `uvicorn api:app --host 127.0.0.1 --port 8000` komutunu çalıştırın; ardından Streamlit'i başlatmadan önce `RAG_API_URL=http://127.0.0.1:8000` değişkenini tanımlayın.

### Test ve Kalite Güvencesi

Etkin sanal ortamdayken yerel doğrulama komutları:

```sh
pytest -v
mypy --strict app.py
ruff check .
```

69 test; API davranışı, Azure sağlayıcı işlemleri, parçalayıcı, CLI komutları, değerlendirme, güvenlik sertleştirmesi, içerik alma, kaynaklı yanıt üretimi ve SQLite depolama katmanını kapsar.

### Test Paketi Matrisi

| Paket | Açıkça doğrulanan davranışlar |
| --- | --- |
| `test_api.py` | Sağlık/hazır olma durumları, kaynaklı sorgu yanıtları, yapılandırılmış 422 hataları, API anahtarı denetimi, 429 eşlemesi, atomik içeri aktarma özeti ve bozulmuş indeks sağlığı. |
| `test_azure_provider.py` | Uç nokta normalizasyonu, `Retry-After` işleme, 429 sınırının tükenmesi ve sınırlı 5xx yeniden denemeleri. |
| `test_chunker.py` | Deterministik parça sınırları ve metadata ile boş, ikili ve geçersiz metin işleme davranışı. |
| `test_cli.py` | CLI bayrak kısaltmaları ve etkileşimli sohbet kesintilerinin kullanıcı dostu yönetimi. |
| `test_evaluate.py` | Sağlayıcı modu kullanılamadığında yapılandırılmış değerlendirme davranışı. |
| `test_hardening.py` | Streamlit Secrets, hatalı girdi reddi, güvenli kaynak fallback'i, NaN/Inf/sıfır vektör reddi, bozuk SQLite verisi, sağlayıcı hata sterilizasyonu ve değerlendirme sınır durumları. |
| `test_ingestion.py` | Null byte/ikili dosya atlama, özyinelemeli göreli provenance ve sağlayıcı ya da vektör boyutu hatasında mevcut indeksin korunması. |
| `test_service.py` | Kaynaklı yanıt üretimi, güven eşiği fallback'i, boş sorgu reddi, kaynaksız yanıt fallback'i ve bağlam bütçesi uygulaması. |
| `test_store.py` | Cosine sıralama, atomik geri alma, sıfır norm ve boyut doğrulaması, WAL/yoğunluk zaman aşımı ayarları, deterministik eşitlik çözümü, metadata sağlığı ve eksik provenance işleme. |
