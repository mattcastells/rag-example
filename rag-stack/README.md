# rag-stack

Repositorio monorepo que expone un servicio Retrieval Augmented Generation (RAG) listo para producción empleando FastAPI, Postgres y pgvector. El stack está diseñado para operar con modelos Claude de Anthropic, embeddings open-source BAAI/bge-m3 y opciones híbridas adicionales.

## Características principales

- **API FastAPI** con autenticación por API Key y endpoints `/healthz`, `/ingest` y `/ask`.
- **Ingesta** de documentos PDF, Markdown, texto plano y HTML desde disco o vía endpoint multipart.
- **Vector store** en Postgres con extensión `pgvector` y migraciones administradas con Alembic.
- **Embeddings** multilingües con `sentence-transformers` (BAAI/bge-m3) con interfaz extensible.
- **Retriever** con filtros por metadatos y ACL. Opcionalmente híbrido con BM25 y re-rankers BAAI.
- **LLM** de respuesta usando los SDK oficiales de Anthropic (Claude) u OpenAI (ChatGPT), seleccionable por request.
- **Observabilidad**: logs estructurados, latencias y estimación de costos.
- **Docker Compose** para ambiente reproducible y Makefile para tareas comunes.

## Requisitos

- Docker 24+
- Docker Compose v2
- Make 4+

## Puesta en marcha rápida

1. Copiar variables de entorno y completar credenciales:

   ```bash
   cp .env.example .env
   # Editar .env y colocar ANTHROPIC_API_KEY y/o OPENAI_API_KEY
   ```

2. Levantar los servicios y aplicar migraciones:

   ```bash
   make up
   make migrate
   ```

3. Ingestar documentos locales (colocar archivos en `./data/docs`):

   ```bash
   make ingest-local
   ```

4. Probar la API (Claude por defecto):

   ```bash
   curl -H "x-api-key: dev-key" "http://localhost:8000/ask?q=Cómo%20configurar%20el%20pipeline&repo=company&tag=v1&k=8"
   ```

   Para usar OpenAI en una petición específica agrega `provider=openai`:

   ```bash
   curl -H "x-api-key: dev-key" "http://localhost:8000/ask?q=Cómo%20configurar%20el%20pipeline&repo=company&tag=v1&k=8&provider=openai"
   ```

   En el cuerpo de un POST utiliza `{"provider": "openai"}` dentro del JSON.

## Arquitectura

- `backend/app`: servicio FastAPI, configuración, dependencias, rutas, lógica de negocio.
- `backend/app/ingest`: loaders, splitter y pipeline de ingesta (CLI y endpoint).
- `backend/app/rag`: módulos de embeddings, retrieval, rerank, prompts y servicio principal.
- `backend/app/routes`: routers para health, ingest y ask.
- `backend/app/migrations`: configuración Alembic y versiones.
- `backend/tests`: pruebas con pytest que mockean LLM y embeddings.
- `backend/scripts`: utilidades de bootstrap y arranque.

## Flujo RAG

1. **Ingesta**: los documentos se cargan desde disco o vía API, se convierten a texto, se fragmentan en chunks (800 tokens, overlap 120) y se generan embeddings con `BAAI/bge-m3`.
2. **Almacenamiento**: los chunks se guardan en `rag_chunks` con metadatos, ACL y embeddings en Postgres/pgvector.
3. **Consulta**: `/ask` aplica filtros por repo, tag y ACL, ejecuta búsqueda vectorial (más híbrido opcional) y re-ranking configurable.
4. **Generación**: se construye un prompt controlado y se invoca el LLM configurado (Claude u OpenAI) vía sus SDKs oficiales.
5. **Respuesta**: se retorna texto sintetizado con citas a las fuentes relevantes.

## Selección de proveedor LLM

- Define el proveedor por defecto en `.env` con `DEFAULT_LLM_PROVIDER` (`claude` u `openai`).
- Configura las credenciales correspondientes (`ANTHROPIC_API_KEY` y/o `OPENAI_API_KEY`).
- Sobrescribe el proveedor por request usando el query param `provider` (GET) o el campo JSON `provider` (POST) en el endpoint `/ask`.
- Los costos estimados se calculan según los valores configurados en las variables `*_COST_PER_1K`.

## Observabilidad y seguridad

- Logs estructurados en formato JSON.
- Middleware de autenticación por API Key (`x-api-key`).
- Rate limiting en memoria configurable.
- Métricas de latencia total, tiempo de retrieval, tokens y costo estimado.
- Auditoría de contexto utilizado.

## Scripts y comandos

- `make up`: levanta Docker Compose.
- `make down`: detiene servicios y elimina contenedores.
- `make migrate`: aplica migraciones Alembic.
- `make ingest-local`: ejecuta la CLI de ingesta en los archivos de `./data/docs`.
- `make test`: ejecuta pytest.

## Roadmap

- Integración opcional con Elastic/OpenSearch para BM25 de alta escala.
- Gestión de ACL avanzada y rotación automática de claves API.
- Dashboard de observabilidad con Prometheus/Grafana.

## Licencia

MIT
