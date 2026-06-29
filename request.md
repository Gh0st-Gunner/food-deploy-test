# Calorie App — Scalable Architecture Summary

# Goal

Scale AI calorie estimation pipeline to:
- 10,000+ users
- low latency
- GPU-efficient inference
- async processing

---

# Core Problems

Current:
- ~30s/image
- synchronous pipeline
- heavy GPU usage
- SAM2 + Grounding DINO bottlenecks

Main risks:
- GPU memory fragmentation
- model loading overhead
- blocking HTTP requests
- sequential execution

---

# Recommended Architecture

```text
Frontend
  ↓
FastAPI Gateway
  ↓
Task Queue / Workflow
  ↓
GPU Workers
  ↓
Aggregator
  ↓
PostgreSQL + Redis + S3
Recommended Tech Stack
Layer	Stack
Frontend	Next.js / React Native
Backend API	FastAPI
Workflow	Temporal
Queue	Redis + Celery
GPU Serving	NVIDIA Triton
Database	PostgreSQL
Cache	Redis
Storage	S3 / Cloudflare R2
Infra	Kubernetes
Monitoring	Prometheus + Grafana
Critical Design Decisions
1. Async Jobs

DO NOT:

POST /analyze → wait 30s

DO:

POST /analyze → return job_id

Then:

polling
websocket updates
2. Parallelize Pipeline

Current:

classification
→ USDA
→ DINO
→ SAM2

Better:

classification
 ├── USDA lookup
 └── ingredient detection

Run:

USDA
DINO
SAM2
in parallel.
3. Separate GPU Workers

DO NOT use one monolith service.

Use:

classification-workers
dino-workers
sam-workers
nutrition-workers

Benefits:

independent scaling
lower VRAM fragmentation
better autoscaling
4. Keep Models Hot

BAD:

load_model()
infer()
destroy_model()

GOOD:

load_once()
reuse_forever()
5. Batch GPU Inference

Instead of:

1 image → 1 inference

Use:

8 images → 1 batch

Use:

Triton Inference Server

Benefits:

higher throughput
lower GPU idle time
Workflow Recommendation

Use Temporal.

Why:

retries
distributed workflows
durable execution
state persistence
observability
Suggested Pipeline
upload
  ↓
classification
  ├── USDA lookup
  └── ingredient detection
         ├── Grounding DINO
         └── SAM2
                ↓
         ingredient nutrients
                ↓
         portion estimation
                ↓
         aggregation
                ↓
             result
Caching

Cache:

USDA responses
ingredient nutrients
class mappings

Use:

Redis
PostgreSQL cache table

This reduces:

API latency
external API costs
Storage

Store in S3/R2:

uploaded images
masks
overlays
depth maps

DO NOT store images in PostgreSQL.

Infrastructure Plan
MVP
FastAPI
Redis
Celery
PostgreSQL
1–2 GPU servers
S3

Deploy on:

RunPod
Lambda Labs
Vast.ai
Scale Phase

Add:

Temporal
Triton
autoscaling
Prometheus
Grafana
websocket updates
Production Scale

Add:

Kubernetes
GPU node pools
autoscaling
distributed tracing
Kubernetes Deployment

Separate deployments:

api
classification
dino
sam
nutrition

Scale independently.

Monitoring

Use:

Prometheus
Grafana
OpenTelemetry

Track:

GPU usage
queue size
inference latency
API latency
worker failures
Optimization Priority
Async jobs
Parallel execution
Hot-loaded models
GPU batching
Redis caching
Separate GPU workers
Triton serving
Temporal workflows
Kubernetes autoscaling
Fast/Accurate modes
Recommended Modes
Fast Mode
classification
+ nutrition lookup

Latency:

2–4s
Accurate Mode
DINO
+ SAM2
+ ingredient analysis

Latency:

10–20s

Default should be Fast Mode.

Final Recommended Stack
MVP
FastAPI
Redis
Celery
PostgreSQL
S3
RunPod GPUs
Production
FastAPI
Temporal
Triton
Kubernetes
Redis
PostgreSQL
S3/R2
Prometheus
Grafana
OpenTelemetry
Expected Result

With optimization:

30s → 5–10s perceived latency

Supports:

scalable inference
GPU efficiency
async workloads
production observability