# Clinical RAG Copilot — Backend (v2)

نسخة معدّلة من الباك اند عشان تتماشى مع **أجندة الـ AI Hackathon** (ITIDA × TIEC ×
Orange Digital Center × INSTANT) وموجّهات الفريق:

1. **الملفات بقت جوه السيستم مش اليوزر اللي يرفعها، وبتتفهرس تلقائيًا لوحدها.**
   الأدلة الطبية الرسمية (WHO / CDC / NICE / USPSTF) بتتحط في
   `data/sources/<project_id>/`، وبمجرد ما تتحط، السيرفر بيكتشفها ويعمله
   تسجيل + فهرسة (embedding) تلقائيًا من غير أي ضغطة زرار ولا API call —
   background scanner شغال طول الوقت (شوف `app/core/auto_ingest.py`).
2. الـ API اتبني بالكامل على الـ endpoints اللي الفريق حددها.
3. الـ pipeline اتبني حرفيًا زي الـ diagram اللي اتبعت:

```
User → Frontend → Backend API
     → Validate/Auth
     → Retrieve Context
     → Safety Threshold
     → Generation Model
     → Validate Answer/Citations
     → Save Logs
     → Return Answer or Refusal
```

بتلاقيه متطبق في `app/core/pipeline.py` وبيتستخدم من `app/routers/conversations.py`.

---

## Endpoints

| Method | Path | ملاحظات |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | readiness (بيتأكد من الداتابيز) |
| POST | `/api/v1/auth/register`, `/login`, `/me` | auth بسيط (JWT) — أول يوزر يتسجل بيبقى admin |
| POST | `/api/v1/projects` | إنشاء مشروع/نطاق سريري |
| GET | `/api/v1/projects` | لستة المشاريع |
| POST | `/api/v1/projects/{project_id}/documents` | **تسجيل يدوي (fallback)** — الأساسي بقى auto-ingest، شوف تحت |
| GET | `/api/v1/projects/{project_id}/documents` | لستة مستندات المشروع |
| POST | `/api/v1/documents/{document_id}/ingest` | يشغّل ingestion job (background) — admin فقط |
| GET | `/api/v1/documents/{document_id}/status` | حالة المستند + عدد الـ chunks |
| GET | `/api/v1/jobs/{job_id}` | تتبع تقدم الـ job (queued/running/succeeded/failed) |
| POST | `/api/v1/projects/{project_id}/conversations` | *إضافة* — لازمة عشان تبدأ محادثة قبل الرسايل |
| GET | `/api/v1/projects/{project_id}/conversations` | *إضافة* |
| POST | `/api/v1/conversations/{conversation_id}/messages` | بيشغل الـ pipeline كامل ويرجع إجابة/رفض |
| GET | `/api/v1/conversations/{conversation_id}/messages` | *إضافة* — تاريخ المحادثة |
| POST | `/api/v1/projects/{project_id}/retrieve` | debug: يرجع الـ chunks والسكورات من غير LLM |
| POST | `/api/v1/projects/{project_id}/evaluations` | Precision@K + Citation Accuracy + Unsupported-claim rate |

الـ endpoints اللي مكتوب جنبها *"إضافة"* مش كانت في اللستة اللي بعتها، لكن
النظام معملوش يشتغل من غيرهم (لازم تبدأ conversation قبل ما تبعت فيها رسايل، ولازم
تقدر تجيب تاريخ المحادثة). لو مش عايزهم قولي أشيلهم.

---

## إزاي مستندات النظام بتتضاف (Auto-Ingest — من غير أي زرار)

1. اعمل مشروع (`POST /api/v1/projects`) — بمجرد ما يتعمل، السيستم بيجهزله
   فولدر خاص بيه تلقائيًا: `data/sources/<project_id>/`.
2. حط ملف الـ PDF/TXT الرسمي **جوه فولدر المشروع ده بس**. مفيش أي API call
   مطلوب، ولا تسجيل، ولا ضغطة "Ingest".
3. في الخلفية، السيرفر شغّال عليه background thread بيدوّر كل
   `AUTO_INGEST_INTERVAL_SECONDS` ثانية (افتراضيًا 15) على فولدر كل مشروع.
   أول ما يلاقي ملف جديد، بيعمله تسجيل (Document) وفهرسة (chunking +
   embeddings) تلقائيًا من غير أي تدخل.
4. تقدر تتابع الحالة من `GET /api/v1/projects/{project_id}/documents` أو من
   واجهة المشروع نفسها — الحالة هتتغير من `queued` → `ingesting` →
   `ingested` لوحدها.

**ملحوظة:** الـ endpoints اليدوية (`POST .../documents` و
`POST .../documents/{id}/ingest`) لسه موجودة كـ fallback — مفيدة أساسًا لو
عايز تسجل مستند بـ `source_url` (رابط رسمي) بدل ما تنزّله يدوي، لكنها مش
الطريقة الأساسية بعد كده.

راجع `data/sources/README.md` لتفاصيل بنية الفولدرات.

---

## التشغيل

```bash
cp .env.example .env      # وحط GROQ_API_KEY بتاعك (من https://console.groq.com/keys)
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Docs تفاعلية: `http://127.0.0.1:8000/docs`

---

## بنية الملفات

```
app/
├── main.py                # FastAPI entrypoint
├── config.py               # thresholds + مسارات النظام
├── database.py / models.py / schemas.py / auth.py
├── core/
│   ├── chunking.py         # Ingestion + section-aware chunking + metadata schema
│   ├── embeddings.py
│   ├── vectorstore.py      # FAISS per-project index
│   ├── safety.py           # Input risk / confidence thresholds / unsupported-claim detection
│   ├── llm.py               # grounded generation prompt
│   ├── pipeline.py         # الـ orchestration بالكامل (الـ diagram)
│   └── jobs.py              # background ingestion runner
└── routers/
    ├── health.py, auth_router.py, projects.py
    ├── documents.py         # register / ingest / status / job
    ├── conversations.py     # messages -> pipeline -> logs
    ├── retrieve.py           # debug retrieval
    └── evaluations.py        # precision@k / citation accuracy / faithfulness
data/
├── sources/                # ملفات النظام الرسمية (مش user uploads)
└── vectorstores/            # فهرس FAISS منفصل لكل project
```

---

## حاجات محتاجة قرار منك/من الفريق

- **الفرونت اند القديم** (`rag-frontend`) بيكلم `/upload` و `/ask` القدام. محتاج
  تعديل عشان يستخدم الـ endpoints الجديدة (projects → documents → conversations →
  messages). لو عايزني أعدله قولي.
- الـ **unsupported-claim detection** حاليًا heuristic بسيط (word-overlap) —
  كافي كـ MVP لهاكاثون بس لو الوقت سمح ينفع نستبدله بـ NLI model حقيقي.
- الـ **similarity → confidence thresholds** في `config.py` قيم بادئة، لازم
  تتظبط على الـ eval set بتاعكم يوم 2/4.

---

## مصادر ضغط الدم الموثوقة

أُضيف كتالوج المصادر الرسمية في `data/hypertension_sources.json`. يغطي الكتالوج التعريف والتشخيص والقياس المنزلي، عوامل الخطورة والمضاعفات، تغييرات نمط الحياة، العلاج الدوائي، المتابعة والأهداف العلاجية، بالإضافة إلى ارتفاع ضغط الدم أثناء الحمل وما قبل تسمم الحمل. المصادر المختارة هي منظمة الصحة العالمية، ومراكز مكافحة الأمراض والوقاية منها CDC، وNICE، وإرشادات AHA/ACC لعام 2025.

لإضافة المصادر إلى مشروع موجود وتشغيل التحميل والفهرسة تلقائيًا، نفّذ من جذر المستودع:

```bash
python scripts/seed_hypertension_sources.py <PROJECT_ID>
```

الأداة idempotent؛ أي أنها تتخطى المصدر إذا كان مسجلًا بالفعل في المشروع نفسه، وتستخدم نفس مسار الـ ingestion الموجود في الـ API. صفحات HTML الرسمية تُنظَّف من JavaScript وCSS والعناصر غير النصية قبل تقسيمها، بينما تُحفظ أدلة PDF كما هي مع بيانات المصدر والرابط الرسمي داخل الـ metadata.

> هذه المصادر تدعم grounded answers ولا تجعل النظام بديلًا عن الطبيب. يجب أن تستمر قواعد الطوارئ والرفض الآمن، خصوصًا عند وجود قراءة شديدة الارتفاع مع ألم صدر أو ضيق تنفس أو أعراض عصبية، أو عند طلب تشخيص فردي أو تعديل دواء.

## الخصوصية وتشغيل نموذج الإجابة

المحادثات أصبحت معزولة حسب `user_id`: قائمة المحادثات، قراءة الرسائل، وإرسال رسالة جديدة لا تعمل إلا إذا كانت المحادثة مملوكة للمستخدم المصادق عليه. عند محاولة الوصول إلى محادثة مستخدم آخر يرجع الـAPI حالة `404` عمدًا حتى لا يكشف وجود المعرّف. سجلات المحادثات القديمة التي لا تحتوي `user_id` لا تُعرض تلقائيًا لأي مستخدم.

يدعم الخادم الآن مزود OpenAI-compatible عبر `OPENAI_API_KEY` و`OPENAI_API_BASE`، مع إبقاء `GROQ_API_KEY` كخيار بديل. إذا تعطل مزود النموذج، يرجع الشات رسالة خطأ مفهومة بدل انتهاء الطلب بلا رد. كما أضيف توسيع ثنائي اللغة لاستعلامات ضغط الدم حتى تتطابق الأسئلة العربية مع نصوص الإرشادات الإنجليزية.

## Supabase Database

تم نقل قاعدة البيانات إلى Supabase Postgres من خلال migration موجود في `supabase/migrations/20260820_initial_clinical_schema.sql`. التطبيق يستخدم SQLAlchemy كطبقة ORM، لكن الاتصال الفعلي في بيئة الإنتاج يكون عبر `SUPABASE_DB_URL` إلى Supabase Session Pooler. تم إيقاف `Base.metadata.create_all` من startup حتى لا ينشئ التطبيق schema خارج نظام migrations.

جدول `users` هو جدول الحسابات الأساسي، وكل `projects` يرتبط بصاحبه عبر `created_by`، وكل `conversations` و`messages` يرتبطان بالمستخدم والمشروع، بينما ترتبط `documents` و`ingest_jobs` بالمشروع والمستند. توجد مفاتيح أجنبية وفهارس على علاقات الملكية، مع تفعيل RLS لمنع الوصول المباشر العام إلى الجداول. عزل المحادثات عبر الـAPI يظل مفروضًا كذلك في طبقة التطبيق.

للتشغيل في Railway أو أي بيئة نشر، أضف متغير `SUPABASE_DB_URL` باستخدام **Session Pooler connection string** من Supabase Dashboard → Connect، مع `sslmode=require`. بعد ذلك شغّل:

```bash
pip install -r requirements.txt
```

ثم أعد تشغيل الخدمة. الـmigration تم تطبيقه بالفعل على مشروع Supabase المتصل، ولا توجد حاجة لإنشاء جداول يدويًا من لوحة التحكم.
