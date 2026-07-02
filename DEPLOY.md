# Deploying to Render (free tier)

## 1. Push this project to GitHub

```bash
cd shl-agent
git init
git add .
git commit -m "SHL Assessment Recommender"
```

Create a new **empty** repo on GitHub (e.g. `shl-assessment-recommender`) at
https://github.com/new — do NOT initialize it with a README (keep it empty),
then:

```bash
git remote add origin https://github.com/<your-username>/shl-assessment-recommender.git
git branch -M main
git push -u origin main
```

## 2. Create the Render service

1. Go to https://dashboard.render.com and sign in (GitHub login is easiest).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if not already connected, and select the
   `shl-assessment-recommender` repo.
4. Render should auto-detect `render.yaml` and pre-fill the settings
   (build command, start command, health check path). If it doesn't
   auto-detect, set manually:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
   - **Instance Type**: Free
5. Under **Environment Variables**, add:
   - `GROQ_API_KEY` = your Groq key
   - `GROQ_MODEL` = `llama-3.3-70b-versatile`
6. Click **Create Web Service**. First deploy takes a few minutes (installs
   dependencies, builds the BM25 index at startup).

## 3. Verify it's live

Render gives you a URL like `https://shl-assessment-recommender.onrender.com`.

```bash
curl https://shl-assessment-recommender.onrender.com/health
```

Should return `{"status":"ok"}`. Free-tier services sleep after ~15 minutes
of inactivity — the first request after sleeping can take up to a minute or
so to wake up, which is why the assignment explicitly allows up to 2 minutes
for the first `/health` call.

Then test `/chat`:
```bash
curl -X POST https://shl-assessment-recommender.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## 4. Re-run the trace harness against the live URL

```bash
python tests/run_traces.py --base-url https://shl-assessment-recommender.onrender.com --traces-dir sample_conversations/GenAI_SampleConversations --delay 15
```

This confirms the deployed service behaves the same as local -- important
because environment differences (missing env var, wrong Python version,
cold-start timing) are a common last-mile failure mode.

## 5. Submit

Use `https://shl-assessment-recommender.onrender.com` as the public API
endpoint URL in the submission form.

## Redeploying after code changes

Render auto-redeploys on every push to `main`:
```bash
git add .
git commit -m "tweak prompt"
git push
```
