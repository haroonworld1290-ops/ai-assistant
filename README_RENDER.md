Render deployment guide

Quick summary
- This repo can be deployed to Render as a Docker web service using the included `Dockerfile`.
- You can enable auto-deploy in Render (connect GitHub) or use the included GitHub Action to trigger a deploy via the Render API.

Steps — create service on Render
1. Sign in to https://dashboard.render.com and connect your GitHub repo.
2. Create a new "Web Service" and choose "Docker" as the environment.
   - Service name: `whatsapp-bot` (or your choice)
   - Branch: `main`
   - Dockerfile path: `/Dockerfile`
   - Set `PORT` environment variable to `8000` (the app will use `PORT` from Render)
3. In the service settings add the following environment variables (from your `.env`):
   - `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `APP_SECRET`, `VERIFY_TOKEN`, `OPENAI_API_KEY`, `ADMIN_TOKEN` (or `ADMIN_USER`/`ADMIN_PASS`) etc.
4. Optionally enable automatic deploys on push.

Using the provided GitHub Action
1. Add repository secrets in GitHub: `RENDER_API_KEY` and `RENDER_SERVICE_ID`.
   - `RENDER_API_KEY`: create an API key in Render dashboard (Account → API Keys).
   - `RENDER_SERVICE_ID`: find the service ID in Render (Service → Settings → Service ID).
2. The workflow `.github/workflows/render-deploy.yml` will call the Render API on push to `main` and create a new deploy.

Manual deploy
- You can also let Render handle builds automatically when you connect the repo and push to `main`.

Local testing (recommended before cloud deploy)
1. Run server locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:WHATSAPP_TOKEN='dummy'
$env:OPENAI_API_KEY='dummy'
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

2. Expose to the internet for webhook testing with `ngrok`:

```powershell
ngrok http 8000
```

Security tips
- Use Render secrets (env vars) or Render's managed secrets rather than committing `.env` to the repo.
- Prefer an API key with minimal permissions for the GitHub Action.

If you want, I can: create a small `Procfile` or update the `Dockerfile` for Render-specific optimizations, or generate an example `render.yaml` for Render's infra-as-code (already added). Tell me which next step you want.