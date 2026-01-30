import os
import hmac
import hashlib
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import httpx
import asyncio
import aiosqlite
import mimetypes
# try to support either 'aioredis' or 'redis.asyncio' as a fallback
try:
    import aioredis
except Exception:
    import redis.asyncio as aioredis  # type: ignore
from dotenv import load_dotenv
import openai

# Load environment
load_dotenv()
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
APP_SECRET = os.getenv("APP_SECRET")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "verify_token")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
DB_URL = os.getenv("DB_URL", "file:whatsapp_chat.db")
CLEANUP_AFTER_DAYS = int(os.getenv("CLEANUP_AFTER_DAYS", "30"))

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("whatsapp-bot")

# Warn instead of asserting so tests and local dev can run without all env vars set
if not WHATSAPP_TOKEN:
    logger.warning("WHATSAPP_TOKEN not set; outgoing WhatsApp calls will fail")
if not PHONE_NUMBER_ID:
    logger.warning("PHONE_NUMBER_ID not set; outgoing WhatsApp calls will fail")
if not APP_SECRET:
    logger.warning("APP_SECRET not set; webhook verification may fail")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set; OpenAI calls will fail")

openai.api_key = OPENAI_API_KEY if OPENAI_API_KEY else None

app = FastAPI()
start_time: datetime = datetime.utcnow()

redis = None
if REDIS_URL:
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)

# serve admin UI static file
from fastapi.responses import FileResponse

@app.get("/admin/ui")
async def admin_ui():
    """Serve a lightweight admin page to send interactive templates and lists."""
    # Note: This is an unauthenticated file serve; the endpoints the page calls are protected via ADMIN_TOKEN
    if not os.path.exists("static/admin.html"):
        raise HTTPException(status_code=404, detail="Admin UI not found; ensure static/admin.html exists")
    return FileResponse("static/admin.html", media_type="text/html")

# Optional Redis client (only if REDIS_URL is set)
redis = None
if REDIS_URL:
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)

# Simple SQLite helpers using aiosqlite
async def init_db():
    async with aiosqlite.connect(DB_URL) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        await db.commit()

# Database initialization will run in startup event to avoid event loop conflicts
# asyncio.get_event_loop().run_until_complete(init_db())


@app.on_event("startup")
async def startup_event():
    global redis, start_time
    start_time = datetime.utcnow()

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception:
        logger.exception("Database initialization failed")

    if REDIS_URL:
        logger.info("Connecting to Redis...")
        try:
            redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.exception("Redis connection failed, continuing without Redis: %s", e)
            redis = None
    # start cleanup background task
    app.state.cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    task = getattr(app.state, "cleanup_task", None)
    if task:
        task.cancel()
        try:
            await task
        except Exception:
            pass
    if redis:
        await redis.close()
    logger.info("Shutdown complete")

async def periodic_cleanup():
    # runs daily to remove old messages from sqlite
    while True:
        try:
            cutoff = int(time.time()) - CLEANUP_AFTER_DAYS * 86400
            async with aiosqlite.connect(DB_URL) as db:
                await db.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
                await db.commit()
            logger.info("Periodic cleanup done (messages older than %d days removed)", CLEANUP_AFTER_DAYS)
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception:
            logger.exception("Error during periodic cleanup")
        await asyncio.sleep(24 * 3600)  # run daily

async def save_message_sqlite(user_id: str, role: str, content: str):
    ts = int(time.time())
    async with aiosqlite.connect(DB_URL) as db:
        await db.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, ts),
        )
        await db.commit()

async def get_recent_messages_sqlite(user_id: str, limit: int = 8):
    async with aiosqlite.connect(DB_URL) as db:
        cur = await db.execute(
            "SELECT role, content, created_at FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
        # return list of tuples (role, content, created_at) in chronological order
        return [(r[0], r[1], r[2]) for r in reversed(rows)]

# Redis helpers
async def save_message_redis(user_id: str, role: str, content: str):
    ts = int(time.time())
    if not redis:
        # reuse sqlite storage which already stores timestamps
        return await save_message_sqlite(user_id, role, content)
    await redis.rpush(f"chat:{user_id}", json.dumps({"role": role, "content": content, "created_at": ts}))
    await redis.ltrim(f"chat:{user_id}", -50, -1)

async def get_recent_messages_redis(user_id: str, limit: int = 8):
    if not redis:
        return await get_recent_messages_sqlite(user_id, limit)
    items = await redis.lrange(f"chat:{user_id}", -limit, -1)
    out = []
    for it in items:
        d = json.loads(it)
        # fallback to None timestamp if not present
        out.append((d.get("role"), d.get("content"), d.get("created_at")))
    return out

# Verify X-Hub-Signature-256 header
def verify_signature(raw_body: bytes, header_signature: str) -> bool:
    if not header_signature:
        return False
    if not APP_SECRET:
        # Can't verify without app secret
        return False
    # header is like: sha256=...  (Meta uses sha256)
    try:
        prefix, sig = header_signature.split("=", 1)
    except Exception:
        return False
    if prefix != "sha256":
        return False
    mac = hmac.new(APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, sig)


@app.get("/webhook")
async def webhook_verify(mode: str = None, challenge: str = None, verify_token: str = None):
    # Meta webhook verification
    # Urdu: verification token check karke challenge return karna hai
    if mode == "subscribe" and verify_token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook_receive(request: Request):
    raw = await request.body()
    sig_header = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(raw, sig_header):
        logger.warning("Invalid signature. Header: %s", sig_header)
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = await request.json()
    # basic navigation for messages; Meta sends entries -> changes -> value -> messages
    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages") or []
                for msg in messages:
                    message_id = msg.get("id")
                    from_number = msg.get("from")
                    mtype = msg.get("type")
                    # Dedup using redis if available
                    if message_id and redis:
                        seen = await redis.get(f"msg:{message_id}")
                        if seen:
                            logger.info("Duplicate message %s ignored", message_id)
                            continue
                        await redis.set(f"msg:{message_id}", "1", ex=24*3600)
                    if mtype == "text":
                        text = msg.get("text", {}).get("body", "").strip()
                        asyncio.create_task(handle_message(from_number, text))
                    elif mtype == "image":
                        image = msg.get("image", {})
                        caption = image.get("caption", "") or ""
                        image_id = image.get("id")
                        if image_id:
                            # fetch and save media asynchronously
                            asyncio.create_task(process_media_message(from_number, image_id, caption))
                        else:
                            notice = f"[Image received]"
                            if caption:
                                notice += f" caption: {caption}"
                            asyncio.create_task(handle_message(from_number, notice))
                            asyncio.create_task(send_whatsapp_text(from_number, "Image received! Thanks."))
                    else:
                        logger.info("Unsupported message type: %s", mtype)
    except Exception as e:
        logger.exception("Webhook handling error")
    return PlainTextResponse("EVENT_RECEIVED")


async def send_whatsapp_text(to: str, text: str):
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    body = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json=body, timeout=10.0)
            r.raise_for_status()
            logger.info("Sent message to %s", to)
            return True
        except Exception:
            logger.exception("Error sending message to WhatsApp %s", to)
            return False


async def send_template_message(to: str, template_name: str, language: str = "en_US", components: Optional[list] = None):
    """Send a template message by name (templates must be approved in Meta Business Manager)."""
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    body = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {"name": template_name, "language": {"code": language}}
    }
    if components:
        body["template"]["components"] = components
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json=body, timeout=10.0)
            r.raise_for_status()
            logger.info("Sent template %s to %s", template_name, to)
            return True
        except Exception:
            logger.exception("Error sending template %s to %s", template_name, to)
            return False


async def send_interactive_buttons(to: str, body_text: str, buttons: list, header_text: Optional[str] = None, footer_text: Optional[str] = None):
    """Send an interactive message with quick-reply buttons.

    - buttons: list of dicts {"id": "btn1", "title": "Yes"}
    """
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    interactive = {
        "type": "button",
        "body": {"text": body_text},
        "action": {"buttons": []}
    }
    if header_text:
        interactive["header"] = {"type": "text", "text": header_text}
    if footer_text:
        interactive["footer"] = {"text": footer_text}
    for b in buttons:
        interactive["action"]["buttons"].append({"type": "reply", "reply": {"id": b.get("id"), "title": b.get("title")}})
    body = {"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": interactive}
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json=body, timeout=10.0)
            r.raise_for_status()
            logger.info("Sent interactive buttons to %s", to)
            return True
        except Exception:
            logger.exception("Error sending interactive buttons to %s", to)
            return False


async def send_list_message(to: str, header: str, body_text: str, sections: list, footer: Optional[str] = None, button_text: str = "Choose"):
    """Send a list message with sections.

    - sections: list of {"title": "Section Title", "rows": [{"id":"r1","title":"Title","description":"desc"}, ...]}
    """
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    interactive = {
        "type": "list",
        "header": {"type": "text", "text": header},
        "body": {"text": body_text},
        "action": {"button": button_text, "sections": sections}
    }
    if footer:
        interactive["footer"] = {"text": footer}
    body = {"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": interactive}
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json=body, timeout=10.0)
            r.raise_for_status()
            logger.info("Sent list message to %s", to)
            return True
        except Exception:
            logger.exception("Error sending list message to %s", to)
            return False


# Admin endpoints to trigger templates/interactives from a trusted client
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ADMIN_USER = os.getenv("ADMIN_USER")
ADMIN_PASS = os.getenv("ADMIN_PASS")
SESSION_TTL = int(os.getenv("ADMIN_SESSION_TTL", "8")) * 3600  # seconds, default 8 hours


def _check_admin(token: Optional[str], request: Optional[Request] = None):
    # Check token header first
    if token and ADMIN_TOKEN and token == ADMIN_TOKEN:
        return True
    # Check cookie-based session
    if request:
        sess = request.cookies.get("admin_session")
        if sess and verify_admin_session(sess):
            return True
    return False


# Simple HMAC-signed session token
def create_admin_session(username: str, expiry_seconds: int = SESSION_TTL) -> str:
    ts = int(time.time()) + expiry_seconds
    payload = f"{username}|{ts}"
    secret = ADMIN_TOKEN or APP_SECRET or ""
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = f"{payload}|{sig}"
    return token


def verify_admin_session(token: str) -> bool:
    try:
        parts = token.split("|")
        if len(parts) != 3:
            return False
        username, ts_s, sig = parts
        payload = f"{username}|{ts_s}"
        secret = ADMIN_TOKEN or APP_SECRET or ""
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return False
        if int(ts_s) < int(time.time()):
            return False
        # Optionally check username matches ADMIN_USER
        if ADMIN_USER and username != ADMIN_USER:
            return False
        return True
    except Exception:
        return False


@app.post("/admin/login")
async def admin_login(request: Request):
    data = await request.json()
    user = data.get("user")
    password = data.get("pass")
    if not ADMIN_USER or not ADMIN_PASS:
        raise HTTPException(status_code=501, detail="Admin login not configured")
    if user == ADMIN_USER and password == ADMIN_PASS:
        token = create_admin_session(user)
        resp = JSONResponse({"ok": True})
        # set cookie, HttpOnly so JS can't read it (browser will send it automatically)
        resp.set_cookie("admin_session", token, httponly=True, samesite="Lax", max_age=SESSION_TTL)
        return resp
    raise HTTPException(status_code=403, detail="invalid credentials")


@app.post("/admin/logout")
async def admin_logout(request: Request):
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("admin_session")
    return resp


@app.post("/admin/send-button")
async def admin_send_button(request: Request):
    token = request.headers.get("X-Admin-Token") or request.query_params.get("admin_token")
    if not _check_admin(token, request):
        raise HTTPException(status_code=403, detail="admin token or session required")
    data = await request.json()
    to = data.get("to")
    body_text = data.get("body")
    buttons = data.get("buttons", [])
    header = data.get("header")
    footer = data.get("footer")
    if not to or not body_text or not buttons:
        raise HTTPException(status_code=400, detail="to, body, buttons required")
    ok = await send_interactive_buttons(to, body_text, buttons, header_text=header, footer_text=footer)
    return JSONResponse({"ok": bool(ok)})


@app.post("/admin/send-list")
async def admin_send_list(request: Request):
    token = request.headers.get("X-Admin-Token") or request.query_params.get("admin_token")
    if not _check_admin(token, request):
        raise HTTPException(status_code=403, detail="admin token or session required")
    data = await request.json()
    to = data.get("to")
    header = data.get("header")
    body_text = data.get("body")
    sections = data.get("sections")
    footer = data.get("footer")
    button_text = data.get("button_text", "Choose")
    if not to or not header or not body_text or not sections:
        raise HTTPException(status_code=400, detail="to, header, body, sections required")
    ok = await send_list_message(to, header, body_text, sections, footer=footer, button_text=button_text)
    return JSONResponse({"ok": bool(ok)})


async def fetch_media(media_id: str) -> Optional[str]:
    """Download media from WhatsApp Cloud and save to ./media, returns file path or None."""
    if not WHATSAPP_TOKEN:
        logger.warning("WHATSAPP_TOKEN not set; cannot fetch media")
        return None
    info_url = f"https://graph.facebook.com/v17.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(info_url, headers=headers, timeout=10.0)
            r.raise_for_status()
            data = r.json()
            download_url = data.get("url")
            mime = data.get("mime_type") or data.get("mimeType") or ""
            if not download_url:
                logger.error("Media info missing download URL for %s", media_id)
                return None
            # Now download binary
            r2 = await client.get(download_url, headers=headers, timeout=30.0)
            r2.raise_for_status()
            content = r2.content
    except Exception:
        logger.exception("Error fetching media %s", media_id)
        return None

    ext = mimetypes.guess_extension(mime.split(";")[0]) or ""
    if not ext:
        if "jpeg" in mime:
            ext = ".jpg"
        elif "png" in mime:
            ext = ".png"
        else:
            ext = ".bin"
    os.makedirs("media", exist_ok=True)
    filename = os.path.join("media", f"{media_id}{ext}")
    try:
        with open(filename, "wb") as f:
            f.write(content)
        logger.info("Saved media %s to %s (mime=%s)", media_id, filename, mime)
        return filename
    except Exception:
        logger.exception("Error saving media file %s", filename)
        return None


async def process_media_message(user_id: str, media_id: str, caption: str = ""):
    """Fetch media, save a note to history, and acknowledge user."""
    path = await fetch_media(media_id)
    if path:
        content = f"[Image saved] {os.path.basename(path)}"
    else:
        content = "[Image received]"
    if caption:
        content += f" caption: {caption}"
    # Store as a user message for context
    await save_message_redis(user_id, "user", content)
    # Acknowledge
    await send_whatsapp_text(user_id, "Image received and saved. Thanks!")


async def send_media_and_message(to: str, file_path: str, mime: Optional[str] = None, caption: Optional[str] = None):
    """Upload a media file to WhatsApp Cloud and send it to a user.

    Returns True on success, False otherwise.
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        logger.warning("WHATSAPP_TOKEN or PHONE_NUMBER_ID not set; cannot send media")
        return False

    # Guess mime if not provided
    mime = mime or (mimetypes.guess_type(file_path)[0] or "application/octet-stream")

    upload_url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/media"
    # files for multipart: file => (filename, bytes, content_type)
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, mime)}
            headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
            async with httpx.AsyncClient() as client:
                r = await client.post(upload_url, headers=headers, files=files, timeout=30.0)
                r.raise_for_status()
                data = r.json()
                media_id = data.get("id")
                if not media_id:
                    logger.error("Media upload did not return id: %s", data)
                    return False
                # Now send message referencing this media id
                msg_url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
                body = {"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"id": media_id}}
                if caption:
                    body["image"]["caption"] = caption
                r2 = await client.post(msg_url, headers=headers, json=body, timeout=10.0)
                r2.raise_for_status()
                logger.info("Uploaded media %s and sent to %s", media_id, to)
                return True
    except Exception:
        logger.exception("Error uploading/sending media %s to %s", file_path, to)
        return False


async def handle_message(user_id: str, text: str):
    logger.info("Handle message from %s: %s", user_id, text[:120])
    await save_message_redis(user_id, "user", text)

    # Get last N messages for context
    history = await get_recent_messages_redis(user_id, limit=6)

    # Build messages for OpenAI. Add system message instructing Roman English reply
    messages = [
        {"role": "system", "content": "You are an assistant that replies in Roman-Urdu/English (Roman script). Keep replies concise and friendly."}
    ]
    for role, content, created_at in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})

    # Call OpenAI ChatCompletion
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.6,
        )
        bot_reply = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI error")
        bot_reply = "Maaf, thora masla hai. Phir try karen."  # Urdu fallback

    # Save bot message
    await save_message_redis(user_id, "assistant", bot_reply)

    # Send reply
    await send_whatsapp_text(user_id, bot_reply)


@app.get("/health")
async def health():
    """Return basic status and uptime."""
    now = datetime.utcnow()
    uptime = (now - start_time).total_seconds()
    redis_ok = False
    try:
        if redis:
            await redis.ping()
            redis_ok = True
    except Exception:
        redis_ok = False
    return JSONResponse({"status": "ok", "uptime_seconds": int(uptime), "redis": redis_ok, "start_time": start_time.isoformat()})


@app.get("/sessions")
async def sessions(request: Request, user_id: str, limit: int = 20):
    """Return recent chat history for a user (debug endpoint). Requires admin token or session."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    token = request.headers.get("X-Admin-Token") or request.query_params.get("admin_token")
    if not _check_admin(token, request):
        raise HTTPException(status_code=403, detail="admin token or session required")
    history = await get_recent_messages_redis(user_id, limit=limit)
    out_messages = []
    for r, c, ts in history:
        try:
            ts_iso = datetime.utcfromtimestamp(int(ts)).isoformat() if ts else None
        except Exception:
            ts_iso = None
        out_messages.append({"role": r, "content": c, "created_at": ts_iso})
    return JSONResponse({"user_id": user_id, "messages": out_messages})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
