import re
import os
import uuid
import asyncio
import tempfile
import shutil
import logging
from typing import Optional
from urllib.parse import quote, urlparse

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vidgrab")

# Browser impersonation target (requires curl_cffi)
try:
    from curl_cffi import requests as _cfreq  # noqa: F401
    _CHROME_TARGET = ImpersonateTarget(client="chrome")
    _IMPERSONATE_AVAILABLE = True
except ImportError:
    _CHROME_TARGET = None
    _IMPERSONATE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Monkey-patch: XHamster changed their format URLs from URL-encoded hex paths
# to bare hex blobs.  yt-dlp's extractor expects /HEX/remainder but now gets
# a raw hex string.  We decrypt it here and return the full URL.
# ---------------------------------------------------------------------------
_BARE_HEX_RE = re.compile(r'^[0-9a-fA-F]{20,}$')

def _patched_decipher_format_url(self, format_url, format_id):
    """Handle both old URL-hex and new bare-hex XHamster format URLs."""
    import urllib.parse as _up
    from yt_dlp.extractor.xhamster import _ByteGenerator
    from yt_dlp.utils import ExtractorError

    # --- NEW: bare hex blob (no scheme, no slashes, just hex chars) ---
    clean = format_url.strip()
    if _BARE_HEX_RE.match(clean):
        try:
            byte_data = bytes.fromhex(clean)
            if len(byte_data) < 6:
                self.report_warning(f'Skipping format "{format_id}": hex data too short')
                return None
            algo_id = byte_data[0]
            seed = int.from_bytes(byte_data[1:5], byteorder='little', signed=True)
            byte_gen = _ByteGenerator(algo_id, seed)
            deciphered = bytearray(b ^ next(byte_gen) for b in byte_data[5:]).decode('latin-1')
            if deciphered.startswith('http'):
                return deciphered
            # Possibly a path, prepend scheme
            return f'https:{deciphered}' if deciphered.startswith('//') else f'https://{deciphered}'
        except ExtractorError as e:
            self.report_warning(f'Skipping format "{format_id}": {e.msg}')
            return None
        except Exception as e:
            self.report_warning(f'Skipping format "{format_id}": decipher error {e}')
            return None

    # --- ORIGINAL logic for old-style URL-encoded hex paths ---
    parsed_url = _up.urlparse(format_url)
    hex_string, path_remainder = self._search_regex(
        r'^/(?P<hex>[0-9a-fA-F]{12,})(?P<rem>[/,].+)$', parsed_url.path,
        'url components', default=(None, None), group=('hex', 'rem'))
    if not hex_string:
        self.report_warning(f'Skipping format "{format_id}": unsupported URL format')
        return None

    byte_data = bytes.fromhex(hex_string)
    seed = int.from_bytes(byte_data[1:5], byteorder='little', signed=True)
    try:
        byte_gen = _ByteGenerator(byte_data[0], seed)
    except ExtractorError as e:
        self.report_warning(f'Skipping format "{format_id}": {e.msg}')
        return None

    deciphered = bytearray(b ^ next(byte_gen) for b in byte_data[5:]).decode('latin-1')
    return parsed_url._replace(path=f'/{deciphered}{path_remainder}').geturl()


try:
    from yt_dlp.extractor.xhamster import XHamsterIE
    XHamsterIE._decipher_format_url = _patched_decipher_format_url
    logger.info("XHamster extractor patched for bare-hex URL support")
except Exception as e:
    logger.warning(f"Could not patch XHamster extractor: {e}")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="All-in-One Video Downloader")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Temp directory for downloads – cleaned after streaming
TEMP_DIR = os.path.join(tempfile.gettempdir(), "viddownloader")
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Supported platforms (regex patterns)
# ---------------------------------------------------------------------------

PLATFORM_PATTERNS = {
    "YouTube": r"(youtube\.com|youtu\.be)",
    "Instagram": r"instagram\.com",
    "Facebook": r"(facebook\.com|fb\.watch)",
    "Twitter/X": r"(twitter\.com|x\.com)",
    "TikTok": r"tiktok\.com",
    "Vimeo": r"vimeo\.com",
    "Dailymotion": r"dailymotion\.com",
    "Reddit": r"reddit\.com",
    "Twitch": r"twitch\.tv",
    "Pinterest": r"pinterest\.(com|co)",
    "LinkedIn": r"linkedin\.com",
    "Tumblr": r"tumblr\.com",
    "SoundCloud": r"soundcloud\.com",
    "Bilibili": r"bilibili\.com",
    "Rumble": r"rumble\.com",
    "Streamable": r"streamable\.com",
    "Bitchute": r"bitchute\.com",
    "OK.ru": r"ok\.ru",
    "VK": r"vk\.com",
    "Imgur": r"imgur\.com",
    "Loom": r"loom\.com",
    "Bandcamp": r"bandcamp\.com",
    "Mixcloud": r"mixcloud\.com",
}


def detect_platform(url: str) -> str:
    for name, pattern in PLATFORM_PATTERNS.items():
        if re.search(pattern, url, re.IGNORECASE):
            return name
    # Extract domain as platform name for unknown sites
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        # Strip www. and .com/.org etc for a clean label
        domain = re.sub(r'^www\.', '', domain)
        parts = domain.rsplit('.', 1)
        return parts[0].capitalize() if parts else "Website"
    except Exception:
        return "Website"


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

URL_REGEX = re.compile(
    r"^https?://"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,}"
    r"(?::\d{1,5})?"
    r"(?:/[^\s]*)?$"
)


def validate_url(url: str) -> str:
    url = url.strip()
    if not URL_REGEX.match(url):
        raise ValueError("Invalid URL format")
    # Block private / local IPs
    for blocked in ("127.0.0.1", "localhost", "0.0.0.0", "::1", "10.", "192.168.", "172.16."):
        if blocked in url:
            raise ValueError("Local/private URLs are not allowed")
    return url


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FetchRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        return validate_url(v)


class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None
    ext: str = "mp4"         # mp4, webm, mp3
    quality: Optional[str] = None  # e.g. "720", "1080", "best"

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        return validate_url(v)

    @field_validator("ext")
    @classmethod
    def clean_ext(cls, v: str) -> str:
        if v not in ("mp4", "webm", "mp3"):
            raise ValueError("Unsupported format. Choose mp4, webm, or mp3.")
        return v


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------

# Common browser user-agent to avoid bot-detection blocks
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _base_opts() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "geo_bypass": True,
        "socket_timeout": 30,
        "extract_flat": False,
        "user_agent": _USER_AGENT,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "http_headers": {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    }


def _has_usable_formats(info: dict | None) -> bool:
    """Check if extracted info actually has playable format URLs."""
    if not info:
        return False
    formats = info.get("formats") or []
    if formats:
        return any(f.get("url") for f in formats)
    return bool(info.get("url"))


def _extract_error_message(exc: Exception) -> str:
    """Get a human-readable error string from yt-dlp exceptions."""
    msg = str(exc)
    # yt-dlp DownloadError wraps the message with 'ERROR: ' prefix
    if msg.startswith("ERROR: "):
        msg = msg[7:]
    # Strip the "please report" boilerplate
    if "; please report" in msg:
        msg = msg[:msg.index("; please report")]
    return msg.strip() or "Unknown extraction error"


def _build_strategy_opts(strategy: int) -> dict:
    """Build yt-dlp options for a given fallback strategy number."""
    opts = _base_opts()
    opts["skip_download"] = True

    if strategy == 1:
        # Default: site-specific extractor + impersonate if available
        if _IMPERSONATE_AVAILABLE:
            opts["impersonate"] = _CHROME_TARGET
    elif strategy == 2:
        # Without impersonate (in case it causes issues)
        pass
    elif strategy == 3:
        # Force generic extractor
        opts["force_generic_extractor"] = True
    elif strategy == 4:
        # Generic + impersonate
        opts["force_generic_extractor"] = True
        if not _IMPERSONATE_AVAILABLE:
            return None
        opts["impersonate"] = _CHROME_TARGET
    elif strategy == 5:
        # Last resort: allow all formats, no format filtering
        if _IMPERSONATE_AVAILABLE:
            opts["impersonate"] = _CHROME_TARGET
        opts["format"] = "best"
        opts["check_formats"] = False
    return opts


def fetch_info(url: str) -> dict:
    """Return video metadata without downloading.

    Uses a multi-stage fallback strategy:
      1. Standard extraction (site-specific extractor)
      2. With browser impersonation (bypasses anti-bot)
      3. Force generic extractor (finds <video> tags, og:video, etc.)
      4. Generic + browser impersonation
      5. Last resort with relaxed format checks
    """
    errors_collected = []

    for strategy_num in range(1, 6):
        opts = _build_strategy_opts(strategy_num)
        if opts is None:
            continue  # strategy not available

        strategy_name = [
            "", "standard", "impersonate", "generic",
            "generic+impersonate", "relaxed"
        ][strategy_num]

        try:
            logger.info(f"Fetch strategy {strategy_num} ({strategy_name}) for: {url}")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if _has_usable_formats(info):
                    logger.info(f"Strategy {strategy_num} succeeded with formats")
                    return info
                elif info:
                    logger.info(f"Strategy {strategy_num} returned info but no usable formats")
                    # Keep the info in case all strategies fail - we may still have metadata
                    errors_collected.append(("no_formats", info))
                else:
                    logger.info(f"Strategy {strategy_num} returned None")
        except Exception as e:
            err_msg = _extract_error_message(e)
            logger.warning(f"Strategy {strategy_num} failed: {err_msg}")
            errors_collected.append(("exception", e))

    # All strategies exhausted - try to return best available info even without formats
    # Some sites return metadata but formats need a different approach for download
    for tag, data in errors_collected:
        if tag == "no_formats" and isinstance(data, dict):
            logger.info("Returning info without formats as last resort")
            return data

    # Build a helpful error message
    error_msgs = []
    for tag, data in errors_collected:
        if tag == "exception":
            error_msgs.append(_extract_error_message(data))

    if error_msgs:
        # De-duplicate and show unique errors
        unique = list(dict.fromkeys(error_msgs))
        detail = "; ".join(unique[:3])
        raise RuntimeError(detail)

    raise RuntimeError(
        "No video found on this page. The site may require login, "
        "use DRM protection, or load video only through JavaScript."
    )


def build_format_list(info: dict) -> list[dict]:
    """Build a simplified list of available formats.

    Works for both sites with rich format data and generic/embedded
    videos that may only expose a single direct URL.
    """
    formats = []
    seen = set()
    raw = info.get("formats") or []

    # If yt-dlp found no separate formats but has a direct url, create one entry
    if not raw and info.get("url"):
        ext = info.get("ext", "mp4")
        raw = [{
            "format_id": "direct",
            "ext": ext,
            "url": info["url"],
            "vcodec": "unknown",
            "acodec": "unknown",
            "height": info.get("height"),
            "filesize": info.get("filesize"),
            "tbr": info.get("tbr"),
        }]

    for f in raw:
        fid = f.get("format_id", "")
        ext = f.get("ext", "")
        height = f.get("height")
        width = f.get("width")
        acodec = f.get("acodec", "none") or "none"
        vcodec = f.get("vcodec", "none") or "none"
        filesize = f.get("filesize") or f.get("filesize_approx")
        tbr = f.get("tbr")
        fps = f.get("fps")

        has_video = vcodec not in ("none", None)
        has_audio = acodec not in ("none", None)

        label_parts = []
        if has_video and height:
            label_parts.append(f"{height}p")
        elif has_video and width:
            label_parts.append(f"{width}w")
        if fps and fps > 30:
            label_parts.append(f"{int(fps)}fps")
        if has_video and not has_audio:
            label_parts.append("video only")
        if not has_video and has_audio:
            label_parts.append("audio only")
        label_parts.append(ext)
        if tbr:
            label_parts.append(f"~{int(tbr)}kbps")

        label = " | ".join(label_parts)
        key = (height, ext, has_video, has_audio)
        if key in seen:
            continue
        seen.add(key)

        formats.append({
            "format_id": fid,
            "ext": ext,
            "height": height,
            "has_video": has_video,
            "has_audio": has_audio,
            "filesize": filesize,
            "label": label,
        })

    # Sort: video first (highest res first), then audio
    formats.sort(key=lambda x: (not x["has_video"], -(x["height"] or 0)))
    return formats


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index = os.path.join(STATIC_DIR, "index.html")
    with open(index, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/fetch")
@limiter.limit("30/minute")
async def api_fetch(req: FetchRequest, request: Request):
    """Fetch video metadata and available formats."""
    try:
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, fetch_info, req.url)
    except Exception as exc:
        detail = _extract_error_message(exc) if hasattr(exc, 'args') else str(exc)
        if not detail or detail == "Unknown extraction error":
            detail = (
                "Could not extract video from this URL. "
                "The site may block automated access, require login, or use DRM."
            )
        logger.error(f"Fetch failed for {req.url}: {detail}")
        raise HTTPException(status_code=400, detail=f"Could not fetch video info: {detail[:500]}")

    platform = detect_platform(req.url)
    formats = build_format_list(info)

    return {
        "title": info.get("title") or info.get("webpage_url_basename", "Unknown"),
        "thumbnail": info.get("thumbnail") or info.get("thumbnails", [{}])[-1].get("url"),
        "duration": info.get("duration"),
        "platform": platform,
        "uploader": info.get("uploader") or info.get("channel") or info.get("uploader_id", ""),
        "webpage_url": info.get("webpage_url", req.url),
        "description": (info.get("description") or "")[:200],
        "formats": formats,
    }


@app.post("/api/download")
@limiter.limit("10/minute")
async def api_download(req: DownloadRequest, request: Request):
    """Download the video/audio and stream it back."""
    dl_id = uuid.uuid4().hex[:12]
    dl_dir = os.path.join(TEMP_DIR, dl_id)
    os.makedirs(dl_dir, exist_ok=True)

    outtmpl = os.path.join(dl_dir, "%(title).80s.%(ext)s")
    opts = _base_opts()
    opts["outtmpl"] = outtmpl

    if req.ext == "mp3":
        opts["format"] = "bestaudio/best"
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
    else:
        if req.format_id:
            # If the chosen format has no audio, merge with best audio
            opts["format"] = f"{req.format_id}+bestaudio/{req.format_id}/best"
        elif req.quality and req.quality != "best":
            opts["format"] = f"bestvideo[height<={req.quality}]+bestaudio/best[height<={req.quality}]/best"
        else:
            opts["format"] = "bestvideo+bestaudio/best"

        if req.ext == "webm":
            opts["merge_output_format"] = "webm"
        else:
            opts["merge_output_format"] = "mp4"

    # Try multiple download strategies (impersonate first since most sites need it)
    download_error = None
    strategies = [
        {"impersonate": _CHROME_TARGET} if _IMPERSONATE_AVAILABLE else {},
        {},
        {"force_generic_extractor": True},
        {"force_generic_extractor": True, "impersonate": _CHROME_TARGET} if _IMPERSONATE_AVAILABLE else None,
    ]
    for i, strategy in enumerate(strategies):
        if strategy is None:
            continue
        try:
            dl_opts = {**opts, **strategy}
            logger.info(f"Download strategy {i+1} for: {req.url}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_download, dl_opts, req.url)
            download_error = None
            break
        except Exception as exc:
            download_error = exc
            logger.warning(f"Download strategy {i+1} failed: {_extract_error_message(exc)}")
            continue

    if download_error:
        shutil.rmtree(dl_dir, ignore_errors=True)
        detail = _extract_error_message(download_error)
        raise HTTPException(status_code=400, detail=f"Download failed: {detail[:500]}")

    # Find the downloaded file
    files = os.listdir(dl_dir)
    if not files:
        shutil.rmtree(dl_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Download produced no file")

    filepath = os.path.join(dl_dir, files[0])
    filename = files[0]
    media_type = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mp3": "audio/mpeg",
    }.get(req.ext, "application/octet-stream")

    def iterfile():
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(1024 * 256):  # 256KB chunks
                    yield chunk
        finally:
            # Cleanup temp files after streaming
            shutil.rmtree(dl_dir, ignore_errors=True)

    file_size = os.path.getsize(filepath)
    # Sanitize filename: keep only ASCII-safe chars to avoid latin-1 encoding errors
    safe_filename = filename.encode("ascii", "ignore").decode("ascii")
    safe_filename = re.sub(r'[^\w\s\-\.\(\)]', '_', safe_filename).strip() or f"download.{req.ext}"
    # Ensure correct extension
    if not safe_filename.endswith(f".{req.ext}"):
        base = os.path.splitext(safe_filename)[0]
        safe_filename = f"{base}.{req.ext}"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{safe_filename}\"; filename*=UTF-8''{quote(filename)}",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(iterfile(), media_type=media_type, headers=headers)


def _run_download(opts: dict, url: str):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}
