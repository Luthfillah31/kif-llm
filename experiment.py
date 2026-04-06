import httpx
import time
import threading
import random
import asyncio

# ── Rate limiting state ────────────────────────────────────────────
_last_call = 0.0
_sync_lock = threading.Lock()
_async_lock = asyncio.Lock() if False else None  # created lazily
MIN_GAP = 15.0
BOT_UA = "KIFQA-Research-Agent/1.0 (mailto:luthfillahatar@gmail.com)"

def _sync_wait(url: str):
    if not ("wikidata" in str(url) or "wikimedia" in str(url)):
        return
    global _last_call
    with _sync_lock:
        gap = MIN_GAP - (time.monotonic() - _last_call)
        if gap > 0:
            wait = gap + random.uniform(1.0, 3.0)
            print(f"[RateLimit] Waiting {wait:.1f}s...")
            time.sleep(wait)
        _last_call = time.monotonic()

async def _async_wait(url: str):
    if not ("wikidata" in str(url) or "wikimedia" in str(url)):
        return
    global _async_lock, _last_call
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    async with _async_lock:
        gap = MIN_GAP - (time.monotonic() - _last_call)
        if gap > 0:
            wait = gap + random.uniform(1.0, 3.0)
            print(f"[RateLimit-Async] Waiting {wait:.1f}s...")
            await asyncio.sleep(wait)
        _last_call = time.monotonic()

# ── Patch sync httpx.Client.send ──────────────────────────────────
_orig_sync_send = httpx.Client.send
def _patched_sync_send(self, request, **kwargs):
    _sync_wait(str(request.url))
    request.headers['User-Agent'] = BOT_UA
    request.headers['Api-User-Agent'] = BOT_UA
    return _orig_sync_send(self, request, **kwargs)
httpx.Client.send = _patched_sync_send

# ── Patch async httpx.AsyncClient.send ────────────────────────────
_orig_async_send = httpx.AsyncClient.send
async def _patched_async_send(self, request, **kwargs):
    await _async_wait(str(request.url))  # ← async-safe wait
    request.headers['User-Agent'] = BOT_UA
    request.headers['Api-User-Agent'] = BOT_UA
    return await _orig_async_send(self, request, **kwargs)
httpx.AsyncClient.send = _patched_async_send

# ── Fix the downloader's bad defaults directly ─────────────────────
# Must do this AFTER import but it patches the class-level dict
from kif_lib.vocabulary.wd.downloader import Downloader
Downloader.default_options['max_requests'] = 1   # no parallel requests
Downloader.default_options['http_headers']['User-Agent'] = BOT_UA

# ── NOW safe to import kif ─────────────────────────────────────────
from kifqa import KIFQA
from kif_lib import Store, Search

store = Store('wikidata', timeout=90.0)

kif_wiki_kbqa = KIFQA(
    store=store,
    search=Search('wikidata-wapi', limit=3),
    model_provider='ollama',
    model_name='deepseek-v3.1:671b-cloud',
    model_endpoint='http://localhost:11434',
    model_apikey='dummy_key',
    model_params={'temperature': 0.0}
)

question = 'Where did Roger Marquis die'
print(f"\n--- QUERYING: {question} ---")

try:
    stmts = kif_wiki_kbqa.query(question)
    found = False
    for stmt in stmts:
        print(stmt)
        found = True
    if not found:
        print("No results found.")
except Exception as e:
    import traceback
    traceback.print_exc()