#!/usr/bin/env python3
"""
OpenRouter Explorer - Backend API
Fetches and serves model data with filtering capabilities
"""

import json
import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import threading
import time

BASE_URL = "https://openrouter.ai/api/v1"
INTERNAL_URL = "https://openrouter.ai/api/internal/v1"
SITE_URL = "https://openrouter.ai"
FAL_API_URL = "https://api.fal.ai/v1"
CACHE_FILE = "model_cache.json"
BENCH_CACHE_FILE = "bench_cache.json"
FAL_CACHE_FILE = "fal_cache.json"
CACHE_TTL = 300  # 5 minutes
BENCH_CACHE_TTL = 86400  # 24h — benchmarks change rarely, fetched on a slower cycle
FAL_CACHE_TTL = 21600  # 6 hours

# OpenRouter's internal/RSC routes 403 plain clients; mimic a browser.
OR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    ),
    "Referer": "https://openrouter.ai/",
    "Accept": "application/json, text/plain, */*",
}

# Benchmark cache (separate, slower TTL than live perf/pricing)
cached_benchmarks = {}  # canonical_slug -> {"aa": {...}|None, "da": {...}|None}
bench_cache_timestamp = 0
bench_cache_lock = threading.Lock()
bench_cache_loading = False

cached_models = []
cache_timestamp = 0
cache_lock = threading.Lock()

# FAL cache
cached_fal_models = []
fal_cache_timestamp = 0
fal_cache_lock = threading.Lock()
fal_cache_loading = False


def get_all_models():
    """Fetch all models from OpenRouter"""
    resp = requests.get(f"{BASE_URL}/models", timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def get_endpoints(model_id):
    """Fetch endpoints for a specific model"""
    try:
        resp = requests.get(f"{BASE_URL}/models/{model_id}/endpoints", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except Exception:
        pass
    return None


_PERF_RE = re.compile(r'"p50_throughput":([0-9.]+),"p50_latency":([0-9.]+)')


def get_perf(model_id):
    """Parse live p50 throughput/latency from a model's /performance RSC payload.

    OpenRouter removed the old /api/frontend/stats/endpoint JSON API, so the
    only remaining source for live speed/latency is the server-component
    payload behind the model's Performance tab. We take the best throughput
    and best (lowest) latency across providers.
    """
    try:
        resp = requests.get(
            f"{SITE_URL}/{model_id}/performance",
            headers={**OR_HEADERS, "RSC": "1"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        pairs = _PERF_RE.findall(resp.text)
        if not pairs:
            return None
        best_tp = max(float(tp) for tp, _ in pairs)
        best_lat = min(float(lat) for _, lat in pairs)
        return {
            "throughput": round(best_tp, 1) if best_tp > 0 else 0,
            "latency": round(best_lat) if best_lat > 0 else None,
        }
    except Exception:
        return None


def get_aa_benchmark(canonical_slug):
    """Artificial Analysis eval scores via OpenRouter's internal API, keyed by permaslug."""
    try:
        resp = requests.get(
            f"{INTERNAL_URL}/artificial-analysis-benchmarks",
            params={"slug": canonical_slug},
            headers=OR_HEADERS,
            timeout=12,
        )
        data = resp.json().get("data", []) if resp.status_code == 200 else []
        if not data:
            return None
        rec = data[0]
        evals = (rec.get("benchmark_data") or {}).get("evaluations") or {}
        pcts = rec.get("percentiles") or {}
        out = {
            "intelligence": evals.get("artificial_analysis_intelligence_index"),
            "coding": evals.get("artificial_analysis_coding_index"),
            "agentic": evals.get("artificial_analysis_agentic_index"),
            "gpqa": evals.get("gpqa"),
            "hle": evals.get("hle"),
            "scicode": evals.get("scicode"),
            "tau2": evals.get("tau2"),
            "intelligence_pct": pcts.get("intelligence_percentile"),
            "coding_pct": pcts.get("coding_percentile"),
            "agentic_pct": pcts.get("agentic_percentile"),
        }
        # Drop all-null records
        return out if any(v is not None for v in out.values()) else None
    except Exception:
        return None


def get_da_benchmark(canonical_slug):
    """Design Arena Elo (UI/code generation) via OpenRouter's internal API."""
    try:
        resp = requests.get(
            f"{INTERNAL_URL}/design-arena-benchmarks",
            params={"slug": canonical_slug},
            headers=OR_HEADERS,
            timeout=12,
        )
        recs = (resp.json().get("data") or {}).get("records", []) if resp.status_code == 200 else []
        if not recs:
            return None
        best = max(recs, key=lambda x: x.get("elo") or 0)
        return {
            "elo": best.get("elo"),
            "category": best.get("category"),
            "win_rate": best.get("win_rate"),
            "elo_pct": best.get("elo_percentile"),
            "tournaments": best.get("total_tournaments"),
            "categories": {
                x["category"]: {"elo": x.get("elo"), "win_rate": x.get("win_rate")}
                for x in recs if x.get("category")
            },
        }
    except Exception:
        return None


def get_price(model):
    """Extract pricing info ($/1M tokens), including prompt-cache read price if present."""
    pricing = model.get("pricing", {})

    def per_m(key):
        try:
            v = pricing.get(key)
            return round(float(v) * 1_000_000, 4) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    return {
        "input": per_m("prompt"),
        "output": per_m("completion"),
        "cache_read": per_m("input_cache_read"),
        "cache_write": per_m("input_cache_write"),
    }


# Capabilities derived from a model's supported_parameters list.
def extract_capabilities(model):
    params = set(model.get("supported_parameters") or [])
    arch = model.get("architecture", {})
    inputs = set(arch.get("input_modalities") or ["text"])
    outputs = set(arch.get("output_modalities") or ["text"])
    return {
        "tools": "tools" in params or "tool_choice" in params,
        "reasoning": "reasoning" in params or "include_reasoning" in params,
        "structured": "structured_outputs" in params or "response_format" in params,
        "vision": "image" in inputs,
        "audio_in": "audio" in inputs,
        "image_out": "image" in outputs,
    }


def get_provider_details(model_id):
    """Per-provider details (uptime, caching, quantization, pricing) from the
    official endpoints API. Live throughput/latency come from get_perf()."""
    try:
        resp = requests.get(
            f"{BASE_URL}/models/{model_id}/endpoints",
            headers=OR_HEADERS, timeout=12,
        )
        if resp.status_code != 200:
            return [], False
    except Exception:
        return [], False

    endpoints = (resp.json().get("data") or {}).get("endpoints", [])
    details = []
    any_cache = False
    for ep in endpoints:
        pricing = ep.get("pricing", {})

        def per_m(key):
            try:
                v = pricing.get(key)
                return round(float(v) * 1_000_000, 4) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None

        uptime = ep.get("uptime_last_30m")
        caching = bool(ep.get("supports_implicit_caching"))
        any_cache = any_cache or caching
        details.append({
            "name": ep.get("provider_name", "Unknown"),
            "quantization": ep.get("quantization") or "unknown",
            "context_length": ep.get("context_length") or 0,
            "max_completion": ep.get("max_completion_tokens"),
            "price_input": per_m("prompt"),
            "price_output": per_m("completion"),
            "uptime": round(uptime, 1) if uptime is not None else None,
            "caching": caching,
        })
    return details, any_cache


def enrich_model(model):
    """Enrich one model: live perf + provider details + capabilities + metadata.

    Benchmarks are merged separately (slower cache) by canonical_slug.
    """
    model_id = model["id"]
    arch = model.get("architecture", {})
    top = model.get("top_provider") or {}

    provider_details, any_cache = get_provider_details(model_id)
    perf = get_perf(model_id) or {}
    price = get_price(model)

    providers = [p["name"] for p in provider_details]
    best_uptime = max((p["uptime"] for p in provider_details if p["uptime"] is not None),
                      default=None)

    return {
        "id": model_id,
        "canonical_slug": model.get("canonical_slug", model_id),
        "name": model.get("name", model_id),
        "description": model.get("description", "")[:240],
        "context_length": model.get("context_length", 0),
        "max_output": top.get("max_completion_tokens"),
        "throughput": perf.get("throughput", 0),
        "latency": perf.get("latency"),
        "uptime": best_uptime,
        "price_input": price["input"],
        "price_output": price["output"],
        "price_cache_read": price["cache_read"],
        "supports_caching": any_cache,
        "providers": providers,
        "provider_details": provider_details,
        "modality": arch.get("modality", "text->text"),
        "input_modalities": arch.get("input_modalities", ["text"]),
        "output_modalities": arch.get("output_modalities", ["text"]),
        "capabilities": extract_capabilities(model),
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "is_moderated": top.get("is_moderated"),
        "hugging_face_id": model.get("hugging_face_id"),
        "created": model.get("created", 0),
        # Benchmarks filled in by merge_benchmarks(); placeholders keep shape stable.
        "aa": None,
        "da": None,
    }


def get_fal_models():
    """Fetch all models from FAL API"""
    all_models = []
    cursor = None
    max_retries = 3

    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor

        resp = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(f"{FAL_API_URL}/models", params=params, timeout=30)
                if resp.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    print(f"FAL rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise e

        if resp is None:
            break

        data = resp.json()
        all_models.extend(data.get("models", []))

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        # Small delay between pages to avoid rate limiting
        time.sleep(0.2)

    return all_models


def get_fal_pricing(endpoint_ids, api_key=None):
    """Fetch pricing for FAL models (requires API key)"""
    if not api_key:
        return {}

    headers = {"Authorization": f"Key {api_key}"}
    pricing_map = {}

    # Batch requests (max 50 per request)
    for i in range(0, len(endpoint_ids), 50):
        batch = endpoint_ids[i:i + 50]
        params = {"endpoint_id": ",".join(batch)}

        try:
            resp = requests.get(
                f"{FAL_API_URL}/models/pricing",
                params=params,
                headers=headers,
                timeout=30
            )
            if resp.status_code == 200:
                for p in resp.json().get("prices", []):
                    pricing_map[p["endpoint_id"]] = {
                        "unit_price": p.get("unit_price"),
                        "unit": p.get("unit"),
                        "currency": p.get("currency", "USD")
                    }
        except Exception as e:
            print(f"FAL pricing error: {e}")

    return pricing_map


def enrich_fal_model(model, pricing_map=None):
    """Enrich a FAL model with metadata"""
    endpoint_id = model.get("endpoint_id", "")
    metadata = model.get("metadata", {})

    # Get pricing if available
    pricing = {}
    if pricing_map and endpoint_id in pricing_map:
        pricing = pricing_map[endpoint_id]

    return {
        "id": endpoint_id,
        "name": metadata.get("display_name", endpoint_id.split("/")[-1]),
        "description": metadata.get("description", "")[:300],
        "category": metadata.get("category", "unknown"),
        "status": metadata.get("status", "active"),
        "tags": metadata.get("tags", []),
        "license": metadata.get("license_type", "unknown"),
        "thumbnail_url": metadata.get("thumbnail_url"),
        "thumbnail_animated_url": metadata.get("thumbnail_animated_url"),
        "model_url": metadata.get("model_url", f"https://fal.ai/models/{endpoint_id}"),
        "updated_at": metadata.get("updated_at"),
        "highlighted": metadata.get("highlighted", False),
        "pinned": metadata.get("pinned", False),
        "duration_estimate": metadata.get("duration_estimate"),
        "group": metadata.get("group", {}),
        # Pricing
        "price": pricing.get("unit_price"),
        "price_unit": pricing.get("unit"),
        "currency": pricing.get("currency", "USD"),
    }


def fetch_all_fal_data():
    """Fetch and enrich all FAL models"""
    print("Fetching FAL models list...")
    models = get_fal_models()
    total = len(models)
    print(f"Found {total} FAL models")

    # Get pricing if API key is available
    api_key = os.environ.get("FAL_API_KEY")
    pricing_map = {}

    if api_key:
        print("Fetching FAL pricing...")
        endpoint_ids = [m.get("endpoint_id") for m in models if m.get("endpoint_id")]
        pricing_map = get_fal_pricing(endpoint_ids, api_key)
        print(f"Got pricing for {len(pricing_map)} models")
    else:
        print("No FAL_API_KEY set - pricing will not be available")

    enriched = []
    for m in models:
        try:
            enriched.append(enrich_fal_model(m, pricing_map))
        except Exception as e:
            print(f"Error enriching FAL model: {e}")

    print(f"Enriched {len(enriched)} FAL models")
    return enriched


def get_cached_fal_models(force_refresh=False):
    """Get FAL models from cache or fetch fresh"""
    global cached_fal_models, fal_cache_timestamp

    with fal_cache_lock:
        now = time.time()

        # Check if cache is valid
        if not force_refresh and cached_fal_models and (now - fal_cache_timestamp) < FAL_CACHE_TTL:
            return cached_fal_models

        # Try to load from file cache
        if not force_refresh and os.path.exists(FAL_CACHE_FILE):
            try:
                with open(FAL_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    if now - data.get("timestamp", 0) < FAL_CACHE_TTL:
                        cached_fal_models = data["models"]
                        fal_cache_timestamp = data["timestamp"]
                        print(f"Loaded {len(cached_fal_models)} FAL models from cache")
                        return cached_fal_models
            except Exception as e:
                print(f"FAL cache load error: {e}")

        # Fetch fresh data
        cached_fal_models = fetch_all_fal_data()
        fal_cache_timestamp = now

        # Save to file
        try:
            with open(FAL_CACHE_FILE, 'w') as f:
                json.dump({"timestamp": fal_cache_timestamp, "models": cached_fal_models}, f)
        except Exception as e:
            print(f"FAL cache save error: {e}")

        return cached_fal_models


def fetch_benchmarks(slugs):
    """Fetch AA + Design Arena benchmarks for a list of canonical slugs, in parallel."""
    result = {}

    def one(slug):
        return slug, {"aa": get_aa_benchmark(slug), "da": get_da_benchmark(slug)}

    with ThreadPoolExecutor(max_workers=40) as ex:
        for fut in as_completed([ex.submit(one, s) for s in slugs]):
            try:
                slug, data = fut.result()
                result[slug] = data
            except Exception as e:
                print(f"Benchmark fetch error: {e}")
    hits = sum(1 for v in result.values() if v["aa"] or v["da"])
    print(f"Benchmarks: {hits}/{len(slugs)} models have AA or Design Arena data")
    return result


def get_cached_benchmarks(slugs, force_refresh=False):
    """Benchmarks on a slower (24h) cache than live perf/pricing."""
    global cached_benchmarks, bench_cache_timestamp, bench_cache_loading

    with bench_cache_lock:
        now = time.time()
        if not force_refresh and cached_benchmarks and (now - bench_cache_timestamp) < BENCH_CACHE_TTL:
            return cached_benchmarks

        if not force_refresh and os.path.exists(BENCH_CACHE_FILE):
            try:
                with open(BENCH_CACHE_FILE) as f:
                    data = json.load(f)
                if now - data.get("timestamp", 0) < BENCH_CACHE_TTL:
                    cached_benchmarks = data["benchmarks"]
                    bench_cache_timestamp = data["timestamp"]
                    print(f"Loaded benchmarks for {len(cached_benchmarks)} models from cache")
                    return cached_benchmarks
            except Exception as e:
                print(f"Benchmark cache load error: {e}")

        bench_cache_loading = True
        try:
            cached_benchmarks = fetch_benchmarks(slugs)
            bench_cache_timestamp = now
            try:
                with open(BENCH_CACHE_FILE, "w") as f:
                    json.dump({"timestamp": now, "benchmarks": cached_benchmarks}, f)
            except Exception as e:
                print(f"Benchmark cache save error: {e}")
        finally:
            bench_cache_loading = False
        return cached_benchmarks


def merge_benchmarks(enriched, benchmarks):
    """Attach cached benchmark records onto enriched models by canonical_slug."""
    for m in enriched:
        b = benchmarks.get(m["canonical_slug"])
        if b:
            m["aa"] = b.get("aa")
            m["da"] = b.get("da")
    return enriched


def fetch_all_data():
    """Fetch and enrich all models (live perf + providers + metadata), then merge benchmarks."""
    print("Fetching models list...")
    models = get_all_models()
    total = len(models)
    print(f"Found {total} models, fetching live perf + provider data...")

    enriched = []
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_model = {executor.submit(enrich_model, m): m for m in models}
        for future in as_completed(future_to_model):
            try:
                enriched.append(future.result())
                completed += 1
                if completed % 25 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  Progress: {completed}/{total} ({rate:.1f}/s, ETA: {eta:.0f}s)")
            except Exception as e:
                print(f"Error enriching model: {e}")

    # Merge benchmarks (own 24h cache, keyed by canonical_slug)
    slugs = sorted({m["canonical_slug"] for m in enriched})
    benchmarks = get_cached_benchmarks(slugs)
    merge_benchmarks(enriched, benchmarks)

    elapsed = time.time() - start_time
    print(f"Enriched {len(enriched)} models in {elapsed:.1f}s")
    return enriched


def get_cached_models(force_refresh=False):
    """Get models from cache or fetch fresh"""
    global cached_models, cache_timestamp

    with cache_lock:
        now = time.time()

        # Check if cache is valid
        if not force_refresh and cached_models and (now - cache_timestamp) < CACHE_TTL:
            return cached_models

        # Try to load from file cache
        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    if now - data.get("timestamp", 0) < CACHE_TTL:
                        cached_models = data["models"]
                        cache_timestamp = data["timestamp"]
                        print(f"Loaded {len(cached_models)} models from cache")
                        return cached_models
            except Exception as e:
                print(f"Cache load error: {e}")

        # Fetch fresh data
        cached_models = fetch_all_data()
        cache_timestamp = now

        # Save to file
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump({"timestamp": cache_timestamp, "models": cached_models}, f)
        except Exception as e:
            print(f"Cache save error: {e}")

        return cached_models


class APIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/models":
            self.handle_models(parsed)
        elif parsed.path == "/api/refresh":
            self.handle_refresh()
        elif parsed.path == "/api/stats":
            self.handle_stats()
        elif parsed.path == "/api/media/models":
            self.handle_fal_models(parsed)
        elif parsed.path == "/api/media/stats":
            self.handle_fal_stats()
        elif parsed.path == "/api/media/refresh":
            self.handle_fal_refresh()
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.serve_html()
        elif parsed.path == "/media" or parsed.path == "/media/":
            self.serve_html("fal.html")
        elif parsed.path == "/blog" or parsed.path == "/blog/":
            self.serve_html("blog.html")
        elif parsed.path.startswith("/blog/"):
            # Individual blog posts - serve the same page (SPA-style)
            self.serve_html("blog.html")
        elif parsed.path == "/favicon.svg":
            self.serve_static("favicon.svg", "image/svg+xml")
        elif parsed.path == "/site.webmanifest":
            self.serve_static("site.webmanifest", "application/manifest+json")
        else:
            super().do_GET()

    def handle_models(self, parsed):
        params = parse_qs(parsed.query)
        models = cached_models  # Use cached directly, don't block

        # Apply filters
        filtered = self.apply_filters(models, params)

        # Apply sorting
        sort_by = params.get("sort", ["throughput"])[0]
        sort_desc = params.get("desc", ["true"])[0] == "true"
        filtered = self.apply_sort(filtered, sort_by, sort_desc)

        # Pagination
        limit = int(params.get("limit", [100])[0])
        offset = int(params.get("offset", [0])[0])

        result = {
            "loading": cache_loading and len(cached_models) == 0,
            "total": len(filtered),
            "models": filtered[offset:offset + limit]
        }

        self.send_json(result)

    def apply_filters(self, models, params):
        filtered = models

        # Search filter
        if "q" in params:
            q = params["q"][0].lower()
            filtered = [m for m in filtered if q in m["id"].lower() or q in m["name"].lower()]

        # Min throughput
        if "min_throughput" in params:
            min_tp = float(params["min_throughput"][0])
            filtered = [m for m in filtered if (m["throughput"] or 0) >= min_tp]

        # Max latency
        if "max_latency" in params:
            max_lat = float(params["max_latency"][0])
            filtered = [m for m in filtered if m["latency"] and m["latency"] <= max_lat]

        # Min price (input)
        if "min_price_in" in params:
            min_price = float(params["min_price_in"][0])
            filtered = [m for m in filtered if m["price_input"] and m["price_input"] >= min_price]

        # Max price (input)
        if "max_price" in params:
            max_price = float(params["max_price"][0])
            filtered = [m for m in filtered if m["price_input"] and m["price_input"] <= max_price]

        # Max price (output)
        if "max_price_out" in params:
            max_price_out = float(params["max_price_out"][0])
            filtered = [m for m in filtered if m["price_output"] and m["price_output"] <= max_price_out]

        # Min context length
        if "min_context" in params:
            min_ctx = int(params["min_context"][0])
            filtered = [m for m in filtered if (m["context_length"] or 0) >= min_ctx]

        # Modality filter
        if "modality" in params:
            mod = params["modality"][0].lower()
            filtered = [m for m in filtered if mod in m["modality"].lower()]

        # Provider filter
        if "provider" in params:
            provider = params["provider"][0].lower()
            filtered = [m for m in filtered if any(provider in p.lower() for p in m["providers"])]

        # Has vision
        if "vision" in params and params["vision"][0] == "true":
            filtered = [m for m in filtered if "image" in m["input_modalities"]]

        return filtered

    def apply_sort(self, models, sort_by, desc):
        def get_sort_key(m):
            val = m.get(sort_by)
            if val is None:
                return float('-inf') if desc else float('inf')
            return val

        return sorted(models, key=get_sort_key, reverse=desc)

    def handle_refresh(self):
        models = get_cached_models(force_refresh=True)
        self.send_json({"status": "ok", "count": len(models)})

    def handle_fal_models(self, parsed):
        """Handle FAL models API endpoint"""
        params = parse_qs(parsed.query)
        models = cached_fal_models

        # Apply filters
        filtered = self.apply_fal_filters(models, params)

        # Apply sorting
        sort_by = params.get("sort", ["name"])[0]
        sort_desc = params.get("desc", ["false"])[0] == "true"
        filtered = self.apply_sort(filtered, sort_by, sort_desc)

        # Pagination
        limit = int(params.get("limit", [200])[0])
        offset = int(params.get("offset", [0])[0])

        result = {
            "loading": fal_cache_loading and len(cached_fal_models) == 0,
            "total": len(filtered),
            "models": filtered[offset:offset + limit]
        }

        self.send_json(result)

    def apply_fal_filters(self, models, params):
        """Apply filters for FAL models"""
        filtered = models

        # Search filter
        if "q" in params:
            q = params["q"][0].lower()
            filtered = [m for m in filtered if
                        q in m["id"].lower() or
                        q in m["name"].lower() or
                        q in m.get("description", "").lower() or
                        any(q in tag.lower() for tag in m.get("tags", []))]

        # Category filter
        if "category" in params:
            cat = params["category"][0].lower()
            if cat != "all":
                filtered = [m for m in filtered if m.get("category", "").lower() == cat]

        # Status filter (default: active only)
        status = params.get("status", ["active"])[0].lower()
        if status != "all":
            filtered = [m for m in filtered if m.get("status", "").lower() == status]

        # Highlighted only
        if params.get("highlighted", ["false"])[0] == "true":
            filtered = [m for m in filtered if m.get("highlighted")]

        # License filter
        if "license" in params:
            lic = params["license"][0].lower()
            if lic != "all":
                filtered = [m for m in filtered if m.get("license", "").lower() == lic]

        # Max price filter
        if "max_price" in params:
            max_price = float(params["max_price"][0])
            filtered = [m for m in filtered if m.get("price") is None or m["price"] <= max_price]

        return filtered

    def handle_fal_stats(self):
        """Handle FAL stats API endpoint"""
        models = cached_fal_models

        if not models:
            self.send_json({"loading": True, "total_models": 0})
            return

        # Count by category
        categories = {}
        licenses = set()
        for m in models:
            cat = m.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            licenses.add(m.get("license", "unknown"))

        stats = {
            "loading": fal_cache_loading,
            "total_models": len(models),
            "categories": categories,
            "licenses": list(licenses),
            "with_pricing": len([m for m in models if m.get("price") is not None]),
            "highlighted": len([m for m in models if m.get("highlighted")]),
        }

        self.send_json(stats)

    def handle_fal_refresh(self):
        """Force refresh FAL cache"""
        models = get_cached_fal_models(force_refresh=True)
        self.send_json({"status": "ok", "count": len(models)})

    def handle_stats(self):
        models = cached_models

        if not models:
            self.send_json({"loading": True, "total_models": 0})
            return

        def has_cap(m, cap):
            return (m.get("capabilities") or {}).get(cap)

        stats = {
            "loading": cache_loading,
            "total_models": len(models),
            "with_throughput": len([m for m in models if m["throughput"] > 0]),
            "with_latency": len([m for m in models if m["latency"]]),
            "with_benchmarks": len([m for m in models if m.get("aa") or m.get("da")]),
            "with_intelligence": len([m for m in models if (m.get("aa") or {}).get("intelligence") is not None]),
            "with_tools": len([m for m in models if has_cap(m, "tools")]),
            "with_reasoning": len([m for m in models if has_cap(m, "reasoning")]),
            "providers": list(set(p for m in models for p in m["providers"])),
            "modalities": list(set(m["modality"] for m in models)),
            "max_context": max((m["context_length"] or 0) for m in models) if models else 0,
            "bench_loading": bench_cache_loading,
        }

        self.send_json(stats)

    def serve_html(self, filename="index.html"):
        html_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(html_path):
            with open(html_path, 'rb') as f:
                content = f.read()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except BrokenPipeError:
                pass  # Client disconnected
        else:
            self.send_error(404)

    def serve_static(self, filename, content_type):
        file_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                content = f.read()
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(content)
            except BrokenPipeError:
                pass
        else:
            self.send_error(404)

    def send_json(self, data):
        content = json.dumps(data).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except BrokenPipeError:
            pass  # Client disconnected

    def log_message(self, format, *args):  # noqa: A002
        if args and isinstance(args[0], str) and "/api/" in args[0]:
            print(f"[API] {args[0]}")


cache_loading = False


def warmup_cache_background():
    """Fetch models in background"""
    global cache_loading
    cache_loading = True
    try:
        get_cached_models(force_refresh=True)
    finally:
        cache_loading = False


def warmup_fal_cache_background():
    """Fetch FAL models in background"""
    global fal_cache_loading
    fal_cache_loading = True
    try:
        get_cached_fal_models(force_refresh=True)
    finally:
        fal_cache_loading = False


def main():
    port = int(os.environ.get("PORT", 8765))

    print(f"Starting OpenRouter Explorer on http://localhost:{port}")

    # Try to load from disk cache first
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                global cached_models, cache_timestamp
                cached_models = data.get("models", [])
                cache_timestamp = data.get("timestamp", 0)
                print(f"Loaded {len(cached_models)} models from disk cache")
        except Exception as e:
            print(f"Cache load failed: {e}")

    # Load FAL cache from disk
    if os.path.exists(FAL_CACHE_FILE):
        try:
            with open(FAL_CACHE_FILE, 'r') as f:
                data = json.load(f)
                global cached_fal_models, fal_cache_timestamp
                cached_fal_models = data.get("models", [])
                fal_cache_timestamp = data.get("timestamp", 0)
                print(f"Loaded {len(cached_fal_models)} FAL models from disk cache")
        except Exception as e:
            print(f"FAL cache load failed: {e}")

    # Start fetching in background - don't block server startup
    print("Starting background model fetch...")
    threading.Thread(target=warmup_cache_background, daemon=True).start()
    threading.Thread(target=warmup_fal_cache_background, daemon=True).start()

    server = HTTPServer(("", port), APIHandler)
    print(f"Server ready at http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
