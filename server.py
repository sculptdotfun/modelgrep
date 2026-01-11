#!/usr/bin/env python3
"""
OpenRouter Explorer - Backend API
Fetches and serves model data with filtering capabilities
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import threading
import time

BASE_URL = "https://openrouter.ai/api/v1"
FRONTEND_STATS_URL = "https://openrouter.ai/api/frontend/stats/endpoint"
FAL_API_URL = "https://api.fal.ai/v1"
CACHE_FILE = "model_cache.json"
FAL_CACHE_FILE = "fal_cache.json"
CACHE_TTL = 300  # 5 minutes
FAL_CACHE_TTL = 21600  # 6 hours

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


def get_frontend_stats(model_slug):
    """Fetch frontend stats for a model (includes throughput/latency)"""
    try:
        resp = requests.get(
            FRONTEND_STATS_URL,
            params={"permaslug": model_slug, "variant": "standard"},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


def extract_stats(model):
    """Extract best throughput/latency from endpoints"""
    endpoints_data = model.get("endpoints") or {}
    endpoints_list = endpoints_data.get("endpoints", [])

    best_throughput = 0
    best_latency = float('inf')
    providers = set()

    for ep in endpoints_list:
        provider = ep.get("provider_name", "Unknown")
        providers.add(provider)

        throughput_data = ep.get("throughput_last_30m") or {}
        tp = throughput_data.get("p50") or 0

        latency_data = ep.get("latency_last_30m") or {}
        lat = latency_data.get("p50") or float('inf')

        if tp > best_throughput:
            best_throughput = tp
        if lat < best_latency:
            best_latency = lat

    return {
        "throughput": round(best_throughput, 2),
        "latency": round(best_latency, 2) if best_latency != float('inf') else None,
        "providers": list(providers)
    }


def get_price(model):
    """Extract pricing info"""
    pricing = model.get("pricing", {})
    try:
        prompt = float(pricing.get("prompt", "0")) * 1_000_000
        completion = float(pricing.get("completion", "0")) * 1_000_000
        return {"input": round(prompt, 4), "output": round(completion, 4)}
    except:
        return {"input": None, "output": None}


def extract_provider_details(endpoints_data):
    """Extract per-provider details from endpoints"""
    endpoints_list = endpoints_data.get("endpoints", []) if endpoints_data else []
    provider_details = []

    for ep in endpoints_list:
        pricing = ep.get("pricing", {})
        latency_data = ep.get("latency_last_30m") or {}
        throughput_data = ep.get("throughput_last_30m") or {}

        try:
            price_in = float(pricing.get("prompt", "0")) * 1_000_000
            price_out = float(pricing.get("completion", "0")) * 1_000_000
        except:
            price_in = None
            price_out = None

        lat = latency_data.get("p50")
        tp = throughput_data.get("p50")
        uptime = ep.get("uptime_last_30m")

        provider_details.append({
            "name": ep.get("provider_name", "Unknown"),
            "quantization": ep.get("quantization", "unknown"),
            "context_length": ep.get("context_length") or 0,
            "max_completion": ep.get("max_completion_tokens"),
            "latency": round(lat, 2) if lat is not None else None,
            "throughput": round(tp, 2) if tp is not None else None,
            "price_input": round(price_in, 4) if price_in is not None else None,
            "price_output": round(price_out, 4) if price_out is not None else None,
            "uptime": round(uptime, 1) if uptime is not None else None,
        })

    return provider_details


def enrich_model(model):
    """Enrich a single model with endpoint data from frontend stats API"""
    model_id = model["id"]

    # Use frontend stats API which has accurate throughput/latency
    frontend_endpoints = get_frontend_stats(model_id)

    # Extract stats and provider details from frontend data
    best_throughput = 0
    best_latency = float('inf')
    providers = set()
    provider_details = []

    for ep in frontend_endpoints:
        provider = ep.get("provider_name", "Unknown")
        providers.add(provider)

        stats = ep.get("stats") or {}
        pricing = ep.get("pricing") or {}

        tp = stats.get("p50_throughput") or 0
        lat = stats.get("p50_latency")

        if tp > best_throughput:
            best_throughput = tp
        if lat is not None and lat < best_latency:
            best_latency = lat

        # Extract pricing
        try:
            price_in = float(pricing.get("prompt", "0")) * 1_000_000
            price_out = float(pricing.get("completion", "0")) * 1_000_000
        except:
            price_in = None
            price_out = None

        provider_details.append({
            "name": provider,
            "quantization": ep.get("quantization", "unknown"),
            "context_length": ep.get("context_length") or 0,
            "max_completion": ep.get("max_completion_tokens"),
            "latency": round(lat, 2) if lat is not None else None,
            "throughput": round(tp, 2) if tp > 0 else None,
            "price_input": round(price_in, 4) if price_in is not None else None,
            "price_output": round(price_out, 4) if price_out is not None else None,
            "uptime": None,  # Not available in frontend API
        })

    # Get pricing from model data as fallback
    price = get_price(model)
    arch = model.get("architecture", {})

    return {
        "id": model_id,
        "name": model.get("name", model_id),
        "description": model.get("description", "")[:200],
        "context_length": model.get("context_length", 0),
        "throughput": round(best_throughput, 2) if best_throughput > 0 else 0,
        "latency": round(best_latency, 2) if best_latency != float('inf') else None,
        "price_input": price["input"],
        "price_output": price["output"],
        "providers": list(providers),
        "provider_details": provider_details,
        "modality": arch.get("modality", "text->text"),
        "input_modalities": arch.get("input_modalities", ["text"]),
        "output_modalities": arch.get("output_modalities", ["text"]),
        "top_provider": model.get("top_provider", {}),
        "created": model.get("created", 0),
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


def fetch_all_data():
    """Fetch and enrich all models"""
    print("Fetching models list...")
    models = get_all_models()
    total = len(models)
    print(f"Found {total} models, fetching endpoint stats...")

    enriched = []
    completed = 0
    start_time = time.time()

    # Use more workers for faster fetching
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_model = {
            executor.submit(enrich_model, m): m
            for m in models
        }

        for future in as_completed(future_to_model):
            try:
                result = future.result()
                enriched.append(result)
                completed += 1
                if completed % 25 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  Progress: {completed}/{total} ({rate:.1f}/s, ETA: {eta:.0f}s)")
            except Exception as e:
                print(f"Error enriching model: {e}")

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

        stats = {
            "loading": cache_loading,
            "total_models": len(models),
            "with_throughput": len([m for m in models if m["throughput"] > 0]),
            "with_latency": len([m for m in models if m["latency"]]),
            "providers": list(set(p for m in models for p in m["providers"])),
            "modalities": list(set(m["modality"] for m in models)),
            "max_context": max((m["context_length"] or 0) for m in models) if models else 0,
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
