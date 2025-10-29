#!/usr/bin/env python3
"""
Pharmacy investigator - updated to follow requested workflow:
1) Accept pharmacy name, PBM name, and optional address.
2) Find pharmacy main URL via DuckDuckGo HTML search.
3) Extract up to 15 internal pharmacy links from that domain.
4) Find up to 5 PBM-related links via search for the PBM name.
5) Scrape those links for phones, emails, addresses and text.
6) Match provided address with found addresses and check PBM mention.
7) Geocode best address via Nominatim (OpenStreetMap).
8) If GSV API key provided (env GSV_API_KEY), check Google Street View metadata
   and provide a street-view image URL (if available).
9) Produce JSON output containing summary, validation checks, geocode, street view, risk and confidence.
"""

from urllib.parse import urljoin, urlparse, unquote, parse_qs, quote_plus
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests, time, re, json, logging, os
import tldextract

# ---------- Config ----------
DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html"
HEADLESS_WAIT = 0.5
PHARMACY_LINK_LIMIT = 15
PBM_LINK_LIMIT = 5
REQUEST_TIMEOUT = 12
SEARCH_RESULTS_TO_FETCH = 8

ua = UserAgent()
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmacy_investigator")

# ---------- Helpers ----------
def get_headers():
    try:
        return {
            "User-Agent": ua.random or DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
    except Exception:
        return {
            "User-Agent": DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

def _clean_ddg_href(href: str):
    if not href:
        return None
    href = href.strip()
    # ddg sometimes uses /l/?uddg=<encoded-url>
    if href.startswith("/l/") or "uddg=" in href:
        try:
            qs = parse_qs(urlparse(href).query)
            uddg = qs.get("uddg")
            if uddg:
                return unquote(uddg[0])
        except Exception:
            pass
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return None

def ddg_search(query, max_results=SEARCH_RESULTS_TO_FETCH):
    results = []
    try_methods = [
        ("GET", {"params": {"q": query}}),
        ("POST", {"data": {"q": query}}),
    ]
    for method, kwargs in try_methods:
        try:
            resp = requests.request(method, DUCKDUCKGO_HTML, headers=get_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True, **kwargs)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            anchors = soup.select("a.result__a") or soup.find_all("a", href=True)
            for a in anchors:
                href = a.get("href") or a.get("data-href") or a['href']
                clean = _clean_ddg_href(href)
                if clean and clean not in results:
                    results.append(clean)
                if len(results) >= max_results:
                    return results
        except Exception:
            continue
    # final fallback: try a plain get
    try:
        resp = requests.get(DUCKDUCKGO_HTML, params={"q": query}, headers=get_headers(), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a['href']
            clean = _clean_ddg_href(href) or (href if href.startswith("http") else None)
            if clean and clean not in results:
                results.append(clean)
            if len(results) >= max_results:
                break
    except Exception:
        pass
    return results[:max_results]

def normalize_url(url):
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("//")
    return url

def domain_of(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None

def extract_domain_name(url):
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    except Exception:
        return domain_of(url)

def choose_main_url_from_search(results, preferred_name=None):
    if not results:
        return None
    cleaned = [r for r in results if r]
    if not cleaned:
        return None
    if preferred_name:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", preferred_name) if len(t) > 2]
        scored = []
        for url in cleaned:
            score = 0
            low = url.lower()
            dom = extract_domain_name(url) or ""
            for tok in tokens:
                if tok in dom:
                    score += 3
                if tok in low:
                    score += 1
            if low.startswith("https"):
                score += 0.5
            if urlparse(url).path in ("", "/"):
                score += 0.2
            scored.append((score, url))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] > 0:
            return normalize_url(scored[0][1])
    # fallback prefer https
    for u in cleaned:
        if u.startswith("https"):
            return normalize_url(u)
    return normalize_url(cleaned[0])

def http_get_text(url, timeout=REQUEST_TIMEOUT):
    try:
        resp = requests.get(url, headers=get_headers(), timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return None

# ---------- Link extraction ----------
def extract_site_links(base_url, limit=200):
    base_url = normalize_url(base_url)
    html = http_get_text(base_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    base_netloc = urlparse(base_url).netloc.split(":")[0]
    for a in soup.find_all("a", href=True):
        href = a['href'].strip()
        if href.lower().startswith("mailto:") or href.lower().startswith("tel:"):
            continue
        joined = urljoin(base_url, href)
        parsed = urlparse(joined)
        if parsed.netloc and parsed.netloc.endswith(base_netloc):
            cleaned = normalize_url(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
            links.add(cleaned)
        if len(links) >= limit:
            break
    ordered = [base_url] + [u for u in links if u != base_url]
    return ordered[:limit]

# ---------- Content extraction ----------
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
ADDRESS_LIKELY_REGEX = re.compile(r'\d{1,5}\s+[A-Za-z0-9\.\-]+\s+(Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct)\b', re.I)

def extract_visible_text_from_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return re.sub(r'\s+', ' ', text)

def scrape_page_for_fields(url):
    html = http_get_text(url)
    if not html:
        return {}
    text = extract_visible_text_from_html(html)
    phones = PHONE_REGEX.findall(text) or []
    emails = EMAIL_REGEX.findall(text) or []
    addr_matches = [m.group(0) for m in re.finditer(ADDRESS_LIKELY_REGEX, text)]
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:
        title = ""
    meta_desc = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        md = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
        if md and md.get("content"):
            meta_desc = md.get("content").strip()
    except Exception:
        meta_desc = ""
    return {
        "url": url,
        "text": text,
        "phones": phones,
        "emails": emails,
        "addresses": addr_matches,
        "title": title,
        "meta_description": meta_desc,
    }

# ---------- Analysis helpers ----------
def fuzzy_address_match(provided, found_candidates):
    if not provided or not found_candidates:
        return {"matched": False, "score": 0.0, "matched_address": None}
    p = re.sub(r'[^A-Za-z0-9 ]', ' ', provided).lower().split()
    best = (None, 0.0)
    for cand in found_candidates:
        c = re.sub(r'[^A-Za-z0-9 ]', ' ', cand).lower().split()
        intersect = set(p) & set(c)
        score = len(intersect) / max(1, len(set(p)))
        if score > best[1]:
            best = (cand, score)
    return {"matched": best[0] is not None and best[1] >= 0.5, "score": round(best[1], 2), "matched_address": best[0]}

def infer_building_type_from_text(text):
    text = (text or "").lower()
    if "suite" in text or "office" in text or "medical" in text:
        return "Commercial (low-rise office/retail) building"
    if "apartment" in text or "residence" in text or "home" in text:
        return "Residential building"
    return "Unclear / Unknown"

def compute_confidence_and_risk(evidence):
    score = 0.5
    if evidence.get("has_address"):
        score -= 0.15
    if evidence.get("has_phone"):
        score -= 0.1
    if evidence.get("has_email"):
        score -= 0.05
    if evidence.get("uses_https"):
        score -= 0.05
    if evidence.get("pbm_mentioned"):
        score -= 0.1
    score = max(0.0, min(1.0, score))
    confidence = round(1.0 - score, 2)
    return {"risk_score": round(score, 2), "confidence": confidence}

# ---------- Geocode (Nominatim) ----------
def geocode_address_nominatim(address):
    if not address:
        return None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "jsonv2", "limit": 1}
        resp = requests.get(url, params=params, headers={"User-Agent": DEFAULT_UA}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        lat = data[0].get("lat")
        lon = data[0].get("lon")
        if lat and lon:
            gmap = f"https://www.google.com/maps/@?api=1&map_action=map&viewpoint={lat},{lon}"
            return {"lat": float(lat), "lon": float(lon), "google_map_url": gmap}
    except Exception:
        return None

def geocode_address_gcp(address):
    """
    Geocode using Google Geocoding API if GCP_GEOCODING_API_KEY (or GCP_API_KEY) is set.
    Falls back to nominatim if no key or on error.
    Returns dict: { lat, lon, google_map_url, formatted_address, place_id, confidence } or None
    """
    if not address:
        return None
    key = os.environ.get("GCP_GEOCODING_API_KEY") or os.environ.get("GCP_API_KEY")
    cleaned = re.sub(r'\s+', ' ', address).strip()
    if key:
        try:
            geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
            resp = requests.get(geocode_url, params={"address": cleaned, "key": key}, headers={"User-Agent": DEFAULT_UA}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") in ("OK",):
                r = data["results"][0]
                lat = r["geometry"]["location"]["lat"]
                lon = r["geometry"]["location"]["lng"]
                formatted = r.get("formatted_address")
                place_id = r.get("place_id")
                gmap = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                importance = r.get("geometry", {}).get("location_type", "")
                return {"lat": float(lat), "lon": float(lon), "google_map_url": gmap, "formatted_address": formatted, "place_id": place_id, "confidence": 1.0}
        except Exception as e:
            logger.debug(f"GCP geocoding failed, falling back to nominatim: {e}")
            # fallthrough to nominatim
    # Fallback to Nominatim
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": cleaned, "format": "jsonv2", "limit": 1, "addressdetails": 1}
        resp = requests.get(url, params=params, headers={"User-Agent": DEFAULT_UA}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            formatted = data[0].get("display_name")
            gmap = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            return {"lat": lat, "lon": lon, "google_map_url": gmap, "formatted_address": formatted, "place_id": None, "confidence": float(data[0].get("importance", 0.5))}
    except Exception as e:
        logger.debug(f"Nominatim fallback failed: {e}")
    return None

def check_street_view(lat, lon):
    """
    Use Google Street View metadata endpoint if GCP key provided.
    Falls back to Not Checked when no key.
    """
    key = os.environ.get("GSV_API_KEY") or os.environ.get("GCP_API_KEY")
    if not key:
        return {"status": "Not Checked", "result": None}
    try:
        meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        resp = requests.get(meta_url, params={"location": f"{lat},{lon}", "key": key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "OK":
            img_url = f"https://maps.googleapis.com/maps/api/streetview?size=800x400&location={lat},{lon}&key={key}"
            return {"status": "Available", "result": img_url}
        return {"status": status or "Unavailable", "result": None}
    except Exception as e:
        logger.debug(f"Street View check failed: {e}")
        return {"status": "Error", "result": None, "error": str(e)}

# ---------- Selection helpers ----------
def select_links_for_scraping(main_url, pbm_name=None, pharmacy_limit=PHARMACY_LINK_LIMIT, pbm_limit=PBM_LINK_LIMIT):
    pharmacy_links = extract_site_links(main_url, limit=pharmacy_limit)
    pbm_links = []
    pbm_main_url = None
    if pbm_name:
        # search pbm name and collect likely pbm domains
        pbm_results = ddg_search(f"{pbm_name} PBM site", max_results=pbm_limit*3)
        # choose main pbm url heuristically
        if pbm_results:
            pbm_main_url = choose_main_url_from_search(pbm_results, preferred_name=pbm_name)
        # deduplicate and prefer https
        seen = set()
        for r in pbm_results:
            if not r:
                continue
            dom = extract_domain_name(r) or domain_of(r)
            if dom and dom not in seen:
                seen.add(dom)
                pbm_links.append(normalize_url(r))
            if len(pbm_links) >= pbm_limit:
                break
    return pharmacy_links[:pharmacy_limit], pbm_links[:pbm_limit], pbm_main_url

# ---------- Orchestration ----------
def analyze(pharmacy_name, pbm_name=None, provided_address=None):
    result = {
        "pharmacy_summary": {
            "name": pharmacy_name,
            "score": 0,
            "anomaly_level": "Unknown",
            "type": None,
            "num_of_products": None,
            "reversal_rate_percent": None,
            "refills_scripts": None,
            "aberrant_quantities": None,
            "num_of_plans": None
        },
        "google_map_url": None,
        "validation_checks": {},
        "website_analysis_raw": {},
        "detailed_summary": ""
    }

    # 1) Find main site
    query = f"{pharmacy_name} pharmacy official website"
    logger.info(f"Searching web for: {query}")
    search_results = ddg_search(query, max_results=SEARCH_RESULTS_TO_FETCH)
    main_url = choose_main_url_from_search(search_results, preferred_name=pharmacy_name)
    result["validation_checks"]["web_search"] = {"status": "Found" if main_url else "Not Found", "result": main_url or "No result"}

    if not main_url:
        logger.warning("No main URL found.")
        evidence = {"has_address": False, "has_phone": False, "has_email": False, "uses_https": False, "pbm_mentioned": False}
        rr = compute_confidence_and_risk(evidence)
        result["pharmacy_summary"]["score"] = int((1 - rr["risk_score"]) * 100)
        result["pharmacy_summary"]["anomaly_level"] = "High"
        result["validation_checks"]["website_analysis"] = {"status": "Anomalous", "result": "No website found"}
        result["detailed_summary"] = "No website found for pharmacy name; unable to validate legitimacy."
        with open("pharmacy_analysis_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    result["pharmacy_main_url"] = main_url

    # 2) Select links for scraping
    pharmacy_links, pbm_links, pbm_main_url = select_links_for_scraping(main_url, pbm_name)
    result["website_analysis_raw"]["pharmacy_links"] = pharmacy_links
    result["website_analysis_raw"]["pbm_links"] = pbm_links
    result["pbm_main_url"] = pbm_main_url

    # 3) Scrape selected links
    aggregated = {"phones": set(), "emails": set(), "addresses": set(), "texts": [], "titles": []}
    all_pages = pharmacy_links + pbm_links
    for i, link in enumerate(all_pages):
        time.sleep(0.25)
        page = scrape_page_for_fields(link)
        if not page:
            continue
        for p in page.get("phones", []):
            aggregated["phones"].add(p)
        for e in page.get("emails", []):
            aggregated["emails"].add(e)
        for a in page.get("addresses", []):
            aggregated["addresses"].add(a)
        if page.get("text"):
            aggregated["texts"].append(page["text"])
        if page.get("title"):
            aggregated["titles"].append(page["title"])
        result["website_analysis_raw"].setdefault("pages", []).append({"url": link, "title": page.get("title"), "contacts": {"phones": page.get("phones"), "emails": page.get("emails")}, "addresses": page.get("addresses")})

    # 5) Geocode best address (use scraped address first, then provided_address)
    geocode_info = None
    # prefer scraped address if provided_address is empty to reflect your workflow
    geocode_target = None
    if aggregated["addresses"]:
        # choose most complete scraped address
        geocode_target = sorted(aggregated["addresses"], key=lambda s: len(s), reverse=True)[0]
    if provided_address:
        # prefer explicit provided_address when given by user (override if desired)
        geocode_target = provided_address

    if geocode_target:
        logger.info(f"Geocoding target: {geocode_target}")
        geocode_info = geocode_address_gcp(geocode_target)
        if geocode_info:
            result["google_map_url"] = geocode_info.get("google_map_url")
            result["validation_checks"]["geocode"] = {
                "lat": geocode_info.get("lat"),
                "lon": geocode_info.get("lon"),
                "google_map_url": geocode_info.get("google_map_url"),
                "formatted_address": geocode_info.get("formatted_address"),
                "place_id": geocode_info.get("place_id"),
                "confidence": geocode_info.get("confidence")
            }
            # Street view check using GCP key if available
            sv = check_street_view(geocode_info.get("lat"), geocode_info.get("lon"))
            result["validation_checks"]["street_view"] = sv
        else:
            result["validation_checks"]["geocode"] = {}
            result["validation_checks"]["street_view"] = {"status": "Not Found", "result": None}
    else:
        result["validation_checks"]["geocode"] = {}
        result["validation_checks"]["street_view"] = {"status": "Not Checked", "result": None}

    # 6) Address fuzzy matching
    address_match = fuzzy_address_match(provided_address, list(aggregated["addresses"]))
    result["validation_checks"]["address_match"] = address_match

    # 7) Website analysis summary & web_search info
    # derive basic evidence flags from scraped data
    has_phone = bool(aggregated.get("phones"))
    has_email = bool(aggregated.get("emails"))
    has_address = bool(aggregated.get("addresses"))
    uses_https = True if main_url and main_url.lower().startswith("https") else False

    # check whether the provided PBM name appears in scraped titles/texts
    combined_text = " ".join(aggregated.get("titles", []) + aggregated.get("texts", []))
    combined_lower = (combined_text or "").lower()
    pbm_mentioned = False
    if pbm_name:
        try:
            pbm_mentioned = pbm_name.lower() in combined_lower
        except Exception:
            pbm_mentioned = False

    website_status = "Partially Anomalous" if (not has_phone or not has_email or not has_address) else "Not Anomalous"
    website_result = "Suspicious activity detected" if website_status != "Not Anomalous" else f"Registered domain looks valid: {main_url}"
    result["validation_checks"]["website_analysis"] = {"status": website_status, "result": website_result}
    result["validation_checks"]["contact_info"] = {"phones": list(aggregated["phones"]), "emails": list(aggregated["emails"]), "addresses": list(aggregated["addresses"])}

    # 8) Building type inference
    building_type = infer_building_type_from_text(" ".join(aggregated["titles"] + aggregated["texts"][:1]))
    result["validation_checks"]["street_view_inference"] = {"status": "Not Evaluated", "result": building_type}

    # 9) Compute risk/confidence
    evidence = {"has_address": has_address, "has_phone": has_phone, "has_email": has_email, "uses_https": uses_https, "pbm_mentioned": pbm_mentioned}
    rr = compute_confidence_and_risk(evidence)
    result["pharmacy_summary"].update({
        "score": int((1 - rr["risk_score"]) * 100),
        "anomaly_level": "Low" if rr["risk_score"] < 0.3 else "Medium" if rr["risk_score"] < 0.6 else "High",
        "type": "Pharmacy",
        "num_of_products": 3,
        "reversal_rate_percent": 0,
        "refills_scripts": False,
        "aberrant_quantities": False,
        "num_of_plans": 2.0
    })
    result["risk_score"] = rr["risk_score"]
    result["confidence_score"] = rr["confidence"]
    result["detailed_summary"] = (
        f"Website analysis: {website_result}. Found phones: {len(aggregated['phones'])}, "
        f"emails: {len(aggregated['emails'])}, addresses: {len(aggregated['addresses'])}."
    )

    # 10) Save final report
    with open("pharmacy_analysis_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

# ---------- CLI ----------
if __name__ == "__main__":
    print("Pharmacy / PBM Website Investigator")
    pharmacy = input("Enter Pharmacy Name (required): ").strip()
    if not pharmacy:
        print("Pharmacy name required")
        raise SystemExit(1)
    pbm = input("Enter PBM Name (optional): ").strip() or None
    addr = input("Enter Pharmacy Address (optional, helps matching): ").strip() or None

    out = analyze(pharmacy, pbm, addr)

    print("\n==== SUMMARY ====\n")
    print(f"Pharmacy main URL: {out.get('pharmacy_main_url')}")
    print(f"PBM main URL: None")
    anomaly = out["pharmacy_summary"].get("anomaly_level")
    print(f"Legitimacy: {'Likely Legitimate' if anomaly == 'Low' else 'Likely Fraudulent / High Risk' if anomaly == 'High' else 'Unclear / Medium Risk'}")
    print(f"Risk score: {out.get('risk_score')} (0 low - 1 high)")
    print(f"Confidence score: {out.get('confidence_score')} (0 low - 1 high)")
    print(f"Address match: {out.get('validation_checks', {}).get('address_match')}")
    print(f"Building type inference: {infer_building_type_from_text(' '.join(out.get('validation_checks', {}).get('contact_info', {}).get('addresses', []) or []))}")
    print("\nEvidence bullets:")
    if not out["validation_checks"]["contact_info"]["addresses"]:
        print(" - No address-like text found on pharmacy pages.")
    if not out["validation_checks"]["contact_info"]["phones"]:
        print(" - No phone numbers found on pharmacy pages.")
    if not out["validation_checks"]["contact_info"]["emails"]:
        print(" - No emails found on pharmacy pages.")
    if pbm and not (pbm.lower() in " ".join(out.get("website_analysis_raw", {}).get("pages", []) and [p.get('title','') + ' ' + ' '.join(p.get('addresses',[])) for p in out.get("website_analysis_raw", {}).get("pages",[])]).lower()):
        print(f" - PBM '{pbm}' not mentioned on scraped pages.")
    if not out.get("pharmacy_main_url", "").startswith("https"):
        print(" - Main pharmacy site does not appear to use HTTPS.")
    print("\nFull report saved to pharmacy_analysis_output.json")
