from __future__ import annotations
from pathlib import Path
import re
import csv
import json
import math
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Set, Iterable, Optional
import ct_rag
rag_system = ct_rag.SCT_GraphRAG.load("rag_knowledge_graph.pkl")

# ------------------------------
# Regexes & helpers
# ------------------------------

POI_NAME_RE = re.compile(r"<([^>]+)>")
COORD_RE = re.compile(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)")
NEARBY_RE = re.compile(r"\[([^\]]*)\]\s*$")

_DIVERSITY_HINTS = [
    "Emphasize network reachability (shared neighbors on the road network) over raw geometric distance.",
    "Emphasize geometric proximity and address-locality cues over category coherence.",
    "Emphasize time-of-day compatibility and opening plausibility over distance if close.",
    "Emphasize category clusters (objects co-occurring in neighborhoods) with small-world effects.",
    "Balance all factors; break ties by denser local neighborhood core."
]

def _canon_text(s: str) -> str:
    """Unicode-friendly canonicalization: lowercase, strip, remove noisy punct, normalize spaces."""
    if s is None:
        return ""
    s = s.lower().strip()
    # keep word chars (\w includes unicode), whitespace, hyphen, middle-dot, slash
    s = re.sub(r"[^\w\s\-/·]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _canonical_name(raw: str) -> str:
    return _canon_text(raw)

def _tokenize(s: str) -> List[str]:
    """Split on whitespace and mild separators; keep unicode word tokens."""
    if not s:
        return []
    return [t for t in re.split(r"[\s/,.\-·]+", _canon_text(s)) if t]

# ------------------------------
# Parsing POI file
# ------------------------------

def parse_poi_line(line: str) -> Dict[str, Any]:
    """
    Parse a single POI line like:
    POI is <name>(major, medium, small). Location: ..., with specific coordinates at (lon, lat).
    The nearby POIs from nearest to farthest are as follows: [<n1>(...): 12.3 meters, <n2>(...): 45.6 meters, ...]
    """
    name_match = POI_NAME_RE.search(line)
    if not name_match:
        return {}
    name = name_match.group(1)

    # categories (first parentheses after name)
    after_name = line[name_match.end():]
    major = medium = small = ""
    cat_match = re.search(r"\(([^)]+)\)", after_name)
    if cat_match:
        cats = [c.strip() for c in cat_match.group(1).split(",")]
        if len(cats) == 3:
            major, medium, small = cats

    # coordinates
    lat = lon = None
    coord = COORD_RE.search(line)
    if coord:
        lon = float(coord.group(1))
        lat = float(coord.group(2))

    # address
    address = ""
    loc_match = re.search(r"Location:\s*(.*?)(?:,?\s*with specific coordinates|,?\s*The nearby POIs|$)", line)
    if loc_match:
        address = loc_match.group(1).strip()

    # nearby list
    nearby = []
    nb_match = NEARBY_RE.search(line)
    if nb_match:
        raw_list = nb_match.group(1)
        # split by '>,', then add back the '>' if missing
        parts = re.split(r">\s*,", raw_list)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not p.endswith(">"):
                p = p + ">"
            nm = POI_NAME_RE.search(p)
            if not nm:
                continue
            nname = nm.group(1)
            ctm = re.search(r">\s*\(([^)]+)\)", p)
            nmajor = nmedium = nsmall = ""
            if ctm:
                cc = [x.strip() for x in ctm.group(1).split(",")]
                if len(cc) == 3:
                    nmajor, nmedium, nsmall = cc
            dm = re.search(r":\s*([-\d\.]+)\s*meters", p)
            dist = None
            if dm:
                try:
                    dist = float(dm.group(1))
                except Exception:
                    dist = None
            nearby.append((nname, (nmajor, nmedium, nsmall), dist))

    return {
        "name": name,
        "major": major, "medium": medium, "small": small,
        "address": address,
        "lat": lat, "lon": lon,
        "nearby": nearby,
    }

def parse_poi_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse the entire POI file and ALSO backfill "stub" nodes for neighbors
    that never appear as primaries (improves recall/connectivity).
    Returns a dict: canonical_name -> record
    """
    pois: Dict[str, Dict[str, Any]] = {}
    neighbors_seen: List[Tuple[str, str, Tuple[str, str, str]]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = parse_poi_line(line)
            if rec and rec.get("name"):
                c = _canonical_name(rec["name"])
                pois[c] = rec
                for (nname, cats, _dist) in rec.get("nearby", []):
                    neighbors_seen.append((rec["name"], nname, cats))

    # Backfill neighbors that don't have records yet
    for owner, nname, cats in neighbors_seen:
        cn = _canonical_name(nname)
        if cn not in pois:
            pois[cn] = {
                "name": nname,
                "major": cats[0], "medium": cats[1], "small": cats[2],
                "address": "",
                "lat": None, "lon": None,
                "nearby": [],
            }
    return pois

# ------------------------------
# Graph + inverted indexes
# ------------------------------

def build_neighbor_graph(pois: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Undirected weighted graph G[u][v] = distance (meters, default 1.0 if unknown)
    """
    G: Dict[str, Dict[str, float]] = defaultdict(dict)
    for cname, rec in pois.items():
        for (nname, _cats, dist) in rec.get("nearby", []):
            v = _canonical_name(nname)
            if v == cname:
                continue
            w = float(dist) if dist is not None else 1.0
            if v not in G[cname] or w < G[cname][v]:
                G[cname][v] = w
            if cname not in G[v] or w < G[v][cname]:
                G[v][cname] = w
    return G

def build_inverted_indexes(pois: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns name_token_idx, addr_token_idx
    token -> set(canonical_name)
    """
    name_idx: Dict[str, Set[str]] = defaultdict(set)
    addr_idx: Dict[str, Set[str]] = defaultdict(set)
    for cname, rec in pois.items():
        for tok in _tokenize(rec["name"]):
            name_idx[tok].add(cname)
        for tok in _tokenize(rec.get("address", "")):
            addr_idx[tok].add(cname)
    return name_idx, addr_idx

# ------------------------------
# Prompt parsing & fuzzy matching
# ------------------------------

def extract_prompt_poi_names(traj_prompt: str) -> List[str]:
    return [_canonical_name(m.group(1)) for m in POI_NAME_RE.finditer(traj_prompt or "")]

def _trigram_set(s: str) -> Set[str]:
    s = "^" + _canon_text(s) + "$"
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s)-2)}

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def fuzzy_match_to_pois(name_like: str, pois: Dict[str, Dict[str, Any]], name_token_idx: Dict[str, Set[str]], top_k: int = 5) -> List[str]:
    """
    Fuzzy match a mention to known POIs:
    - collect candidates by token overlap with the POI names
    - score by trigram Jaccard between mention and candidate's name
    """
    tokens = _tokenize(name_like)
    cands: Set[str] = set()
    for t in tokens:
        cands |= name_token_idx.get(t, set())
    if not cands:
        return []
    s_tri = _trigram_set(name_like)
    scored = []
    for c in cands:
        tri = _trigram_set(pois[c]["name"])
        score = _jaccard(s_tri, tri)
        if score > 0:
            scored.append((score, c))
    scored.sort(reverse=True)
    return [c for (s, c) in scored[:top_k]]

# ------------------------------
# Candidate recall (multi-channel + scoring)
# ------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> Optional[float]:
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371000.0
    from math import radians, sin, cos, sqrt, atan2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R*c

def recall_candidates(
    traj_prompt: str,
    pois: Dict[str, Dict[str, Any]],
    G: Dict[str, Dict[str, float]],
    name_token_idx: Dict[str, Set[str]],
    addr_token_idx: Dict[str, Set[str]],
    max_candidates: int = 200,
    max_hops: int = 3,
    use_name_keywords: bool = True,
) -> List[str]:
    """
    Returns a list of candidate canonical names (ranked by composite score).
    Channels:
      - exact mentions from <...> in prompt
      - fuzzy matches for mentions that don't exactly match
      - address keyword hits (e.g., 'xiaonanjie') from prompt tokens
      - (optional) name keyword hits from prompt tokens
      - graph expansion with distance-decay & shared-neighbor bonus
      - centroid proximity bonus (if seed coords exist)
    """

    # 1) exact mentions
    mentions = extract_prompt_poi_names(traj_prompt)
    seeds: List[str] = [m for m in mentions if m in pois]
    seed_source: Dict[str, str] = {m: "mention" for m in seeds}

    # 2) fuzzy for non-hit mentions
    for m in mentions:
        if m not in pois:
            for cand in fuzzy_match_to_pois(m, pois, name_token_idx, top_k=10):
                if cand not in seed_source:
                    seeds.append(cand)
                    seed_source[cand] = "fuzzy"

    # tokens from entire prompt
    prompt_tokens = _tokenize(traj_prompt)

    # 3) address keyword hits
    addr_hits: Set[str] = set()
    for t in prompt_tokens:
        if t in addr_token_idx:
            addr_hits |= addr_token_idx[t]
    for a in addr_hits:
        if a not in seed_source:
            seeds.append(a)
            seed_source[a] = "addr_kw"

    # 4) (optional) name keyword hits
    if use_name_keywords:
        name_hits: Set[str] = set()
        for t in prompt_tokens:
            name_hits |= name_token_idx.get(t, set())
        for n in name_hits:
            if n not in seed_source:
                seeds.append(n)
                seed_source[n] = "name_kw"

    # No seeds -> no candidates
    if not seeds:
        return []

    # centroid from seed coords
    coords = [(pois[s]["lat"], pois[s]["lon"]) for s in seeds if pois[s]["lat"] is not None and pois[s]["lon"] is not None]
    centroid = None
    if coords:
        clat = sum(p[0] for p in coords) / len(coords)
        clon = sum(p[1] for p in coords) / len(coords)
        centroid = (clat, clon)

    # Scoring init with seed priors by source type
    scores: Dict[str, float] = defaultdict(float)
    SEED_PRIOR = {"mention": 2.0, "fuzzy": 1.4, "addr_kw": 1.1, "name_kw": 0.9}
    for s in seeds:
        scores[s] += SEED_PRIOR.get(seed_source.get(s, "mention"), 1.0)

    # BFS-like expansion up to max_hops with distance-decay
    visited = set(seeds)
    frontier = list(seeds)
    hops = 0
    while frontier and hops < max_hops and len(scores) < 10000:
        next_frontier: List[str] = []
        for u in frontier:
            for v, dist in sorted(G.get(u, {}).items(), key=lambda kv: kv[1]):
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
                d = max(1.0, dist or 1.0)
                scores[v] += 1.2 / (1.0 + math.log1p(d))  # distance decay
        frontier = next_frontier
        hops += 1

    # Shared-neighbor bonus relative to seed set
    seed_neighbors: Counter = Counter()
    for s in seeds:
        for v in G.get(s, {}):
            seed_neighbors[v] += 1
    for v, cnt in seed_neighbors.items():
        scores[v] += 0.4 * cnt

    # Centroid proximity bonus (if centroid known)
    if centroid is not None:
        clat, clon = centroid
        for v in list(scores.keys()):
            lat, lon = pois[v]["lat"], pois[v]["lon"]
            dd = _haversine(clat, clon, lat, lon)
            if dd is not None:
                scores[v] += 0.6 / (1.0 + math.log1p(dd))

    # Rank and keep top-N
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    cand = [c for c, s in ranked[:max_candidates]]
    return cand

# ------------------------------
# LLM prompt + evaluation
# ------------------------------
def craft_llm_prompt_io(traj_prompt: str, candidates: List[Dict[str, Any]], top_n: int = 60) -> Tuple[str, str]:
    """
    Creates system and user prompts for a compact IO-style ranking of candidate POIs.
    Forces the model to only select from the given list to reduce hallucination.

    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are ranking candidate POIs near the destination of a vehicle trip.\n"
        "Use time-of-day, spatial proximity via shared neighbors, and category consistency to reason.\n"
        "Only choose from the candidate list.\n\n"
        "Instructions:\n"
        "No reasoning steps needed.\n"
        "Return EXACTLY 20 names as a Python list of strings, in descending confidence.\n"
        "If you are uncertain or think fewer are highly plausible, STILL FILL to the full length by including the next-best candidates.\n"
        "Use ONLY names from the candidate list. Do NOT invent names. Do NOT include duplicates.\n"
        "Format example: [\"poi_name_1\", \"poi_name_2\", ..., \"poi_name_K\"]\n"
    )
    lines = []
    for i, r in enumerate(candidates[:top_n], 1):
        nb_names = [n for (n, _cats, _d) in r.get('nearby', [])][:5]
        lines.append(f"{i}) <{r['name']}> | ({r['major']}, {r['medium']}, {r['small']}) | {r.get('address','')} | neighbors: {nb_names}")
    user_prompt = (
        f"Trip description:\n{traj_prompt.strip()}\n\n"
        "Candidate POIs (name | categories | address | neighbors):\n"
        f"{chr(10).join(lines)}"
    )
    return system_prompt, user_prompt

def craft_llm_prompt(traj_prompt: str, candidates: List[Dict[str, Any]], top_n: int = 60) -> Tuple[str, str]:
    """
    Creates system and user prompts for the LLM to rank candidate POIs with Chain-of-Thought reasoning.
    Forces the model to only select from the given list to reduce hallucination.
    
    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert at ranking candidate POIs near the destination of a vehicle trip. "
        "Use time-of-day, spatial proximity via shared neighbors, and category consistency to reason. "
        "Return EXACTLY 20 names as a Python list of strings, in descending confidence.\n"
        "If you are uncertain or think fewer are highly plausible, STILL FILL to the full length by including the next-best candidates.\n"
        "Use ONLY names from the candidate list. Do NOT invent names. Do NOT include duplicates.\n"
        "Output format:\n"
        "Reasoning: [brief step by step reasoning]\n"
        "Ranking: [\"name1\", \"name2\", ...]"
    )
    
    lines = []
    for i, r in enumerate(candidates[:top_n], 1):
        nb_names = [n for (n, _cats, _d) in r.get('nearby', [])][:5]
        lines.append(f"{i}) <{r['name']}> | ({r['major']}, {r['medium']}, {r['small']}) | {r.get('address','')} | neighbors: {nb_names}")
    
    user_prompt = (
        f"Trip description:\n{traj_prompt.strip()}\n\n"
        "Candidate POIs (name | categories | address | neighbors):\n"
        f"{chr(10).join(lines)}\n\n"
        "Please rank these candidates and explain your reasoning."
    )
    
    return system_prompt, user_prompt


def craft_prompt_spatial_cot(traj_prompt: str, candidates: list[dict], top_n: int = 60) -> Tuple[str, str]:
    """
    Core-concepts–aware prompt for generative QA ranking with in-prompt examples.
    - Uses Xu et al.'s spatial core concepts + functional roles
    - Adds few-shot examples showing concept labeling + transformation path
    - Forces output as ONLY a Python list of POI names
    """

    header = (
        "You are ranking candidate POIs near the destination of a vehicle trip.\n"
        "Think in two stages: (A) interpret the trip using spatial core concepts and functional roles; "
        "(B) map them into concept transformations to judge which candidates are near the destination.\n\n"
        "Core Concepts:\n"
        "- Location: spatial reference used for geometry/extent.\n"
        "- Field: continuously varying/spatially homogeneous values (e.g., distance, land use).\n"
        "- Object: discrete bounded entities with identity/attributes (e.g., POIs, zones).\n"
        "- Event: time-bounded spatial occurrences (e.g., trips, visits).\n"
        "- Network: structured relations/flows between objects (e.g., roads, connectivity).\n"
        "- Amount: aggregated values of concepts (Content Amount = count/sum/avg; Coverage Amount = area/length/cluster size).\n"
        "- Proportion: ratio between amounts (e.g., density, rate).\n\n"
        "Functional Roles:\n"
        "- Measure: the goal you estimate (what the destination vicinity consists of).\n"
        "- Condition / Sub-condition: constraints via spatial relations/attributes (e.g., near/within/along, opening hours).\n"
        "- Support: spatial control/aggregation units if any.\n"
        "- Extent / Temporal Extent: spatial/temporal boundaries for the analysis.\n"
        "Use these roles to set an analysis order: Sub-condition → Condition → Support → Measure (Extent/TemporalExtent bound the process).\n"
        "Then define an abstract transformation path (DAG) over the concepts, e.g., Objects + Network/Field → proximity Field → selection of Objects.\n"
        "Leverage time-of-day, shared neighbors, network proximity, fields (distance), and category coherence.\n"
        "Critically: choose ONLY from the provided candidates; do not invent POIs.\n\n"
        "=== Examples ===\n"
        "Example 1:\n"
        "Trip: Tuesday 8:15 am, passed POIs include <city_park> (Object), <main_st_bus_stop> (Object, Network access)\n"
        "Candidates: <coffee_corner>, <24hr_gym>, <bookstore>\n"
        "Concepts:\n"
        "- trip: Event, temporal extent = morning\n"
        "- passed bus stop: Object, Network link to destination\n"
        "- coffee shop: Object, category match to morning trip context\n"
        "Transformation Path:\n"
        "1. [passed bus stop:Object] + [Network] → [distance Field to destination]\n"
        "2. Filter candidates by category/time-of-day relevance\n"
        "3. Rank by smallest network distance + temporal category match\n"
        "Output: [\"coffee_corner\", \"bookstore\", \"24hr_gym\"]\n\n"
        "Example 2:\n"
        "Trip: Friday 6:45 pm, passed POIs <office_building>, <subway_station>\n"
        "Candidates: <italian_restaurant>, <flower_shop>, <movie_theater>\n"
        "Concepts:\n"
        "- trip: Event, temporal extent = evening\n"
        "- restaurant/theater: Objects relevant to post-work leisure\n"
        "Transformation Path:\n"
        "1. Identify Objects near subway station (shared neighbor)\n"
        "2. Prioritize leisure categories matching evening timeframe\n"
        "Output: [\"italian_restaurant\", \"movie_theater\", \"flower_shop\"]\n"
        "=== End of Examples ===\n\n"
    )

    lines = []
    for i, r in enumerate(candidates[:top_n], 1):
        nb_names = [n for (n, _cats, _d) in r.get("nearby", [])][:5]
        lines.append(
            f"{i}) <{r['name']}> | ({r['major']}, {r['medium']}, {r['small']}) | "
            f"{r.get('address','')} | neighbors: {nb_names}"
        )

    tail = (
        "\nReasoning protocol:\n"
        "1) Label concepts in the trip: Objects (start/passed POIs), the trip as Event, relevant Networks/Fields, any Amount/Proportion.\n"
        "2) Assign roles: Sub-condition/Condition (spatial/temporal), Support (if any), Measure (destination vicinity composition).\n"
        "3) Build a transformation path: Objects + Network/Field → proximity/reachability Field → select Objects; "
        "justify via shared neighbors, geometric/network distance, time-of-day compatibility, and category coherence.\n"
        "4) Produce a ranked list with strict tie-breaking: prefer higher shared-neighbor support, then smaller distance, "
        "then stronger temporal/category fit, then denser local neighborhood.\n\n"
        "Output requirements (IMPORTANT):\n"
        "- Firstly return your reasoning process, especially how you use spatial core concepts and transformation path to do reasoning\n"
        "- Then, return EXACTLY 20 names as a Python list of strings, in descending confidence.\n"
        "- If you are uncertain or think fewer are highly plausible, STILL FILL to the full length by including the next-best candidates.\n"
        "- Use ONLY names from the candidate list. Do NOT invent names. Do NOT include duplicates.\n"
        "Format example: [\"poi_name_1\", \"poi_name_2\", ..., \"poi_name_K\"]\n"
    )
    system_prompt = header + tail
    user_prompt = (
        f"Trip description:\n{traj_prompt.strip()}\n\n"
        "Candidate POIs (name | categories | address | neighbors):\n"
        f"{chr(10).join(lines)}"
    )
    return system_prompt, user_prompt



def craft_prompt_spatial_cotp(traj_prompt: str, candidates: list[dict], top_n: int = 60) -> Tuple[str, str]:
    """
    Core-concepts–aware prompt for generative QA ranking with in-prompt examples.
    - Uses Xu et al.'s spatial core concepts + functional roles
    - Adds few-shot examples showing concept labeling + transformation path
    - Forces output as ONLY a Python list of POI names
    """

    header = (
        "You are ranking candidate POIs near the destination of a vehicle trip.\n"
        "Think in two stages: (A) interpret the trip using spatial core concepts and functional roles; "
        "(B) map them into concept transformations to judge which candidates are near the destination.\n\n"
        "Core Concepts:\n"
        "- Location: spatial reference used for geometry/extent.\n"
        "- Field: continuously varying/spatially homogeneous values (e.g., distance, land use).\n"
        "- Object: discrete bounded entities with identity/attributes (e.g., POIs, zones).\n"
        "- Event: time-bounded spatial occurrences (e.g., trips, visits).\n"
        "- Network: structured relations/flows between objects (e.g., roads, connectivity).\n"
        "- Amount: aggregated values of concepts (Content Amount = count/sum/avg; Coverage Amount = area/length/cluster size).\n"
        "- Proportion: ratio between amounts (e.g., density, rate).\n\n"
        "Functional Roles:\n"
        "- Measure: the goal you estimate (what the destination vicinity consists of).\n"
        "- Condition / Sub-condition: constraints via spatial relations/attributes (e.g., near/within/along, opening hours).\n"
        "- Support: spatial control/aggregation units if any.\n"
        "- Extent / Temporal Extent: spatial/temporal boundaries for the analysis.\n"
        "Use these roles to set an analysis order: Sub-condition → Condition → Support → Measure (Extent/TemporalExtent bound the process).\n"
        "Then define an abstract transformation path (DAG) over the concepts, e.g., Objects + Network/Field → proximity Field → selection of Objects.\n"
        "Leverage time-of-day, shared neighbors, network proximity, fields (distance), and category coherence.\n"
        "Critically: choose ONLY from the provided candidates; do not invent POIs.\n\n"
    )

    lines = []
    for i, r in enumerate(candidates[:top_n], 1):
        nb_names = [n for (n, _cats, _d) in r.get("nearby", [])][:5]
        lines.append(
            f"{i}) <{r['name']}> | ({r['major']}, {r['medium']}, {r['small']}) | "
            f"{r.get('address','')} | neighbors: {nb_names}"
        )
    user_prompt = (
    f"Trip description:\n{traj_prompt.strip()}\n\n"
    "Candidate POIs (name | categories | address | neighbors):\n"
    f"{chr(10).join(lines)}"
    )
    tail = (
        "\nReasoning protocol:\n"
        "1) Label concepts in the trip: Objects (start/passed POIs), the trip as Event, relevant Networks/Fields, any Amount/Proportion.\n"
        "2) Assign roles: Sub-condition/Condition (spatial/temporal), Support (if any), Measure (destination vicinity composition).\n"
        "3) Build a transformation path: Objects + Network/Field → proximity/reachability Field → select Objects; "
        "justify via shared neighbors, geometric/network distance, time-of-day compatibility, and category coherence.\n"
        "4) Produce a ranked list with strict tie-breaking: prefer higher shared-neighbor support, then smaller distance, "
        "then stronger temporal/category fit, then denser local neighborhood.\n\n"
        "Output requirements (IMPORTANT):\n"
        "- Firstly return your reasoning process, especially how you use spatial core concepts and transformation path to do reasoning\n"
        "- Then, return EXACTLY 20 names as a Python list of strings, in descending confidence.\n"
        "- If you are uncertain or think fewer are highly plausible, STILL FILL to the full length by including the next-best candidates.\n"
        "- Use ONLY names from the candidate list. Do NOT invent names. Do NOT include duplicates.\n"
        "Format example: [\"poi_name_1\", \"poi_name_2\", ..., \"poi_name_K\"]\n"
    )
    rag_results = rag_system.generate_transformation_path_iteratively(user_prompt, mode='concept_transformations_knowledge', max_steps=5)
    rag_info = (
        f" Here is the core concepts transformation path relevant to the user question, you can follow to do reasoning: :\n{rag_results}\n\n"
    )
    system_prompt = header + tail+rag_info
    return system_prompt, user_prompt



def parse_label(label_str: str) -> List[str]:
    names = [m.group(1) for m in POI_NAME_RE.finditer(label_str or "")]
    return [_canonical_name(n) for n in names]

def hr_at_k(pred: List[str], gold: List[str], k: int) -> float:
    topk = set(pred[:k])
    return 1.0 if any(g in topk for g in gold) else 0.0

def dcg_at_k(pred: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    dcg = 0.0
    for i, p in enumerate(pred[:k], 1):
        rel = 1.0 if p in gold_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(pred: List[str], gold: List[str], k: int) -> float:
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(pred, gold, k) / idcg

def evaluate_rank(pred_ranked_names: List[str], gold_names: List[str]) -> Dict[str, float]:
    pred_canonical = [_canonical_name(x) for x in pred_ranked_names]
    gold_canonical = [_canonical_name(x) for x in gold_names]
    metrics = {}
    for k in (5, 10, 20):
        metrics[f"HR@{k}"] = hr_at_k(pred_canonical, gold_canonical, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(pred_canonical, gold_canonical, k)
    return metrics

# ------------------------------
# End-to-end prep
# ------------------------------

def pipeline_prepare_candidates(
    poi_file: Path,
    traj_prompt: str,
    max_candidates: int = 200,
    max_hops: int = 3,
    use_name_keywords: bool = True,
    strategy: str = "spatial_cot",
    cache_file: Optional[Path] = None,
    prompt_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse POIs -> build graph & indexes -> recall candidates -> craft prompt
    
    Args:
        poi_file: Path to POI file
        traj_prompt: Trajectory prompt string
        max_candidates: Maximum number of candidates to recall
        max_hops: Maximum hops for graph expansion
        use_name_keywords: Whether to use name keywords in recall
        strategy: Prompting strategy to use
        cache_file: Optional path to cache file for storing/loading candidates
        prompt_id: Optional unique identifier for this prompt (e.g., row index)
    
    Returns:
    {
      "candidates": [record, ...],
      "llm_prompt": Dict[str, str],
      "candidate_names": [str, ...],  # canonical names in same order as candidates
      "pois_count": int
    }
    """
    import hashlib
    
    # Generate cache key from prompt and parameters
    cache_key = None
    cache_data = {}
    
    if cache_file is not None:
        # Use prompt_id if provided, otherwise hash the prompt
        if prompt_id is not None:
            cache_key = str(prompt_id)
        else:
            prompt_hash = hashlib.md5(traj_prompt.encode('utf-8')).hexdigest()
            cache_key = f"{prompt_hash}_{max_candidates}_{max_hops}_{use_name_keywords}"
        
        # Load existing cache if file exists
        if cache_file.exists():
            with cache_file.open('r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if candidates exist in cache
            if cache_key in cache_data:
                print(f"Loading candidates from cache for key: {cache_key}")
                cached_entry = cache_data[cache_key]
                candidates = cached_entry["candidates"]
                candidate_names = cached_entry["candidate_names"]
                pois_count = cached_entry.get("pois_count", len(candidates))
                
                print(f"Using strategy {strategy} with {len(candidates)} cached candidates")
                if strategy == "spatial_cot":
                    system_prompt, user_prompt = craft_prompt_spatial_cot(traj_prompt, candidates, top_n=min(60, len(candidates)))
                    llm_prompt = {"system": system_prompt, "user": user_prompt}
                elif strategy == "spatial_cotp":
                    system_prompt, user_prompt = craft_prompt_spatial_cotp(traj_prompt, candidates, top_n=min(60, len(candidates)))
                    llm_prompt = {"system": system_prompt, "user": user_prompt}
                elif strategy == "cot":
                    system_prompt, user_prompt = craft_llm_prompt(traj_prompt, candidates, top_n=min(60, len(candidates)))
                    llm_prompt = {"system": system_prompt, "user": user_prompt}
                elif strategy == "io":
                    system_prompt, user_prompt = craft_llm_prompt_io(traj_prompt, candidates, top_n=min(60, len(candidates)))
                    llm_prompt = {"system": system_prompt, "user": user_prompt}
                else:
                    raise ValueError(f"Unknown strategy {strategy}")
                
                return {
                    "candidates": candidates,
                    "llm_prompt": llm_prompt,
                    "candidate_names": candidate_names,
                    "pois_count": pois_count
                }
    
    # Not in cache or cache disabled - compute candidates
    print(f"Computing candidates for key: {cache_key if cache_key else 'no-cache'}")
    pois = parse_poi_file(poi_file)
    G = build_neighbor_graph(pois)
    name_idx, addr_idx = build_inverted_indexes(pois)
    cand_names = recall_candidates(
        traj_prompt, pois, G, name_idx, addr_idx,
        max_candidates=max_candidates, max_hops=max_hops,
        use_name_keywords=use_name_keywords
    )
    candidates = [pois[c] for c in cand_names if c in pois]

    # Light re-rank to push mentioned items up and well-connected nodes up
    def degree(cname: str) -> int:
        return len(G.get(cname, {}))
    mentioned = set(extract_prompt_poi_names(traj_prompt))
    candidates.sort(key=lambda r: (0 if _canonical_name(r["name"]) in mentioned else 1,
                                   -degree(_canonical_name(r["name"]))))
    
    candidate_names = [_canonical_name(r["name"]) for r in candidates]
    
    # Save to cache if enabled
    if cache_file is not None and cache_key is not None:
        cache_data[cache_key] = {
            "candidates": candidates,
            "candidate_names": candidate_names,
            "pois_count": len(pois),
            "prompt": traj_prompt,
            "params": {
                "max_candidates": max_candidates,
                "max_hops": max_hops,
                "use_name_keywords": use_name_keywords
            }
        }
        # Write cache atomically
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open('w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"Saved candidates to cache: {cache_file}")
    
    print(f"Using strategy {strategy} with {len(candidates)} candidates")
    if strategy == "spatial_cot":
        system_prompt, user_prompt = craft_prompt_spatial_cot(traj_prompt, candidates, top_n=min(60, len(candidates)))
        llm_prompt = {"system": system_prompt, "user": user_prompt}
    elif strategy == "spatial_cotp":
        system_prompt, user_prompt = craft_prompt_spatial_cotp(traj_prompt, candidates, top_n=min(60, len(candidates)))
        llm_prompt = {"system": system_prompt, "user": user_prompt}
    elif strategy == "cot":
        system_prompt, user_prompt = craft_llm_prompt(traj_prompt, candidates, top_n=min(60, len(candidates)))
        llm_prompt = {"system": system_prompt, "user": user_prompt}
    elif strategy == "io":
        system_prompt, user_prompt = craft_llm_prompt_io(traj_prompt, candidates, top_n=min(60, len(candidates)))
        llm_prompt = {"system": system_prompt, "user": user_prompt}
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return {"candidates": candidates,
            "llm_prompt": llm_prompt,
            "candidate_names": candidate_names,
            "pois_count": len(pois)}

# ------------------------------
# CLI
# ------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--poi", type=Path, required=True, help="Path to POI_ENG.txt (or POI_CN.txt)")
    ap.add_argument("--traj_csv", type=Path, help="CSV with columns: prompt[,label]")
    ap.add_argument("--prompt", type=str, help="Single trajectory prompt string")
    ap.add_argument("--row", type=int, default=None, help="Only process a specific CSV row index (0-based)")
    ap.add_argument("--out", type=Path, help="Output JSONL file with llm_prompt & candidate_names")
    ap.add_argument("--max-candidates", type=int, default=200)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--no-name-keywords", action="store_true", help="Disable name-keyword channel")
    ap.add_argument("--cache", type=Path, default=None, help="Path to candidate cache file (JSON)")
    ap.add_argument("--strategy", type=str, default="spatial_cot", help="Prompting strategy")
    args = ap.parse_args()

    if not args.traj_csv and not args.prompt:
        ap.error("Provide --traj_csv or --prompt")

    def process_one(idx: int, traj_prompt: str):
        prep = pipeline_prepare_candidates(
            args.poi, traj_prompt,
            max_candidates=args.max_candidates,
            max_hops=args.max_hops,
            use_name_keywords=(not args.no_name_keywords),
            strategy=args.strategy,
            cache_file=args.cache,
            prompt_id=idx if idx >= 0 else None
        )
        rec = {"row": idx, "llm_prompt": prep["llm_prompt"], "candidate_names": prep["candidate_names"]}
        return rec

    out_f = None
    if args.out:
        out_f = args.out.open("w", encoding="utf-8")

    if args.prompt:
        rec = process_one(-1, args.prompt)
        if out_f:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        with args.traj_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if args.row is not None:
            r = rows[args.row]
            rec = process_one(args.row, r["prompt"])
            if out_f:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                print(json.dumps(rec, ensure_ascii=False, indent=2))
        else:
            for i, r in enumerate(rows):
                rec = process_one(i, r["prompt"])
                if out_f:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    print(json.dumps(rec, ensure_ascii=False))

    if out_f:
        out_f.close()

if __name__ == "__main__":
    _cli()
