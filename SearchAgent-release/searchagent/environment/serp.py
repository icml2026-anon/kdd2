"""Search Engine Results Page (SERP) generation and representation.

Supports two modes:
1. Template-based generation (SERPGenerator) — synthetic SERPs from templates
2. TREC replay (TRECSERPLoader) — real SERPs from TREC Session Track 2014
"""

from __future__ import annotations

import logging
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from searchagent.config import EnvironmentConfig
from searchagent.environment.serp_templates import (
    AD_TEMPLATES,
    ORGANIC_TEMPLATES,
    ResultTemplate,
)

logger = logging.getLogger(__name__)

# URL patterns heuristically classified as ad/commercial
_AD_URL_PATTERNS = [
    "amazon.com", "ebay.com", "walmart.com", "bestbuy.com", "target.com",
    "shop", "store", "buy", "deal", "offer", "coupon",
    "aliexpress", "etsy.com", "overstock.com",
    "ads.", "sponsor", "affiliate", "pricegrabber", "shopzilla",
]


@dataclass
class SearchResult:
    """A single result on a SERP page."""

    rank: int
    title: str
    snippet: str
    url: str
    is_ad: bool = False
    relevance: float = 0.5

    def to_display(self, hide_ad_labels: bool = False) -> str:
        tag = " Sponsored" if (self.is_ad and not hide_ad_labels) else ""
        return f"  {self.rank}. {self.title}{tag}\n     {self.snippet}\n     {self.url}"


@dataclass
class SERPPage:
    """A complete search engine results page."""

    query: str
    page_number: int
    results: List[SearchResult] = field(default_factory=list)

    def to_display(self, hide_ad_labels: bool = False) -> str:
        lines = [f'Results for "{self.query}" (page {self.page_number}):']
        for r in self.results:
            lines.append(r.to_display(hide_ad_labels=hide_ad_labels))
        return "\n".join(lines)

    @property
    def organic_results(self) -> List[SearchResult]:
        return [r for r in self.results if not r.is_ad]

    @property
    def ad_results(self) -> List[SearchResult]:
        return [r for r in self.results if r.is_ad]


def _best_matching_key(query: str, template_dict: Dict[str, list]) -> str:
    """Find the template key most similar to the query via fuzzy matching.

    Falls back to 'default' if no good match found.
    """
    query_lower = query.lower()
    if query_lower in template_dict:
        return query_lower

    best_key = "default"
    best_score = 0.0
    for key in template_dict:
        if key == "default":
            continue
        score = SequenceMatcher(None, query_lower, key).ratio()
        # Also check if key words appear in query
        key_words = set(key.split())
        query_words = set(query_lower.split())
        overlap = len(key_words & query_words) / max(len(key_words), 1)
        combined = 0.4 * score + 0.6 * overlap
        if combined > best_score:
            best_score = combined
            best_key = key

    # Require minimum similarity to avoid irrelevant matches
    if best_score < 0.15:
        return "default"
    return best_key


class SERPGenerator:
    """Generates realistic SERP pages for search queries.

    Features:
    - Fuzzy matching to find the best template for reformulated queries
    - Page-aware result shuffling for subsequent pages
    - Controlled relevance decay across ranks and pages
    - Separate ad placement with configurable positions
    """

    def __init__(self, config: EnvironmentConfig, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)

    def generate(self, query: str, page_number: int = 1) -> SERPPage:
        """Generate a SERP page for a given query."""
        results = []
        rank = 1

        # Fuzzy match to find the best template pool for this query
        organic_key = _best_matching_key(query, ORGANIC_TEMPLATES)
        ad_key = _best_matching_key(query, AD_TEMPLATES)
        organic_pool = list(ORGANIC_TEMPLATES[organic_key])
        ad_pool = list(AD_TEMPLATES[ad_key])

        slug = query.lower().replace(" ", "-")[:30]

        # Shuffle templates for page > 1 to get different results per page
        if page_number > 1:
            page_rng = random.Random(hash(query) + page_number)
            page_rng.shuffle(organic_pool)
            page_rng.shuffle(ad_pool)
            # Reduce number of ads on later pages
            max_ads_this_page = max(1, self.config.max_ads_per_page - (page_number - 1))
        else:
            max_ads_this_page = self.config.max_ads_per_page

        ad_positions = set(self.config.ad_positions[:max_ads_this_page])

        # Build result list with ads at specified positions
        organic_idx = 0
        ad_idx = 0
        total_slots = self.config.max_results_per_page + min(
            len(ad_positions), max_ads_this_page
        )

        for slot in range(total_slots):
            if slot in ad_positions and ad_idx < len(ad_pool) and ad_idx < max_ads_this_page:
                title, snippet, url = ad_pool[ad_idx % len(ad_pool)]
                title = title.format(query=query, slug=slug)
                snippet = snippet.format(query=query, slug=slug)
                url = url.format(query=query, slug=slug)
                # Ad relevance: partially random, partially dependent on query match
                ad_rel = self.rng.uniform(0.2, 0.6)
                if organic_key != "default":
                    ad_rel += 0.15  # known query topic → more relevant ads
                results.append(SearchResult(
                    rank=rank, title=title, snippet=snippet,
                    url=url, is_ad=True,
                    relevance=min(ad_rel, 0.85),
                ))
                ad_idx += 1
            else:
                if organic_idx < len(organic_pool):
                    title, snippet, url = organic_pool[organic_idx % len(organic_pool)]
                    title = title.format(query=query, slug=slug)
                    snippet = snippet.format(query=query, slug=slug)
                    url = url.format(query=query, slug=slug)
                    # Relevance decreases with rank and page number
                    base_rel = max(0.1, 1.0 - organic_idx * 0.07 - (page_number - 1) * 0.2)
                    noise = self.rng.uniform(-0.08, 0.08)
                    results.append(SearchResult(
                        rank=rank, title=title, snippet=snippet,
                        url=url, is_ad=False,
                        relevance=max(0.05, min(base_rel + noise, 0.98)),
                    ))
                    organic_idx += 1

            rank += 1
            if len(results) >= self.config.max_results_per_page + max_ads_this_page:
                break

        return SERPPage(query=query, page_number=page_number, results=results)


# ── TREC Session Track 2014 replay support ─────────────────────────────

@dataclass
class TRECClick:
    """A real click from TREC session data."""
    rank: int
    start_time: float
    end_time: float

    @property
    def dwell_time(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TRECInteraction:
    """One query→SERP→clicks interaction from a real TREC session."""
    interaction_num: int
    query: str
    interaction_type: str
    start_time: float
    results: List[SearchResult] = field(default_factory=list)
    clicks: List[TRECClick] = field(default_factory=list)

    def to_serp(self) -> SERPPage:
        return SERPPage(query=self.query, page_number=1, results=list(self.results))


@dataclass
class TRECSession:
    """A complete real search session from TREC Session Track 2014."""
    session_num: int
    user_id: int
    topic_num: int
    topic_desc: str
    interactions: List[TRECInteraction] = field(default_factory=list)
    current_query: str = ""

    @property
    def num_queries(self) -> int:
        return len(self.interactions)

    @property
    def total_clicks(self) -> int:
        return sum(len(i.clicks) for i in self.interactions)

    @property
    def click_positions(self) -> List[int]:
        return [c.rank for i in self.interactions for c in i.clicks]

    @property
    def has_clicks(self) -> bool:
        return self.total_clicks > 0

    @property
    def all_queries(self) -> List[str]:
        return [i.query for i in self.interactions if i.query]

    @property
    def total_dwell_time(self) -> float:
        return sum(c.dwell_time for i in self.interactions for c in i.clicks)


def _is_ad_url(url: str) -> bool:
    url_lower = url.lower()
    return any(p in url_lower for p in _AD_URL_PATTERNS)


class TRECSERPLoader:
    """Parse TREC Session Track 2014 XML and provide real sessions/SERPs.

    Usage:
        loader = TRECSERPLoader("data/search_logs/trec_session2014.xml")
        for session in loader.sessions:
            serp = session.interactions[0].to_serp()
    """

    def __init__(self, xml_path: str):
        self.sessions: List[TRECSession] = []
        self._parse(xml_path)

    @staticmethod
    def _safe_float(val: str, default: float = 0.0) -> float:
        """Parse float, handling 'NA', timestamps like '09:54:08', etc."""
        if not val or val.upper() == 'NA':
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def _parse(self, xml_path: str) -> None:
        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"TREC XML not found: {xml_path}")

        # Sanitize XML: fix unescaped & (common in TREC 2012)
        raw = path.read_text(encoding='utf-8', errors='replace')
        import re
        raw = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)', '&amp;', raw)

        root = ET.fromstring(raw)

        for s_xml in root.findall('.//session'):
            try:
                snum = int(s_xml.attrib.get('num', 0))
            except ValueError:
                snum = 0
            try:
                uid = int(s_xml.attrib.get('userid', 0))
            except ValueError:
                uid = -1

            topic_xml = s_xml.find('topic')
            tnum = int(topic_xml.attrib.get('num', 0)) if topic_xml is not None else 0
            desc_xml = topic_xml.find('desc') if topic_xml is not None else None
            tdesc = desc_xml.text.strip() if desc_xml is not None and desc_xml.text else ""

            session = TRECSession(session_num=snum, user_id=uid,
                                  topic_num=tnum, topic_desc=tdesc)

            for ix in s_xml.findall('interaction'):
                inum = int(ix.attrib.get('num', 0))
                itype = ix.attrib.get('type', '')
                istart = self._safe_float(ix.attrib.get('starttime', '0'))
                q_xml = ix.find('query')
                query = q_xml.text.strip() if q_xml is not None and q_xml.text else ""

                interaction = TRECInteraction(
                    interaction_num=inum, query=query,
                    interaction_type=itype, start_time=istart,
                )

                res_xml = ix.find('results')
                if res_xml is not None:
                    for r_xml in res_xml.findall('result'):
                        rank = int(r_xml.attrib.get('rank', 0))
                        url_x = r_xml.find('url')
                        tit_x = r_xml.find('title')
                        snip_x = r_xml.find('snippet')
                        url = url_x.text.strip() if url_x is not None and url_x.text else ""
                        title = tit_x.text.strip() if tit_x is not None and tit_x.text else ""
                        snippet = snip_x.text.strip() if snip_x is not None and snip_x.text else ""
                        interaction.results.append(SearchResult(
                            rank=rank, title=title, snippet=snippet,
                            url=url, is_ad=_is_ad_url(url), relevance=0.5,
                        ))
                    interaction.results.sort(key=lambda r: r.rank)

                cl_xml = ix.find('clicked')
                if cl_xml is not None:
                    for c_xml in cl_xml.findall('click'):
                        interaction.clicks.append(TRECClick(
                            rank=int(c_xml.attrib.get('num', 0)),
                            start_time=self._safe_float(c_xml.attrib.get('starttime', '0')),
                            end_time=self._safe_float(c_xml.attrib.get('endtime', '0')),
                        ))

                session.interactions.append(interaction)

            cq_xml = s_xml.find('currentquery')
            if cq_xml is not None:
                q_xml = cq_xml.find('query')
                session.current_query = q_xml.text.strip() if q_xml is not None and q_xml.text else ""

            if session.interactions and any(i.query for i in session.interactions):
                self.sessions.append(session)

        logger.info("Loaded %d TREC sessions (%d users, %d topics)",
                     len(self.sessions),
                     len(set(s.user_id for s in self.sessions)),
                     len(set(s.topic_num for s in self.sessions)))

    def get_sessions_by_user(self) -> Dict[int, List[TRECSession]]:
        from collections import defaultdict
        by_user: Dict[int, List[TRECSession]] = defaultdict(list)
        for s in self.sessions:
            by_user[s.user_id].append(s)
        return dict(by_user)

    def get_user_profiles(self) -> Dict[int, Dict[str, Any]]:
        """Extract behavioral profiles from real user data (like Agent4Rec)."""
        by_user = self.get_sessions_by_user()
        profiles = {}
        for uid, u_sessions in by_user.items():
            total_inter = sum(s.num_queries for s in u_sessions)
            total_clicks = sum(s.total_clicks for s in u_sessions)
            total_dwell = sum(s.total_dwell_time for s in u_sessions)
            reform_sessions = sum(1 for s in u_sessions if s.num_queries > 1)
            topics = len(set(s.topic_num for s in u_sessions))

            profiles[uid] = {
                "user_id": uid,
                "num_sessions": len(u_sessions),
                "activity": total_inter,
                "click_rate": round(total_clicks / max(total_inter, 1), 3),
                "reformulation_rate": round(reform_sessions / max(len(u_sessions), 1), 3),
                "avg_dwell_time": round(total_dwell / max(total_clicks, 1), 1),
                "topic_diversity": topics,
            }
        return profiles


class YandexSERPLoader:
    """Parse Yandex Personalized Web Search Challenge TSV.gz click log format.

    Reuses TRECSession, TRECInteraction, and SearchResult structures to achieve
    instant 100% interoperability with all existing downstream evaluation pipelines.
    """

    def __init__(self, gz_path: str, max_sessions: int = 20):
        self.sessions: List[TRECSession] = []
        self._parse(gz_path, max_sessions)

    def _parse(self, gz_path: str, max_sessions: int) -> None:
        import gzip
        path = Path(gz_path)
        if not path.exists():
            raise FileNotFoundError(f"Yandex dataset not found: {gz_path}")

        session_dict = {}

        # Read line by line from compressed gzip
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts:
                    continue

                sid = int(parts[0])
                if parts[1] == 'M':
                    record_type = 'M'
                else:
                    record_type = parts[2]

                if record_type == 'M':
                    # Move completed sessions with clicks to self.sessions
                    for s_id, s in list(session_dict.items()):
                        if s_id != sid:
                            if s.interactions and any(i.clicks for i in s.interactions):
                                self.sessions.append(s)
                            del session_dict[s_id]

                    if len(self.sessions) >= max_sessions:
                        break

                    uid = int(parts[2])
                    day = int(parts[3])
                    session_dict[sid] = TRECSession(
                        session_num=sid,
                        user_id=uid,
                        topic_num=day,
                        topic_desc=f"Yandex User Session Search Day {day}",
                        interactions=[],
                        current_query=""
                    )
                
                elif record_type in ('Q', 'T'):
                    if sid not in session_dict:
                        continue
                    time_passed = float(parts[1])  # in Yandex, the 2nd field is TimePassed for non-M rows
                    qtype = record_type
                    qidx = int(parts[3])
                    qid = int(parts[4])
                    # parts[5] contains terms, parts[6:] contains URL,domain pairs
                    results = []
                    rank = 1
                    for pair in parts[6:]:
                        if ',' in pair:
                            url_id, domain_id = pair.split(',')
                            # Classify as ad heuristically using domain_id % 7 to test commercial ad actions
                            is_ad = (int(domain_id) % 7 == 0)
                            results.append(SearchResult(
                                rank=rank,
                                title=f"Yandex Document {url_id}",
                                snippet=f"Search result on domain {domain_id} for query {qid}",
                                url=f"http://yandex.ru/url/{url_id}",
                                is_ad=is_ad,
                                relevance=0.5
                            ))
                            rank += 1

                    interaction = TRECInteraction(
                        interaction_num=qidx,
                        query=f"query_{qid}",
                        interaction_type=qtype,
                        start_time=time_passed,
                        results=results,
                        clicks=[]
                    )
                    session_dict[sid].interactions.append(interaction)
                    session_dict[sid].current_query = f"query_{qid}"

                elif record_type == 'C':
                    if sid not in session_dict:
                        continue
                    time_passed = float(parts[1])
                    qidx = int(parts[3])
                    url_id = parts[4]

                    # Find interaction with corresponding index
                    session = session_dict[sid]
                    target_interaction = None
                    for inter in session.interactions:
                        if inter.interaction_num == qidx:
                            target_interaction = inter
                            break
                    
                    if target_interaction:
                        # Find rank of clicked URL
                        target_url = f"http://yandex.ru/url/{url_id}"
                        click_rank = None
                        for r in target_interaction.results:
                            if r.url == target_url:
                                click_rank = r.rank
                                break
                        
                        if click_rank is not None:
                            target_interaction.clicks.append(TRECClick(
                                rank=click_rank,
                                start_time=time_passed,
                                end_time=time_passed + 30.0
                            ))

            # Move any remaining active sessions in session_dict that have clicks
            for s_id, s in list(session_dict.items()):
                if len(self.sessions) >= max_sessions:
                    break
                if s.interactions and any(i.clicks for i in s.interactions):
                    self.sessions.append(s)

        logger.info("Loaded %d Yandex sessions (%d users, %d days)",
                    len(self.sessions),
                    len(set(s.user_id for s in self.sessions)),
                    len(set(s.topic_num for s in self.sessions)))


class OrcasSERPLoader:
    """Parse Microsoft ORCAS TSV.gz click log format.

    Reuses TRECSession, TRECInteraction, and SearchResult structures to achieve
    instant 100% interoperability with all existing downstream evaluation pipelines.
    """

    def __init__(self, gz_path: str, max_sessions: int = 20):
        self.sessions: List[TRECSession] = []
        self._parse(gz_path, max_sessions)

    def _parse(self, gz_path: str, max_sessions: int) -> None:
        import gzip
        path = Path(gz_path)
        if not path.exists():
            raise FileNotFoundError(f"ORCAS dataset not found: {gz_path}")

        # Read line by line from compressed gzip and group by qid
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
            current_qid = None
            current_query = None
            current_clicks = []
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                qid, query, did, url = parts[0], parts[1], parts[2], parts[3]
                
                if qid != current_qid:
                    if current_qid is not None and current_clicks:
                        self._add_session(current_qid, current_query, current_clicks)
                        if len(self.sessions) >= max_sessions:
                            break
                    current_qid = qid
                    current_query = query
                    current_clicks = []
                
                if (did, url) not in current_clicks:
                    current_clicks.append((did, url))
            
            # Handle the last group
            if len(self.sessions) < max_sessions and current_qid is not None and current_clicks:
                self._add_session(current_qid, current_query, current_clicks)

        logger.info("Loaded %d ORCAS sessions", len(self.sessions))

    def _add_session(self, qid: str, query: str, clicks: List[Tuple[str, str]]) -> None:
        results = []
        rank = 1
        
        # Add real clicked documents at top ranks
        for did, url in clicks[:10]:
            results.append(SearchResult(
                rank=rank,
                title=f"ORCAS Doc {did} for {query}",
                snippet=f"Highly relevant search result matching query '{query}' pointing to {url}",
                url=url,
                is_ad=False,
                relevance=0.85
            ))
            rank += 1
            
        # Pad with generic results up to 10
        slug = query.lower().replace(" ", "-")[:30]
        while len(results) < 10:
            results.append(SearchResult(
                rank=rank,
                title=f"Synthetic Web Page for {query} - Result {rank}",
                snippet=f"This is a generic search result page for query '{query}' designed to provide content padding at rank {rank}.",
                url=f"http://example.com/search/{slug}-{rank}",
                is_ad=False,
                relevance=0.2
            ))
            rank += 1
            
        # Build TRECClick records
        trec_clicks = []
        for i in range(len(clicks[:10])):
            trec_clicks.append(TRECClick(
                rank=i + 1,
                start_time=10.0 + i * 20.0,
                end_time=10.0 + i * 20.0 + 15.0
            ))
            
        interaction = TRECInteraction(
            interaction_num=1,
            query=query,
            interaction_type='Q',
            start_time=0.0,
            results=results,
            clicks=trec_clicks
        )
        
        session = TRECSession(
            session_num=int(qid) if qid.isdigit() else hash(qid),
            user_id=1,
            topic_num=1,
            topic_desc=f"ORCAS Query Session for task: {query}",
            interactions=[interaction],
            current_query=query
        )
        self.sessions.append(session)

