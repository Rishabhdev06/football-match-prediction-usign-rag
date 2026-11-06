"""
Semantic & Metadata-Filtered Retrieval Engine for Football RAG
==============================================================
Supports semantic similarity queries, chronological date cutoffs,
team-specific filters, and formatted source attribution.
"""

from datetime import datetime
from rag.vector_store import query_collection, get_collection


class RetrievedPassage:
    def __init__(self, record_id, text, metadata, similarity=0.0):
        self.record_id = record_id
        self.text = text
        self.metadata = metadata or {}
        self.similarity = similarity
        self.doc_type = self.metadata.get('doc_type', 'reference')
        self.source = self.metadata.get('source', 'Unknown Source')
        self.date = self.metadata.get('date', None)
        self.home_team = self.metadata.get('home_team', None)
        self.away_team = self.metadata.get('away_team', None)
        self.referee = self.metadata.get('referee', None)

    @property
    def citation_label(self):
        """Returns human-readable citation label for UI and LLM attribution."""
        if self.doc_type == 'match_record':
            return f'Match: {self.home_team} vs {self.away_team} ({self.date}) [{self.record_id}]'
        elif self.doc_type == 'team_summary':
            team = self.metadata.get('team', 'Team')
            season = self.metadata.get('season', 'EPL')
            return f'Summary: {team} ({season} Season) [{self.record_id}]'
        else:
            title = self.metadata.get('title', self.source)
            return f'Doc: {self.source} - {title} [{self.record_id}]'

    def to_dict(self):
        return {
            'record_id': self.record_id,
            'text': self.text,
            'metadata': self.metadata,
            'similarity': self.similarity,
            'doc_type': self.doc_type,
            'source': self.source,
            'date': self.date,
            'citation': self.citation_label
        }


def retrieve_evidence(query, top_k=5, date_cutoff=None, doc_type=None, team_filter=None):
    """
    Searches vector store and applies post-filtering for dates and teams
    to guarantee temporal accuracy (preventing future-data leakage).
    """
    # Request slightly larger candidate pool to accommodate post-filtering
    fetch_k = max(top_k * 3, 15)
    where_filter = None
    if doc_type:
        where_filter = {'doc_type': doc_type}

    raw_results = query_collection(query_text=query, top_k=fetch_k, where_filter=where_filter)

    passages = []
    for item in raw_results:
        meta = item['metadata']
        item_date = meta.get('date')

        # Chronological cutoff enforcement: exclude matches later than cutoff
        if date_cutoff and item_date:
            try:
                if str(item_date) > str(date_cutoff):
                    continue
            except (TypeError, ValueError):
                pass

        # Optional team filter
        if team_filter:
            h_team = meta.get('home_team')
            a_team = meta.get('away_team')
            team_match = False
            if isinstance(team_filter, list):
                if h_team in team_filter or a_team in team_filter or meta.get('team') in team_filter:
                    team_match = True
            elif isinstance(team_filter, str):
                if h_team == team_filter or a_team == team_filter or meta.get('team') == team_filter:
                    team_match = True
            if not team_match:
                continue

        p = RetrievedPassage(
            record_id=item['id'],
            text=item['text'],
            metadata=meta,
            similarity=item.get('similarity', 0.0)
        )
        passages.append(p)
        if len(passages) >= top_k:
            break

    return passages


def retrieve_head_to_head_history(team1, team2, date_cutoff=None, limit=5):
    """
    Retrieves head-to-head match records between two teams strictly before date_cutoff.
    """
    query = f'Match between {team1} and {team2} head to head result score'
    candidates = retrieve_evidence(
        query=query,
        top_k=limit * 4,
        date_cutoff=date_cutoff,
        doc_type='match_record'
    )
    # Strictly filter where both teams played against each other
    h2h = []
    for p in candidates:
        ht = p.home_team
        at = p.away_team
        if (ht == team1 and at == team2) or (ht == team2 and at == team1):
            h2h.append(p)
            if len(h2h) >= limit:
                break
    return h2h


def retrieve_recent_matches(team_name, date_cutoff=None, limit=5):
    """
    Retrieves a team's most recent verified match records strictly prior to date_cutoff.
    """
    query = f'{team_name} recent match performance and result'
    candidates = retrieve_evidence(
        query=query,
        top_k=limit * 4,
        date_cutoff=date_cutoff,
        doc_type='match_record',
        team_filter=team_name
    )
    # Sort strictly descending by date to get chronological last N games
    dated = [p for p in candidates if p.date is not None]
    dated.sort(key=lambda x: x.date, reverse=True)
    return dated[:limit]


def format_evidence_for_prompt(passages):
    """
    Formats retrieved passages into a structured evidence block for LLM prompt context.
    """
    if not passages:
        return 'No historical match evidence retrieved for this query.'

    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f'[{i}] SOURCE: {p.citation_label}')
        lines.append(f'    PASSAGE: {p.text}')
    return "\n".join(lines)
