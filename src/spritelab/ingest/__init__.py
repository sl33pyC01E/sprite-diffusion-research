"""Corpus-to-index ingestion routines."""

from spritelab.ingest.freedoom import ingest_freedoom_sequences
from spritelab.ingest.spritecook import ingest_spritecook_sequences

__all__ = ["ingest_freedoom_sequences", "ingest_spritecook_sequences"]
