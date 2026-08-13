from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from spritelab.db import IndexDB
from spritelab.fetch import FetchResult, HttpFetcher, fetch_indexed
from spritelab.sources import SourceDefinition
from spritelab.storage import DiskFloorReached, DiskGuard


@dataclass(frozen=True)
class GitHubSnapshot:
    source_id: str
    repository: str
    commit_sha: str
    item_id: str
    run_id: str
    metadata_blob_sha256: str
    commit_blob_sha256: str
    license_blob_sha256: str | None
    archive_blob_sha256: str | None
    archive_path: Path | None


class GitHubRepositoryAdapter:
    """Acquire a commit-pinned repository snapshot and its evidence payloads."""

    def __init__(
        self,
        *,
        database: IndexDB,
        fetcher: HttpFetcher,
        guard: DiskGuard,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.database = database
        self.fetcher = fetcher
        self.guard = guard
        self.sleep = sleep

    def acquire(
        self,
        source: SourceDefinition,
        *,
        download_archive: bool = True,
        ref: str | None = None,
    ) -> GitHubSnapshot:
        owner, repository_name = _github_coordinates(source.root_url)
        repository = f"{owner}/{repository_name}"
        self.database.initialize()
        item_id = self.database.upsert_item(
            source_id=source.id,
            external_id=repository,
            canonical_url=source.root_url,
            title=repository_name,
            creator_name=owner,
            creator_url=f"https://github.com/{owner}",
            metadata={"acquisition_state": "started"},
        )
        status = self.guard.status()
        run_id = self.database.create_crawl_run(
            source_id=source.id,
            parameters={"repository": repository, "ref": ref, "archive": download_archive},
            free_bytes_start=status.free_bytes,
        )
        interval = 0.0
        if source.default_requests_per_second > 0:
            interval = 1.0 / source.default_requests_per_second

        try:
            metadata_url = f"https://api.github.com/repos/{owner}/{repository_name}"
            _retrieval_id, metadata_fetch = fetch_indexed(
                fetcher=self.fetcher,
                database=self.database,
                url=metadata_url,
                run_id=run_id,
                item_id=item_id,
                role="repository_metadata",
                original_filename="repository.json",
            )
            metadata = _read_json(metadata_fetch)
            resolved_ref = ref or str(metadata["default_branch"])

            self._pace(interval)
            commit_url = (
                f"https://api.github.com/repos/{owner}/{repository_name}/commits/"
                f"{quote(resolved_ref, safe='')}"
            )
            _retrieval_id, commit_fetch = fetch_indexed(
                fetcher=self.fetcher,
                database=self.database,
                url=commit_url,
                run_id=run_id,
                item_id=item_id,
                role="commit_metadata",
                original_filename="commit.json",
            )
            commit = _read_json(commit_fetch)
            commit_sha = str(commit["sha"])

            license_fetch = self._acquire_license(
                owner=owner,
                repository_name=repository_name,
                commit_sha=commit_sha,
                item_id=item_id,
                run_id=run_id,
                interval=interval,
            )

            archive_fetch: FetchResult | None = None
            if download_archive:
                self._pace(interval)
                archive_url = (
                    f"https://codeload.github.com/{owner}/{repository_name}/zip/{commit_sha}"
                )
                _retrieval_id, archive_fetch = fetch_indexed(
                    fetcher=self.fetcher,
                    database=self.database,
                    url=archive_url,
                    run_id=run_id,
                    item_id=item_id,
                    role="source_archive",
                    original_filename=f"{repository_name}-{commit_sha}.zip",
                )

            self.database.upsert_item(
                source_id=source.id,
                external_id=repository,
                canonical_url=str(metadata.get("html_url", source.root_url)),
                title=str(metadata.get("name") or repository_name),
                description=metadata.get("description"),
                creator_name=str(metadata.get("owner", {}).get("login") or owner),
                creator_url=metadata.get("owner", {}).get("html_url"),
                published_at=metadata.get("created_at"),
                metadata={
                    "full_name": metadata.get("full_name"),
                    "default_branch": metadata.get("default_branch"),
                    "resolved_ref": resolved_ref,
                    "commit_sha": commit_sha,
                    "repository_size_kib": metadata.get("size"),
                    "archived": metadata.get("archived"),
                    "fork": metadata.get("fork"),
                    "topics": metadata.get("topics", []),
                    "updated_at": metadata.get("updated_at"),
                    "pushed_at": metadata.get("pushed_at"),
                },
            )
            self.database.finish_crawl_run(
                run_id,
                status="completed",
                free_bytes_end=self.guard.status().free_bytes,
            )
            return GitHubSnapshot(
                source_id=source.id,
                repository=repository,
                commit_sha=commit_sha,
                item_id=item_id,
                run_id=run_id,
                metadata_blob_sha256=metadata_fetch.blob.sha256,
                commit_blob_sha256=commit_fetch.blob.sha256,
                license_blob_sha256=(license_fetch.blob.sha256 if license_fetch else None),
                archive_blob_sha256=(archive_fetch.blob.sha256 if archive_fetch else None),
                archive_path=(archive_fetch.blob.path if archive_fetch else None),
            )
        except DiskFloorReached as error:
            self.database.finish_crawl_run(
                run_id,
                status="disk_floor",
                free_bytes_end=self.guard.status().free_bytes,
                error=str(error),
            )
            raise
        except BaseException as error:
            self.database.finish_crawl_run(
                run_id,
                status="failed",
                free_bytes_end=self.guard.status().free_bytes,
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def _acquire_license(
        self,
        *,
        owner: str,
        repository_name: str,
        commit_sha: str,
        item_id: str,
        run_id: str,
        interval: float,
    ) -> FetchResult | None:
        self._pace(interval)
        url = (
            f"https://api.github.com/repos/{owner}/{repository_name}/license"
            f"?ref={quote(commit_sha, safe='')}"
        )
        try:
            _retrieval_id, result = fetch_indexed(
                fetcher=self.fetcher,
                database=self.database,
                url=url,
                run_id=run_id,
                item_id=item_id,
                role="license_metadata",
                original_filename="license.json",
            )
        except Exception as error:
            if getattr(error, "status_code", None) == 404:
                self.database.add_rights_observation(
                    item_id=item_id,
                    license_raw=None,
                    license_expression=None,
                    license_url=None,
                    terms_url=None,
                    basis="github_license_api_no_root_license",
                    metadata={"commit_sha": commit_sha, "api_url": url},
                )
                return None
            raise
        payload = _read_json(result)
        license_data = payload.get("license") or {}
        encoded_content = payload.get("content")
        decoded_content: str | None = None
        if isinstance(encoded_content, str) and payload.get("encoding") == "base64":
            decoded_content = base64.b64decode(encoded_content).decode("utf-8", "replace")
        self.database.add_rights_observation(
            item_id=item_id,
            license_raw=decoded_content or license_data.get("name"),
            license_expression=license_data.get("spdx_id"),
            license_url=payload.get("html_url"),
            terms_url=None,
            terms_blob_sha256=result.blob.sha256,
            basis="github_root_license_at_commit",
            metadata={
                "commit_sha": commit_sha,
                "license_key": license_data.get("key"),
                "license_name": license_data.get("name"),
                "api_url": url,
            },
        )
        return result

    def _pace(self, interval: float) -> None:
        if interval > 0:
            self.sleep(interval)


def _github_coordinates(root_url: str) -> tuple[str, str]:
    parsed = urlparse(root_url)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        raise ValueError(f"Not a GitHub repository URL: {root_url}")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"GitHub repository URL lacks owner/repository: {root_url}")
    return parts[0], parts[1].removesuffix(".git")


def _read_json(result: FetchResult) -> dict[str, Any]:
    payload = json.loads(result.blob.path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {result.requested_url}")
    return payload
