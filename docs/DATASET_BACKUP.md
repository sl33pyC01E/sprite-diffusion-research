# Dataset backup

The research dataset is backed up separately from GitHub because it is large and
contains third-party source material under mixed or unknown terms. The canonical
Drive destination for this snapshot is:

- folder: `Sprite Diffusion Research Dataset Backup — 2026-08-15`
- Drive folder ID: `1ov3R_eXN8-BnJT88_rwBRWsepl-Oa7oq`
- selection manifest: `releases/dataset-drive-selection-v1.json`

The snapshot is reconstruction-oriented. It contains the complete acquired Anime
Ascension RAR, immutable content-addressed source objects, the four canonical RGBA
six-action corpora used for broad MUGEN training, caption data, and the complete
provenance index/reports/snapshots. The selected source content is 112,572,985,487
bytes before archive-container overhead.

It deliberately excludes:

- the incomplete MUGEN X Alpha partial download;
- downloaded foundation models;
- experiment checkpoints, which are published through the GitHub weight release;
- inference output and temporary release staging;
- duplicate or superseded materializations;
- VAE/CLIP/latent caches that can be regenerated from the canonical RGBA records,
  manifests, and published project weights.

The Anime Ascension source file is a verified compact representation of an original
RAR with a zero-filled tail. Its compact SHA-256 is
`0a16a93be8971843ea1822cffd95942364e2b9f6ce05a1dd921ce490f1a71294`.
The original logical file can be reconstructed by appending exactly 27,158,350,425
zero bytes; the original logical SHA-256 is
`d3aa7e4ba16e7983851850ae1bb6d01f09eeeb8d19b05e793f31697c9ae3d142`.
The exact sparse-trim evidence is retained in
`data/index/reports/mugen-anime-ascension-original-sparse-trim-v1.json`.

This is a research backup, not a redistribution license. Source index records retain
where each corpus or archive came from, the hashes observed, rights evidence where
available, and unresolved caveats. No ownership or permission for third-party
characters or artwork is asserted by placing a private backup in Drive.
