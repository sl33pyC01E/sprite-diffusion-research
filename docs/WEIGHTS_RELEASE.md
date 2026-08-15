# Research weight releases

The GitHub releases for this repository contain model artifacts produced by this
research project. They do **not** contain the collected sprite corpus, original
MUGEN character archives, downloaded foundation models, or cached training tensors.

`releases/best-weights-v1.json` is the authoritative asset index. It records the
exact local source path, published asset name, byte count, SHA-256, experiment tier,
and a short claim boundary for every weight file. Verify a downloaded asset before
loading it.

See `docs/STUDY_GALLERY.md` for animated targets, generated outputs, held-out
examples, failed controls, codec reconstructions, and the classifier evaluation
associated with these studies.

The release has two purposes:

- preserve the current compact three-stage MUGEN pipeline and its resumable Stage-1
  state;
- preserve one best or most informative checkpoint from each materially different
  architecture tested during the research.

The `rejected-control` label is deliberate. Those weights are retained for
reproducibility, not recommended as a quality path. The `historical-best` and
`pipeline-predecessor` labels likewise do not mean production quality.

These weights were trained on a provenance-indexed research corpus containing
third-party artwork under mixed and sometimes unclear terms. No license or right to
any underlying character, franchise, sprite, or source asset is granted with the
weights. The artifacts are published for non-commercial research and reproducibility;
users remain responsible for their own use and for obtaining any rights they need.

Several historical adapters require separately obtained upstream model weights. The
upstream weights are intentionally not republished here. Their pinned identities and
source revisions remain documented in `docs/EXPERIMENTS.md` and the source indexes.
