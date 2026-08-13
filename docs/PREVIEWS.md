# Generated clip previews

Model samples remain canonical uint8 RGBA NumPy arrays. `spritelab.previews`
creates two display-only derivatives from one sample:

- an animated PNG with the recorded per-frame durations and loop behavior; and
- a horizontal contact sheet.

Every visible pixel is enlarged only by a positive integer with nearest-neighbor
sampling. RGB values under fully transparent pixels are zeroed for robust display
across viewers; the sidecar retains the exact original array-content hash and records
this policy explicitly.
The JSON sidecar records native/display dimensions, timing, loop semantics, source
sample/report hashes, array-content hash, and hashes of both preview files. Preview
generation refuses unsafe filename stems and existing targets unless replacement is
explicitly requested. These derivatives are for inspection; they are not fed back
into training or represented as new source evidence.
