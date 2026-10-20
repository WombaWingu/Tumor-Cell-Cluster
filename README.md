# Tumor-Cell-Cluster

## Label quality audit

Masks come from per-slice Otsu thresholding, not expert annotation. Each
z-slice gets its own threshold estimated from its own histogram, so the
labels are only as stable as that estimate.

![xz cross-section, 107_Scan](figures/xz_107.png)
*Orthogonal (xz) view through 107_Scan. Otsu produces razor-straight
horizontal edges spanning ~180 px; the model's boundary is smooth.*

Those horizontal edges are the finding. They run exactly parallel to the
z-sampling axis, which no biological structure does. They are single
slices where the Otsu threshold landed lower than its neighbours and
captured dim tissue the adjacent slices rejected.

The artifact appears in 4 of 7 stacks (107, 111, 127, 59), and those are
the four with the largest single-slice coverage drops. The three stacks
with the smallest drops (35, 74, 81) show no bars and near-identical
Otsu/model masks.

![xz cross-section, 35_Scan](figures/xz_35.png)
*35_Scan, smallest Otsu drop (0.005). Both masks agree closely.*


Both methods run per-slice in 2D with no access to neighbouring slices.
The model's z-coherence therefore comes from applying a fixed learned
decision function rather than re-estimating a threshold each slice.

This also explains the gap between test Dice mean (0.863) and median
(0.918): a small band of unstable-label slices drags the mean while
typical slices match validation performance.

### Limitations

Dice here measures agreement with a weak-label baseline, not accuracy.
No expert masks exist for this data. z-coherence alone is not a quality
metric, since a mask predicting everything would score perfectly on it.

