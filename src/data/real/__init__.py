"""Real external dataset adapters.

Adapters convert public research datasets into the same per-sample schema used by
`src/train.py` ({text, feature_vector, hook_label, drop_off_risk, quality_score,
group_id, ...}).

Licensing note: datasets are fetched by the user at run time and are never
redistributed by this repo. Each adapter records its source and license flags in
the produced metadata so reports can state provenance honestly.
"""
