"""Protocol constants for the published PepSite route."""

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GENERATION_ALPHABET = CANONICAL_AMINO_ACIDS + "X"
CANONICAL_SET = frozenset(CANONICAL_AMINO_ACIDS)
GENERATION_ALPHABET_SET = frozenset(GENERATION_ALPHABET)

# These values are validation-frozen protocol values, not tunable test-time
# parameters.  The ratio is an eligibility gate; it is never a ranking term.
PPL_RATIO_MAX = 1.05
COMPOSITION_L1_MAX = 0.75
CONTACT_CUTOFF_ANGSTROM = 5.0

HYDROPHOBIC_AMINO_ACIDS = frozenset("AILMFWVY")
CHARGED_AMINO_ACIDS = frozenset("DEKR")
