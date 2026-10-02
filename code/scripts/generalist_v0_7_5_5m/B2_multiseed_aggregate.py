"""Report the predeclared five-model B2 subset of the complete audited seed grid."""
from code.scripts.generalist_v0_7_5.aggregate_master import parser
from code.scripts.generalist_v0_7_5_5m.G5_multiseed_aggregate import publish

MODELS = ("stjewm_trace_only", "stjewm_spike_only", "stacked_lif_trace", "lewm_baseline_v2", "mlp_baseline")


def main():
    publish(parser(__doc__).parse_args(), MODELS, "Audited B2 five-model three-seed comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
