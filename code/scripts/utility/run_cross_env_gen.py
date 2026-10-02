"""Run the complete 12-model paired cross-environment audit grid."""
from code.scripts.audited_results import TrainingAudit
from code.scripts.utility.cross_env_gen import TARGET_MODELS, aggregate, parser, run_model


def main():
    args = parser().parse_args()
    audit = TrainingAudit(args.training_manifest)
    if not args.aggregate_only:
        audit.protect_output(args.out_root)
        for model in TARGET_MODELS:
            run_model(args, audit, model)
    aggregate(args.out_root, args.table_path, audit, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
