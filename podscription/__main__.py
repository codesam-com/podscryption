import argparse
from pathlib import Path

from podscription.runner import run_job
from podscription.resolver import resolve_unknowns


def main() -> None:
    p = argparse.ArgumentParser(prog="podscription")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a job YAML end-to-end")
    p_run.add_argument("job_yaml", type=str)

    p_res = sub.add_parser("resolve", help="Resolve UNKNOWNs using review YAML referenced by job")
    p_res.add_argument("job_yaml", type=str)

    args = p.parse_args()
    job_path = Path(args.job_yaml)

    if args.cmd == "run":
        run_job(job_path)
    elif args.cmd == "resolve":
        resolve_unknowns(job_path)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
