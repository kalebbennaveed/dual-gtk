from pathlib import Path
import argparse
import json
from .simulation import simulate


def main():
    p=argparse.ArgumentParser(description="Dual Gatekeeper · DRS Racing")
    sub=p.add_subparsers(dest="command",required=True)
    run=sub.add_parser("simulate",help="Run the bicycle dynamics and gatekeeper")
    run.add_argument("--method",choices=["proposed","passive","weighted","robust","all"],default="proposed")
    run.add_argument("--seconds",type=float,default=65.)
    run.add_argument("--seed",type=int,default=7)
    run.add_argument("--mu",type=float,default=.85)
    run.add_argument("--initial-mu",type=float,nargs=2,metavar=('LOWER','UPPER'),default=(.45,1.05),help="Declared friction support, shared by estimation and certification")
    run.add_argument("--start",type=float,default=0.,help="Initial circuit arc length in metres")
    run.add_argument("--scenario",choices=["barcelona","traction","traction_relaxed"],default="traction")
    run.add_argument("--information-weight",type=float,default=.4,help="Weighted-baseline information weight; proposed always uses fixed gamma=2")
    run.add_argument("--max-cells",type=int,default=256)
    run.add_argument("--output",type=Path,default=Path("output"))
    web=sub.add_parser("demo",help="Build a self-contained browser replay of recorded runs")
    web.add_argument("--input",type=Path,default=Path("output"))
    web.add_argument("--output",type=Path,default=Path("output/mppi-web"))
    web.add_argument("--results",type=Path,help="Include a generated paper report in the browser demo")
    audit=sub.add_parser("audit",help="Independently check records, bounds and commitments")
    audit.add_argument("--input",type=Path,default=Path("output"))
    audit.add_argument("--backup-stress",action="store_true",help="Also run 39 empirical backup laps across the friction range")
    plot=sub.add_parser("plot",help="Plot recorded uncertainty, progress, speed and cost")
    plot.add_argument("--input",type=Path,default=Path("output"))
    plot.add_argument("--output",type=Path,default=Path("output/comparison.png"))
    video=sub.add_parser("video",help="Render the original PyBullet track and vehicle")
    video.add_argument("--input",type=Path,default=Path("output/proposed.json"))
    video.add_argument("--output",type=Path,default=Path("output/proposed.mp4"))
    video.add_argument("--seconds",type=float,default=15.)
    experiments=sub.add_parser("experiments",help="Run paired Monte Carlo episodes, a friction grid and weight sensitivity")
    experiments.add_argument("--output",type=Path,default=Path("results/paper"))
    experiments.add_argument("--trials",type=int,default=100,help="Independent friction draws per method (4 episodes per trial)")
    experiments.add_argument("--seed",type=int,default=20261007)
    experiments.add_argument("--seconds",type=float,default=360.)
    experiments.add_argument("--workers",type=int,default=4)
    experiments.add_argument("--scenario",choices=["traction","traction_relaxed"],default="traction")
    experiments.add_argument("--no-grid",action="store_true")
    experiments.add_argument("--no-weights",action="store_true")
    report=sub.add_parser("report",help="Generate paper figures, CSV/LaTeX tables and an HTML results gallery")
    report.add_argument("--input",type=Path,default=Path("results/paper"))
    comparison=sub.add_parser("comparison-report",help="Compare matched 1% and 5% sampled DRS studies")
    comparison.add_argument("--first",type=Path,default=Path("results/sampled-risk-01"))
    comparison.add_argument("--second",type=Path,default=Path("results/sampled-risk-05"))
    comparison.add_argument("--output",type=Path,default=Path("results/sampled-paper"))
    comparison.add_argument("--illustrations",type=Path,default=Path("output/sampled"))
    suite=sub.add_parser("demo-suite",help="Build the two-tolerance browser demo")
    suite.add_argument("--first",type=Path,default=Path("output/mppi/risk-01"))
    suite.add_argument("--second",type=Path,default=Path("output/mppi/risk-05"))
    suite.add_argument("--output",type=Path,default=Path("output/mppi-web"))
    suite.add_argument("--results",type=Path,help="Optionally include a generated paper report (omitted from public demo exports)")
    mppi_report=sub.add_parser("mppi-report",help="Build the local MPPI demonstration and validation report")
    mppi_report.add_argument("--source",type=Path,default=Path("output/mppi"))
    mppi_report.add_argument("--validation",type=Path,default=Path("results/mppi-validation"))
    mppi_report.add_argument("--output",type=Path,default=Path("results/mppi-report"))
    args=p.parse_args()
    if args.command=="mppi-report":
        from .mppi_report import build
        print(build(args.source,args.validation,args.output))
    elif args.command=="simulate":
        args.output.mkdir(parents=True,exist_ok=True)
        methods=["proposed","passive","weighted","robust"] if args.method=="all" else [args.method]
        for method in methods:
            print(f"Running {method} …",flush=True)
            result=simulate(method,args.seconds,args.seed,args.mu,args.max_cells,args.start,
                lambda t,x,b: print(f"  {t:6.2f} s  {x[0]:8.2f} m  μ ∈ [{b.lower:.5f}, {b.upper:.5f}]",flush=True),scenario=args.scenario,information_weight=args.information_weight,initial_mu=args.initial_mu)
            (args.output/f"{method}.json").write_text(json.dumps(result,allow_nan=False))
            print(json.dumps(result["summary"],indent=2))
    elif args.command=="demo":
        from .web import build
        print(build(args.input,args.output,args.results))
    elif args.command=="audit":
        from .validation import audit_directory
        print(json.dumps(audit_directory(args.input,args.backup_stress),indent=2))
    elif args.command=="video":
        from .video import render_video
        render_video(args.input,args.output,args.seconds)
    elif args.command=="plot":
        from .plots import plot_results
        print(plot_results(args.input,args.output))
    elif args.command=="experiments":
        from .experiments import run_suite
        print(run_suite(args.output,args.trials,args.seed,args.seconds,args.workers,not args.no_grid,not args.no_weights,args.scenario))
    elif args.command=="comparison-report":
        from .comparison_report import build_comparison
        print(build_comparison(args.first,args.second,args.output,args.illustrations))
    elif args.command=="demo-suite":
        from .web import build_suite
        print(build_suite(args.first,args.second,args.output,args.results))
    elif args.command=="report":
        from .paper_report import build_report
        print(build_report(args.input))


if __name__=="__main__": main()
