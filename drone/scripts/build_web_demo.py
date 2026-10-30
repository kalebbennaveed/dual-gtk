"""Build the three recorded demos into a static hosting directory."""
import argparse
from pathlib import Path

from scripts.export_web_demo import export

ROOT = Path(__file__).resolve().parents[1]
DEMOS = (
    ('informative_demo', 'focused_demo', 'index.html'),
    ('spatial_learning_course', 'spatial_course', 'scenario-2/index.html'),
    ('slalom_course', 'slalom_course', 'scenario-3/index.html'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dist/web-demo'))
    args = parser.parse_args()
    # Check every input before writing any page.
    missing = [ROOT / 'results' / source / f'{scenario}_{method}.json'
               for scenario, source, _ in DEMOS
               for method in ('proposed', 'passive', 'weighted', 'robust')
               if not (ROOT / 'results' / source / f'{scenario}_{method}.json').is_file()]
    if missing:
        parser.error('Missing simulation records: ' + ', '.join(map(str, missing)))
    for scenario, source, filename in DEMOS:
        export(ROOT / 'results' / source, args.output / filename, scenario)
    # Keep previously shared file URLs working while all navigation uses folders.
    for old,target in (('scenario-2.html','scenario-2/'),('learning.html','scenario-2/'),('slalom.html','scenario-3/')):
        (args.output/old).write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Dual Gatekeeper Drone</title><meta http-equiv="refresh" content="0;url={target}"><script>location.replace("{target}"+location.search+location.hash);</script><a href="{target}">Open scenario</a></html>')
    print(f'Upload the contents of {args.output.resolve()} to your static host.')


if __name__ == '__main__':
    main()
