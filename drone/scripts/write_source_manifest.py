"""Record exact working-tree source and dataset hashes, beyond the base Git SHA."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/focused_demo'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = list(root.glob('*.py')) + list(root.glob('*.md'))
    sources += [root / 'pyproject.toml', root / 'uv.lock']
    for directory in ['drone_dual_gatekeeper', 'web_demo', 'scripts', 'tests']:
        sources += [p for p in (root / directory).rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.suffix != '.pyc']
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    output = args.output.resolve()
    records = sorted(p for p in output.glob('*.json')
                     if p.stem.endswith(('_proposed', '_passive', '_weighted', '_robust')))
    artifacts = list(output.glob('*.html')) + [output / 'independent_validation.json',
                                              output / 'geometry_audit.json']
    git = lambda *cmd: subprocess.check_output(['git', *cmd], cwd=root, text=True).strip()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'base_git_revision': git('rev-parse', 'HEAD'),
        'working_tree_dirty': bool(git('status', '--porcelain')),
        'python': platform.python_version(),
        'dependencies': {name: version(name) for name in ['numpy', 'scipy', 'casadi', 'meshcat']},
        'source_sha256': {str(p.relative_to(root)): digest(p) for p in sorted(set(sources))},
        'record_sha256': {p.name: digest(p) for p in records + artifacts if p.exists()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Source and dataset hashes: {output / "source_manifest.json"}')


if __name__ == '__main__':
    main()
