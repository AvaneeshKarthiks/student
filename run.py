"""
Root execution entry point for the Business Entity Resolution Pipeline.
Passes arguments directly to the pipeline engine in code/business_entity_resolution/src.
"""
import runpy
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Run as a proper package module so relative imports inside src work correctly
runpy.run_module(
    "code.business_entity_resolution.src.run_pipeline",
    run_name="__main__",
    alter_sys=True,
)