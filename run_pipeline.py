"""Root-level entry point — sets import path and delegates to src/pipeline.py."""
import sys, os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from pipeline import main
if __name__ == "__main__":
    main()
