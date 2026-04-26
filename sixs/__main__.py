"""Allow `python -m sixs` to invoke the model."""
from .sixs_main import main
import sys
sys.exit(main())
