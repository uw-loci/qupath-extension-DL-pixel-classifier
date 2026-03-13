"""Deep Learning Pixel Classifier Server for QuPath."""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("dlclassifier-server")
except Exception:
    __version__ = "0.2.5"  # fallback when running from JAR-bundled scripts

# Protocol version for Java/Python compatibility checking.
# Bump this integer when the Python service API changes in a way that
# affects correctness (see docs/APPOSE_DEV_GUIDE.md for guidelines).
PROTOCOL_VERSION = 2
