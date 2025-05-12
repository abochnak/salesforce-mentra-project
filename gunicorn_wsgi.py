import os
import tempfile

try:
    from app import app
except ImportError:
    from . import app


if __name__ == '__main__':
    app.run()