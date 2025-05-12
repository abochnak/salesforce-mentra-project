import os
import tempfile

if not 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
    os.environ['PROMETHEUS_MULTIPROC_DIR'] = tempfile.mkdtemp()

try:
    from app import app
except ImportError:
    from . import app


from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics

def child_exit(server, worker):
    GunicornInternalPrometheusMetrics.mark_process_dead_on_child_exit(worker.pid)

if __name__ == '__main__':
    app.run()