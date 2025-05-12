#!/usr/bin/env python3

import os
import arrow
import logging
import json
import time
import asyncio

from flask import Flask, Response, request, abort, redirect, jsonify

is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
if is_gunicorn:
    from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics as PrometheusMetrics
else:
    from prometheus_flask_exporter import PrometheusMetrics

from prometheus_client import Counter, Gauge

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

app = Flask(__name__)

metrics = PrometheusMetrics(app)

@app.route('/')
def index():
    return jsonify(
        {}
    )

@app.route('/healthz')
def healthz_route():
    return 'ok'