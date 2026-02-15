"""Vercel/WSGI entrypoint for Vers3Dynamics Search."""

from vers3dynamics_search import app, ensure_system_initialized

# Ensure the RF processing loop is available when imported by serverless runtimes.
ensure_system_initialized()

