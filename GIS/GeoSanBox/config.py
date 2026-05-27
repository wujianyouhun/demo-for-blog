import os

WORKSPACE_DIR = os.environ.get("GEOSANDBOX_WORKSPACE", "./data")
DOCKER_IMAGE = os.environ.get("GEOSANDBOX_IMAGE", "geosandbox-env:latest")
TIMEOUT_SECONDS = int(os.environ.get("GEOSANDBOX_TIMEOUT", "300"))
MEMORY_LIMIT = os.environ.get("GEOSANDBOX_MEMORY", "2g")
CPU_LIMIT = float(os.environ.get("GEOSANDBOX_CPU", "2.0"))
USE_DOCKER = os.environ.get("GEOSANDBOX_USE_DOCKER", "true").lower() != "false"
TOOL_MODULES = [
    "tools.vector_tools",
    "tools.raster_tools",
]
API_HOST = os.environ.get("GEOSANDBOX_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("GEOSANDBOX_API_PORT", "8090"))

MAX_RETRIES = int(os.environ.get("GEOSANDBOX_MAX_RETRIES", "3"))
