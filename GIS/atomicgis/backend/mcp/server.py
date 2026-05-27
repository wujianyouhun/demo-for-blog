from fastapi import FastAPI

app = FastAPI(title="AtomicGIS MCP")

@app.get("/tools")
def list_tools():
    return {
        "tools": [
            "vector.buffer",
            "io.load_vector"
        ]
    }