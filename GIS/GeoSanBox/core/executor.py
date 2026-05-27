import os
import re
import sys
import io
import traceback
import importlib
import ast
from typing import Any, Dict, Optional

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

DANGEROUS_PATTERNS = [
    r"import\s+subprocess",
    r"from\s+subprocess",
    r"import\s+os\s*\n.*system",
    r"os\.system\(",
    r"os\.popen\(",
    r"subprocess\.(call|run|Popen)",
    r"exec\(",
    r"eval\(",
    r"__import__\(",
    r"import\s+shutil",
    r"shutil\.rmtree",
    r"os\.remove\(",
    r"os\.unlink\(",
    r"importlib\.import_module\(",
    r"compile\(",
]

DEFAULT_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "FileNotFoundError": FileNotFoundError,
    "StopIteration": StopIteration,
}


def sanitize_code(code: str) -> str:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            raise ValueError(f"Harmful code pattern detected: {pattern}")
    return code


def validate_syntax(code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Python syntax error: {e}")


class GeoExecutor:
    DEFAULT_TIMEOUT = 300
    DEFAULT_MEMORY_LIMIT = "2g"
    DEFAULT_CPU_LIMIT = 2.0

    def __init__(
        self,
        workspace_path: str,
        image: str = "geosandbox-env:latest",
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
        use_docker: bool = True,
    ):
        self.workspace = os.path.abspath(workspace_path)
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.use_docker = use_docker and HAS_DOCKER
        self._client = None

        os.makedirs(self.workspace, exist_ok=True)

        if self.use_docker:
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as e:
                print(f"[GeoExecutor] Docker unavailable, falling back to local mode: {e}")
                self.use_docker = False

    def _build_runner_script(self, python_code: str) -> str:
        sanitize_code(code=python_code)
        validate_syntax(code=python_code)
        return python_code

    def _run_in_docker(self, python_code: str) -> Dict[str, Any]:
        sanitize_code(code=python_code)
        validate_syntax(code=python_code)

        escaped_code = python_code.replace('"', '\\"')
        cmd = f'python3 -c "{escaped_code}"'

        try:
            container = self._client.containers.run(
                self.image,
                command=cmd,
                volumes={self.workspace: {"bind": "/home/data", "mode": "rw"}},
                working_dir="/home/data",
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                detach=True,
                remove=True,
            )

            try:
                result = container.wait(timeout=self.timeout)
                logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            except Exception as e:
                container.kill()
                return {
                    "exit_code": -1,
                    "logs": f"Execution timeout or error: {str(e)}",
                    "error": str(e),
                }

            return {
                "exit_code": result.get("StatusCode", -1),
                "logs": logs,
            }

        except docker.errors.ImageNotFound:
            raise RuntimeError(
                f"Docker image '{self.image}' not found. "
                f"Build it with: docker build -t {self.image} ./docker"
            )
        except Exception as e:
            return {
                "exit_code": -1,
                "logs": f"Docker execution error: {str(e)}",
                "error": str(e),
            }

    def _run_locally(self, python_code: str) -> Dict[str, Any]:
        sanitize_code(code=python_code)
        validate_syntax(code=python_code)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            sys.path.insert(0, os.path.dirname(self.workspace))

            exec_globals = {"__builtins__": DEFAULT_SAFE_BUILTINS}
            exec_globals.update({
                "os": __import__("os"),
                "sys": sys,
            })

            exec(python_code, exec_globals)

            exit_code = 0
        except Exception:
            traceback.print_exc(file=stderr_capture)
            exit_code = 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        out = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        logs = out
        if err:
            logs += "\n[STDERR]\n" + err

        return {
            "exit_code": exit_code,
            "logs": logs,
        }

    def run_code(self, python_code: str) -> Dict[str, Any]:
        try:
            validated_code = self._build_runner_script(python_code)
        except (ValueError, SyntaxError) as e:
            return {
                "exit_code": -1,
                "logs": f"Code validation error: {str(e)}",
                "error": str(e),
            }

        if self.use_docker:
            return self._run_in_docker(validated_code)

        return self._run_locally(validated_code)

    def run_with_tool_registry(
        self, tool_name: str, tool_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from core.tool_manager import get_registry

        reg = get_registry()
        result = reg.execute(tool_name, **tool_kwargs)

        return {
            "exit_code": 0 if result["status"] == "success" else 1,
            "logs": str(result),
            "result": result,
        }

    @property
    def docker_available(self) -> bool:
        return self.use_docker


class AgentCodeRunner:
    def __init__(self, workspace_path: str, image: str = "geosandbox-env:latest"):
        self.executor = GeoExecutor(
            workspace_path=workspace_path,
            image=image,
            timeout=600,
            memory_limit="4g",
            cpu_limit=4.0,
        )

    def run_agent_code(
        self, python_code: str, retry_on_error: bool = True, max_retries: int = 3
    ) -> Dict[str, Any]:
        last_result = None
        for attempt in range(1, max_retries + 1):
            result = self.executor.run_code(python_code)
            if result["exit_code"] == 0:
                result["attempt"] = attempt
                return result
            if not retry_on_error:
                result["attempt"] = attempt
                return result
            last_result = result

        last_result["attempt"] = max_retries
        last_result["logs"] += f"\nAll {max_retries} attempts failed."
        return last_result


class CodeResult:
    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw

    @property
    def success(self) -> bool:
        return self._raw.get("exit_code", -1) == 0

    @property
    def exit_code(self) -> int:
        return self._raw.get("exit_code", -1)

    @property
    def logs(self) -> str:
        return self._raw.get("logs", "")

    @property
    def error(self) -> Optional[str]:
        return self._raw.get("error")

    @property
    def raw(self) -> Dict[str, Any]:
        return self._raw

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"CodeResult(status={status}, exit_code={self.exit_code})"
