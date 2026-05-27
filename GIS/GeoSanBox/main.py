import argparse
import json
import os
import sys

from core.executor import GeoExecutor, AgentCodeRunner, CodeResult
from core.tool_manager import registry, get_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geosandbox",
        description="GeoSandbox - 原子级 GIS 智能体执行沙盒",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    list_parser = subparsers.add_parser("list", help="列出所有可用工具")
    list_parser.add_argument(
        "--category", "-c", type=str, default=None, help="按分类过滤（vector / raster）"
    )
    list_parser.add_argument(
        "--schema", action="store_true", help="以 OpenAI function-calling 格式输出"
    )

    run_parser = subparsers.add_parser("run", help="在沙盒中执行代码")
    run_parser.add_argument(
        "--code", "-c", type=str, help="直接传入 Python 代码字符串"
    )
    run_parser.add_argument(
        "--file", "-f", type=str, help="从文件读取 Python 代码"
    )
    run_parser.add_argument(
        "--workspace", "-w", type=str, default="./data", help="工作空间路径（默认 ./data）"
    )
    run_parser.add_argument(
        "--local", action="store_true", help="强制使用本地模式（不使用 Docker）"
    )
    run_parser.add_argument(
        "--timeout", type=int, default=300, help="执行超时秒数（默认 300）"
    )
    run_parser.add_argument(
        "--retry", type=int, default=1, help="失败时重试次数（默认 1）"
    )

    info_parser = subparsers.add_parser("info", help="查看沙盒环境信息")
    info_parser.add_argument(
        "--workspace", "-w", type=str, default="./data", help="工作空间路径"
    )

    serve_parser = subparsers.add_parser("serve", help="启动 HTTP API 服务")
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="监听地址（默认 127.0.0.1）"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8090, help="监听端口（默认 8090）"
    )
    serve_parser.add_argument(
        "--workspace", "-w", type=str, default="./data", help="工作空间路径"
    )
    serve_parser.add_argument(
        "--local", action="store_true", help="本地模式"
    )

    return parser


def cmd_list(args):
    reg = get_registry()

    if not registry._tools:
        reg.auto_discover(["tools.vector_tools", "tools.raster_tools"])

    if args.schema:
        tools = reg.to_openai_schema()
        print(json.dumps(tools, indent=2, ensure_ascii=False))
        return

    if args.category:
        tools = reg.list_by_category(args.category)
    else:
        tools = reg.list_all()

    if not tools:
        print("No tools registered. Run with --schema to see the format.")
        return

    print(f"\nTotal tools: {len(tools)}")
    print(f"Categories: {', '.join(reg.get_categories())}\n")
    print("=" * 70)

    current_category = ""
    for tool in tools:
        cat = tool.get("category", "general")
        if cat != current_category:
            current_category = cat
            print(f"\n--- {cat.upper()} ---\n")

        required = ", ".join(tool.get("required", []))
        params = tool.get("parameters", {})
        param_lines = []
        for pname, pinfo in params.items():
            req_mark = "*" if pname in tool.get("required", []) else ""
            param_lines.append(f"    {pname}{req_mark}: {pinfo.get('description', '')}")

        print(f"  {tool['name']}")
        print(f"    {tool['description']}")
        if param_lines:
            print("\n".join(param_lines))
        print()

    print("=" * 70)
    print("* = required parameter\n")


def cmd_run(args):
    code = args.code
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    if not code:
        print("Error: Must provide --code or --file", file=sys.stderr)
        sys.exit(1)

    executor = GeoExecutor(
        workspace_path=args.workspace,
        timeout=args.timeout,
        use_docker=not args.local,
    )

    runner = AgentCodeRunner(
        workspace_path=args.workspace,
    )

    for attempt in range(args.retry):
        if attempt > 0:
            print(f"\n--- Retry {attempt + 1}/{args.retry} ---\n")

        result = executor.run_code(code)
        result_obj = CodeResult(result)

        if result_obj.success:
            print(f"[SUCCESS] exit_code={result_obj.exit_code}")
            if result_obj.logs:
                print(result_obj.logs)
            return
        else:
            print(f"[FAILED] exit_code={result_obj.exit_code}")
            if result_obj.logs:
                print(result_obj.logs, file=sys.stderr)

    print("\nAll retries exhausted.", file=sys.stderr)
    sys.exit(1)


def cmd_info(args):
    executor = GeoExecutor(
        workspace_path=args.workspace,
        use_docker=True,
    )
    reg = get_registry()

    if not registry._tools:
        reg.auto_discover(["tools.vector_tools", "tools.raster_tools"])

    print(f"GeoSandbox Environment Info")
    print("=" * 50)
    print(f"  Workspace:      {executor.workspace}")
    print(f"  Docker image:   {executor.image}")
    print(f"  Docker mode:    {'Available' if executor.docker_available else 'Unavailable (local fallback)'}")
    print(f"  Default timeout: {executor.timeout}s")
    print(f"  Memory limit:   {executor.memory_limit}")
    print(f"  CPU limit:      {executor.cpu_limit} cores")
    print(f"  Python:         {sys.version}")
    print(f"  Tools loaded:   {len(registry._tools)}")
    print("=" * 50)

    categories = reg.get_categories()
    for cat in categories:
        tools = reg.list_by_category(cat)
        print(f"\n  [{cat}] {len(tools)} tools:")
        for t in tools:
            print(f"    - {t['name']}: {t['description']}")


def cmd_serve(args):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Error: Flask is required for serve mode. Install with: pip install flask", file=sys.stderr)
        sys.exit(1)

    app = Flask(__name__)
    executor = GeoExecutor(
        workspace_path=args.workspace,
        use_docker=not args.local,
    )
    reg = get_registry()
    reg.auto_discover(["tools.vector_tools", "tools.raster_tools"])

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "docker_available": executor.docker_available,
            "tools_count": len(registry._tools),
        })

    @app.route("/tools", methods=["GET"])
    def list_tools():
        category = request.args.get("category")
        schema = request.args.get("schema", "").lower() == "openai"

        if schema:
            return jsonify(reg.to_openai_schema())
        if category:
            return jsonify(reg.list_by_category(category))
        return jsonify(reg.list_all())

    @app.route("/execute", methods=["POST"])
    def execute():
        data = request.get_json(silent=True) or {}
        code = data.get("code", "")
        if not code:
            return jsonify({"error": "Missing 'code' field"}), 400

        result = executor.run_code(code)
        return jsonify(result)

    @app.route("/tools/<tool_name>", methods=["POST"])
    def execute_tool(tool_name):
        data = request.get_json(silent=True) or {}
        result = reg.execute(tool_name, **data)
        return jsonify(result)

    print(f"\nGeoSandbox API starting at http://{args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  GET  /health          - Health check")
    print(f"  GET  /tools           - List tools")
    print(f"  POST /execute         - Execute Python code")
    print(f"  POST /tools/<name>    - Execute a named tool directly\n")
    app.run(host=args.host, port=args.port, debug=False)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
