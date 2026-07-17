# GeoAI 空间 Agent

默认 local 模式不需要 API Key：它解析请求中的候选点数量，在演示边界内执行确定性空间计算并返回 GeoJSON。设置 OPENAI_API_KEY 后可选 openai 模式，但最终空间结果仍由可复现的本地工具生成。

启动：conda activate geoai，然后运行 python start.py --no-install。

默认前后端端口为 5186 / 8026。
