from planner.react_agent import ReActPlanner
from core.executor import execute_workflow

planner = ReActPlanner()

workflow = planner.build(
    "buffer roads"
)

result = execute_workflow(workflow)

print(result)