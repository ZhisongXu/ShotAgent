from executor import executor


tool_cnt = 0
for subtask, toolbox in executor.toolbox_router.items():
    print(f"Subtask: {subtask}\n{len(toolbox)} tools:", end=' ')
    for tool in toolbox:
        print(tool.tool_name, end=' ')
        tool_cnt += 1
    print('\n')


print(f"Total: {len(executor.subtasks)} subtasks, {tool_cnt} tools.")