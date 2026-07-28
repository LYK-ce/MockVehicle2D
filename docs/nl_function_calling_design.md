# 自然语言意图到 Robot Controller v4 的边界

自然语言解析不属于模拟车的执行环。可选的
`mockvehicle2d.instruction.dispatcher` 只把已校验意图翻译成一条 v4 命令，
不持有模式、任务队列或车辆控制权。

```python
from mockvehicle2d.instruction import translate

result = translate(
    {"intent": "goto", "parameters": {"x_m": 20.0, "y_m": 30.0}},
    seq=11,
    mission_id="nl-11",
)
```

`goto` 返回合法的 `auto/push`，但不会隐式切换模式。调用方必须先发送自己的
`mode/switch_to_auto`，并负责连接内严格递增的 `seq` 和进程生命周期内唯一的
`mission_id`。

`stop` 翻译成模式无关的 `mode/stop_motion`；`clarify` 不产生执行命令。
Robot Controller v4 尚不支持 `patrol`，因此翻译器明确拒绝该意图。

WebSocket Server 不接受 `nl_command`、`nl_task_update` 或任何旧自然语言控制消息。
所有可执行命令仍经过 v4 协议校验并由唯一的 `RobotController` 执行。
