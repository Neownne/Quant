# 项目规范

## Superpowers 工作流

在每次会话开始时，你必须：
1. 检查当前任务是否适用任何 skill（即使只有 1% 的可能性）
2. 如果适用，先加载 skill，再执行任何操作
3. 遵循 skill 中的检查清单和流程

## 可用 Skills 目录

项目 skills 位于 `.claude/skills/` 目录下：
- `using-superpowers` - 元技能（必须先加载）
- `brainstorming` - 需求澄清
- `writing-plans` - 制定计划
- `executing-plans` - 执行计划
- `test-driven-development` - TDD 流程
- `systematic-debugging` - 系统调试

## Skill 加载指令

当需要加载 skill 时，使用 Skill 工具调用对应的 skill 名称。
