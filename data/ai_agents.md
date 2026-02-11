# AI Agents

AI Agents are autonomous entities that use a Large Language Model (LLM) as their "brain" to reason through tasks, use tools, and interact with the world to achieve specific objectives.

## Functional Components
- **Planning**: The ability to break down complex goals into smaller, manageable steps (e.g., Chain of Thought).
- **Memory**: 
    - *Short-term*: The conversation history or current context.
    - *Long-term*: Information retrieved from a vector database.
- **Tool Use**: The ability to call external APIs (like `search_docs`) to gather information or take actions.

## Reasoning Loops
Agents often operate in loops. A common pattern is **ReAct** (Reason + Act), where the agent thinks about what to do, takes an action, observes the result, and then repeats the process until the task is complete.
