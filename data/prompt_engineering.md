# Prompt Engineering

Prompt Engineering is the art and science of crafting inputs (prompts) to get the most accurate and high-quality outputs from a generative AI model.

## Key Strategies
- **Zero-Shot**: Asking the model a question without any examples.
- **Few-Shot**: Providing 2-3 examples of the desired output format within the prompt.
- **Chain of Thought (CoT)**: Encouraging the model to "think step-by-step" to improve reasoning accuracy.

## Grounding & Safety
In RAG systems, prompt engineering is used to **ground** the model. By including instructions like *"Answer ONLY using the provided context"*, developers can prevent the model from using external, potentially incorrect information.
