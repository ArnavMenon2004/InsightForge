from langchain_core.prompts import ChatPromptTemplate

CONTRADICTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a critical reader checking a multi-section research report for internal consistency. "
                "Identify claims that directly contradict each other across different sections. "
                "Only flag genuine factual contradictions (e.g. Section A says X increased, Section B says X decreased). "
                "Do not flag nuance, caveats, or different aspects of the same topic. "
                "If the report is internally consistent, return an empty contradictions list."
            ),
        ),
        (
            "human",
            "Report sections:\n\n{sections_text}\n\nFind all factual contradictions between sections.",
        ),
    ]
)
