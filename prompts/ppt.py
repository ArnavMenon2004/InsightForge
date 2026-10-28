from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Slide(BaseModel):
    title: str
    bullets: list[str] = Field(
        description="5-7 substantive bullet points; each 1-2 full sentences (15-30 words) with concrete details."
    )


class SlideOutline(BaseModel):
    presentation_title: str
    slides: list[Slide]


PPT_OUTLINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert at creating conference presentation slide decks from research papers. "
                "For Quick Summary produce exactly 5 slides: Title, Problem + Gap, Approach, Results, Conclusion. "
                "For Full Conference PPT produce 10-12 slides: Title, Problem Statement, Related Work (2 slides), "
                "Methodology, System Architecture, Key Results, Evaluation, Limitations & Future Work, Conclusion. "
                "Each slide must have a clear title and 5-7 substantive bullet points. "
                "Every bullet must be a complete, informative statement of 1-2 sentences (roughly 15-30 words) "
                "that carries real content from the paper - definitions, reasoning, method details, numbers, "
                "findings and their interpretation - not a short phrase or a vague headline. "
                "Slides must be dense enough to present the paper on their own, even in Quick Summary mode. "
                "The first slide's title must be the paper's actual title, and every other slide title "
                "must be descriptive of its content, not a generic label such as 'Title Slide'. "
                "Use concrete specifics from the paper (sample size, methods, instruments, statistics, "
                "findings) wherever they exist. Never invent facts or numbers that are not in the paper text."
            ),
        ),
        (
            "human",
            (
                "Paper text:\n{paper_text}\n\n"
                "Mode: {mode}\n"
                "Paper title (if found): {paper_title}\n\n"
                "Generate the complete slide outline."
            ),
        ),
    ]
)
