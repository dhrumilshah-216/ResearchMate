from pydantic import BaseModel, Field, model_validator

class TopicBrief(BaseModel):
    topic: str = Field(
        description="The main topic being discussed."
    )
    summary: str = Field(
        description="A concise explanation of the topic."
    )
    key_points: list[str] = Field(
        description="Important concepts or findings."
    )
    open_questions: list[str] = Field(
        description="Questions that need further investigation."
    )

class ResearchQuestion(BaseModel):
    question: str = Field(
        min_length=10,
        description="A specific, independently researchable question."
    )
    purpose: str = Field(
        min_length=10,
        description="Why investigating this question matters."
    )

class ResearchPlan(BaseModel):
    objective: str = Field(
        min_length=15,
        description="The main objective of the research."
    )
    research_questions: list[ResearchQuestion] = Field(
        min_length=3,
        max_length=5,
        description="Three to five distinct research questions."
    )
    assumptions: list[str] = Field(
        description="Assumptions made while interpreting the request."
    )
    excluded_topics: list[str] = Field(
        description="Topics intentionally excluded from the research."
    )

    @model_validator(mode="after")
    def validate_unique_questions(self):
        normalized = [
            item.question.strip().casefold()
            for item in self.research_questions
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Research questions must be unique.")
        return self

class ResearchSource(BaseModel):
    title: str
    url: str
    content: str = Field(
        description="Available extracted content from the source."
    )
    snippet: str = ""
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Search provider relevance score."
    )

class SearchResponse(BaseModel):
    query: str
    sources: list[ResearchSource]