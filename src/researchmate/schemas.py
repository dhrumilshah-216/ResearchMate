from pydantic import BaseModel, Field


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