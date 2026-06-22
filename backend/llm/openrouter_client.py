from openai import OpenAI

from config.settings import settings


class OpenRouterClient:

    def __init__(self):

        self.client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

    def generate_response(
        self,
        prompt: str,
        model: str = None
    ) -> str:
        model = model or settings.DEFAULT_MODEL

        response = (
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful "
                            "PDF assistant."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )