import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-None-FtOT9TruXQa3Pxx2Vud5T3BlbkFJD3gsHBUidyz92NPaoBCF"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)