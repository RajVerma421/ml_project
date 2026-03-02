from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
def Script(content_type,topic,emotion):
    promptTemplete = PromptTemplate(
    template="""
You are a viral social media influencer with over 1 million followers.

Create a high-engagement {content_type} script about: {topic}
Target Emotion: {emotion}

Audience: Gen-Z and Millennials
Platform: Instagram / YouTube Shorts / TikTok

Script Requirements:
• Start with a scroll-stopping hook in the first 3 seconds.
• Deliver the core message clearly and confidently.
• Maintain the emotion: {emotion}.
• Keep sentences short and punchy.
• Use conversational language (like speaking to a friend).
• Add natural emojis (not too many).
• End with a strong call-to-action that encourages engagement.

Constraints:
• Maximum duration: 30–45 seconds.
• No robotic tone.
• No generic phrases.
• Make it feel authentic and relatable.

Output only the final script. No explanations.
""",
    input_variables=["content_type","topic","emotion"],
    validate_template=True
    )

    prompt=promptTemplete.invoke(
        {
            'content_type':content_type,
            'topic':topic,
            'emotion':emotion
        }
    )
    llm=HuggingFaceEndpoint(   # here we are telling which repo we are going to use and for which task.
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
    )
    model=ChatHuggingFace(llm=llm)
    response=model.invoke(prompt)

    return response.content


