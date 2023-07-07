import os
from typing import Optional, Tuple

import gradio as gr
from about_time import about_time
from types import SimpleNamespace

# from langchain.llms import CTransformers
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import CTransformers
from utils import format_prompt


from loguru import logger

load_dotenv()
model_path = os.environ.get("MODEL_PATH")


@dataclass
class GenerationConfig:
    # sample
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    last_n_tokens: int
    seed: int

    # eval
    batch_size: int
    threads: int

    # generate
    max_new_tokens: int
    stop: list[str]
    stream: bool
    reset: bool


# @dataclass
# class Namespace:
# ns = Namespace()
ns = SimpleNamespace(
    response="",
    generator=[],
)


def predict0(prompt,bot):
    logger.debug(f"{prompt=}, {bot=}")

    ns.response = ""
    with about_time() as atime:  # type: ignore
        try:

            generator = llm(format_prompt(prompt), **asdict(generation_config))

            response = ""

            for word in generator:
                # print(word, end="", flush=True)
                print(word, flush=True)  # vertical stream
                response += word
                ns.response = response

       

            logger.debug(f"{response=}")
        except Exception as exc:
            print(str(exc))
            raise

    # bot = {"inputs": [response]}
    _ = (
        f"(time elapsed: {atime.duration_human}, "  # type: ignore
        f"{atime.duration/(len(prompt) + len(response)):.1f}s/char)"  # type: ignore
    )

    bot.append([prompt, f"{response} {_}"])

    return prompt, bot




class Chat:
    default_system_prompt = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
    system_format = "<|im_start|>system\n{}<|im_end|>\n"

    def __init__(
        self, system: str | None = None, user: str | None = None, assistant: str | None = None
    ) -> None:
        if system is not None:
            self.set_system_prompt(system)
        else:
            self.reset_system_prompt()
        self.user = user if user else "<|im_start|>user\n{}<|im_end|>\n"
        self.assistant = (
            assistant if assistant else "<|im_start|>assistant\n{}<|im_end|>\n"
        )
        self.response_prefix = self.assistant.split("{}", maxsplit=1)[0]

    def set_system_prompt(self, system_prompt):
        # self.system = self.system_format.format(system_prompt)
        return system_prompt

    def reset_system_prompt(self):
        return self.set_system_prompt(self.default_system_prompt)

    def history_as_formatted_str(self, system, history) -> str:
        system = self.system_format.format(system)
        text = system + "".join(
            [
                "\n".join(
                    [
                        self.user.format(item[0]),
                        self.assistant.format(item[1]),
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += self.user.format(history[-1][0])
        text += self.response_prefix
        # stopgap solution to too long sequences
        if len(text) > 4500:
            # delete from the middle between <|im_start|> and <|im_end|>
            # find the middle ones, then expand out
            start = text.find("<|im_start|>", 139)
            end = text.find("<|im_end|>", 139)
            while end < len(text) and len(text) > 4500:
                end = text.find("<|im_end|>", end + 1)
                text = text[:start] + text[end + 1 :]
        if len(text) > 4500:
            # the nice way didn't work, just truncate
            # deleting the beginning
            text = text[-4500:]

        return text

    # def clear_history(self, history):
    def clear_history(self):
        return []

    # def turn(self, user_input: str):
    def turn(self, user_input: str, system, history):
        # self.user_turn(user_input)
        self.user_turn(user_input, history)
        # return self.bot_turn()
        return self.bot_turn(system, history)

    def user_turn(self, user_input: str, history):
        history.append([user_input, ""])
        return user_input, history

    def bot_turn(self, system, history):
        conversation = self.history_as_formatted_str(system, history)
        assistant_response = call_inf_server(conversation)
        history[-1][-1] = assistant_response
        print(system)
        print(history)
        return "", history


if os.path.exists(model_path):
    logger.info("Loading model...")
    llm = CTransformers(
        model=os.path.abspath(model_path),
        model_type="mpt",
        callbacks=[StreamingStdOutCallbackHandler()],
    )
else:
    logger.info("Model not found. Please run `python download_model.py` to download the model.")


system_prompt = "A conversation between a user and an LLM-based AI assistant named Local Assistant. Local Assistant gives helpful and honest answers."
generation_config = GenerationConfig(
    temperature=0.1,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=512,  # adjust as needed
    seed=42,
    reset=False, # reset history (cache)
    stream=True,  # streaming per word/token
    threads=int(os.cpu_count() / 2),  # adjust for your CPU
    stop=["<|im_end|>", "|<"],
    last_n_tokens=64,
    batch_size=8,
)

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")
with block:
    with gr.Row():
        gr.Markdown("<h3><center>MPT 30B CPU Demo</center></h3>")

    conversation = Chat()       
    chatbot = gr.Chatbot(scroll_to_output=True).style(height=700) 
    with gr.Row():
        with gr.Column(scale=1):
            message = gr.Textbox(
                label="Chat Message Box",
                placeholder="What's the answer to life, the universe, and everything?",
                lines=1,
            )
        with gr.Column(scale=0.1):
            with gr.Row():
                submit = gr.Button("Submit", elem_classes="xsmall")
                stop = gr.Button("Stop", visible=False)
                clear = gr.Button("Clear History", visible=True)


    gr.Examples(
        examples=[
            "Hi! How's it going?",
            "What should I do tonight?",
            "Whats 2 + 2?",
            "为什么爸妈结婚没有叫我参加婚礼",
            "十二生肖为什么没有猫",
            "一个笼子里面装着兔子和鸡，兔子有四只，鸡有三只，问有几只脚",
            "一个笼子里面装着兔子和鸡，一共20个头，问有几只兔子，几只鸡",
            "为什么我总是不开心",
            "js 判断一个数是不是质数",
            "js 实现python 的 range(10)",
            "js 实现python 的 [*(range(10)]"
        ],
        inputs=message,
    )

    gr.HTML("Demo application of mpt 30B in cpu")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/jiangsi/mpt-30B-cpu'>mpt-30B-cpu</a></center>"
    )



    _ = """
    submit_event = msg.submit(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot],
        outputs=[msg, chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        # fn=conversation.bot_turn,
        inputs=[system, chatbot],
        outputs=[msg, chatbot],
        queue=True,
    )

    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False).then(
        fn=conversation.clear_history,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=False,
    )
    change.click(
        fn=conversation.set_system_prompt,
        inputs=[system],
        outputs=[system],
        queue=False,
    )
    reset.click(
        fn=conversation.reset_system_prompt,
        inputs=[],
        outputs=[system],
        queue=False,
    )
    # """

    message.submit(
        fn=predict0,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
        queue=False,
        show_progress="full",
    ).then(
        conversation.bot_turn, chatbot, chatbot
    )

    submit.click(
    fn=lambda x, y: ("",) + predict0(x, y)[1:],  # clear msg
    inputs=[message, chatbot],
    outputs=[message, chatbot],
    queue=True,
    show_progress="full",
    )


    clear.click(lambda: None, None, chatbot, queue=False)

block.queue(concurrency_count=5, max_size=20).launch(debug=True)