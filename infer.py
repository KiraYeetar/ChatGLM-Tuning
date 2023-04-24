import os
import sys

import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

def main(
    load_8bit: bool = True,
    base_model: str = "THUDM/chatglm-6b",
    lora_weights: str = "mymusise/chatGLM-6B-alpaca-lora",
    share_gradio: bool = False,
):
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = AutoModel.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights
    )
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


    def evaluate(
        input_text=None,
    ):
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_length=240,
                do_sample=False,
                temperature=0
            )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        yield answer

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Context",
                placeholder="今晚月色真美啊",
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="ChatGLM-Tuning",
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
