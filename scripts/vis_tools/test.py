import gradio as gr
import numpy as np

def load_file(file):
    # 假设文件内容是点云数据
    data = np.load(file.name)  # 加载文件
    return data.tolist()

def update_dropdown(file):
    data = load_file(file)
    return gr.Dropdown(choices=data, label="Select a point")

with gr.Blocks() as demo:
    with gr.Row():
        file_input = gr.File(label="Upload a file")
        dropdown = gr.Dropdown(choices=[], label="Select a point", interactive=True)
    file_input.change(fn=update_dropdown, inputs=file_input, outputs=dropdown)

demo.launch()