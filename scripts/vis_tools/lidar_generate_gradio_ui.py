import gradio as gr
from utils.gradio_ui_utils import load_point_cloud_uncon

with gr.Blocks(title='AL', theme=gr.themes.Monochrome()) as demo:
    # layout definition
    with gr.Row():
        gr.Markdown("""
        #  Lidar generation with LiDM [LiDM](https://github.com/AlanLiangC/CentralScene)
        """)

    with gr.Tab(label=''):

        with gr.Row():
            canvas = gr.Plot(label="3D Point Cloud")
            
        with gr.Tab("Uncondition generate"):
            with gr.Row():
                dataset_choice = ['kitti', 'nuscenes']
                selected_dataset = gr.Dropdown(
                    label="Dataset for generate",
                    choices=dataset_choice + ['waymo'],
                    interactive=True
                )

                selected_model = gr.Dropdown(
                    label="Model for generate",
                    choices=['../model/lidm/kitti/uncond/model.ckpt',
                             '../model/lidm/nuscenes/uncond/model.ckpt'],
                    interactive=True
                )

            with gr.Row():
                uncon_generate_button = gr.Button("Uncondition Generate Point")

        with gr.Tab("Condition generate with layout"):
            with gr.Row():
                layout_generate_button = gr.Button("Generate Point w Layout")

    uncon_generate_button.click(
        load_point_cloud_uncon,
        [selected_dataset, selected_model],
        [canvas]
    )

demo.queue().launch(debug=True)
