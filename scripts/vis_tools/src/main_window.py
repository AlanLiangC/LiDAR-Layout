import socket
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout,\
                            QComboBox, QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit, QListWidget

from utils import gl_engine as gl
from utils.nusc_pcdet import NUSC_PCDet
from utils.generate_graph import generate_graph
from utils.common import box2coord3d
from vis_tools.functions import LidarScene, LiDM_Sampler
class MainWindow(QWidget):

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        host_name = socket.gethostname()
        if host_name == 'Liang':
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(self.monitor.height()*0.8))
            self.monitor.setWidth(int(self.monitor.width()*0.6))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(self.monitor.height()))
            self.monitor.setWidth(int(self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.grid_dimensions = 10
        self.sample_index = 0
        self.logger = gl.create_logger()
        self.init_window()

    def init_window(self):
        main_layout = QHBoxLayout()
        self.init_display_window()
        self.init_scene_graph_window()
        self.init_scene_graph_functions_window()
        main_layout.addLayout(self.display_layout)
        main_layout.addLayout(self.scene_graph_functions_layout)
        main_layout.addLayout(self.scene_graph_layout)

        main_layout.setStretch(0, 8)
        main_layout.setStretch(1, 2)
        main_layout.setStretch(2, 8)
        self.setLayout(main_layout)

    def init_scene_graph_functions_window(self):
        self.scene_graph_functions_layout = QVBoxLayout()
        # show words
        self.scene_triples_list = QListWidget()
        # self.scene_triples_list.itemClicked.connect(self.on_box_selected)
        self.scene_graph_functions_layout.addWidget(self.scene_triples_list)     

        # generate layout
        self.generate_layout_button = QPushButton("Gen Layout")
        self.scene_graph_functions_layout.addWidget(self.generate_layout_button)
        self.generate_layout_button.clicked.connect(self.generate_layout)
        # generate points
        self.generate_points_w_layout_button = QPushButton("Gen Points")
        self.scene_graph_functions_layout.addWidget(self.generate_points_w_layout_button)
        self.generate_points_w_layout_button.clicked.connect(self.generate_points_w_layout)
        # show scene graph
        self.show_scene_graph_button = QPushButton("Show Scene Graph")
        self.scene_graph_functions_layout.addWidget(self.show_scene_graph_button)
        self.show_scene_graph_button.clicked.connect(self.show_scene_graph)

    def init_scene_graph_window(self):
        self.scene_graph_layout = QVBoxLayout()
        # viewer
        self.scene_graph_viewer = gl.AL_viewer()
        self.scene_graph_layout.addWidget(self.scene_graph_viewer)

    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        # viewer
        self.viewer = gl.AL_viewer()
        self.display_layout.addWidget(self.viewer)
        # load dataset
        self.load_nusc_dataset_button = QPushButton('Load Nuscenes')
        self.display_layout.addWidget(self.load_nusc_dataset_button)
        self.load_nusc_dataset_button.clicked.connect(self.load_nusc_dataset)
        # << *** >>
        temp_layout = QHBoxLayout()
        # prev view
        self.prev_view_button = QPushButton('<<<')
        temp_layout.addWidget(self.prev_view_button)
        self.prev_view_button.clicked.connect(self.decrement_index)

        # Qlabel
        # show sample index
        self.sample_index_info = QLabel("")
        self.sample_index_info.setAlignment(Qt.AlignCenter)
        temp_layout.addWidget(self.sample_index_info)

        # Button
        # next view
        self.next_view_button = QPushButton('>>>')
        temp_layout.addWidget(self.next_view_button)
        self.next_view_button.clicked.connect(self.increment_index)
        self.display_layout.addLayout(temp_layout)

    def show_points(self):
        points = self.dataset.get_points(self.sample_index)
        mesh = gl.get_points_mesh(points[:,:3], 5)
        self.current_mesh = mesh
        self.viewer.addItem(mesh)

    def add_custom_points(self, points, viewer):
        viewer.items = []
        mesh = gl.get_points_mesh(points[:,:3], 5)
        viewer.addItem(mesh)

    def show_boxes_3d(self):
        scaled_boxes = self.scene_data_dict['encoder']['boxes']
        box_names = [self.dataset.scene_dataset.classes_r[idx] for idx in self.scene_data_dict['encoder']['objs'].numpy()]
        raw_boxes = self.dataset.scene_dataset.re_scale_box(scaled_boxes)
        box_info = gl.create_boxes(bboxes_3d=raw_boxes, box_texts=box_names)
        self.add_boxes_to_viewer(box_info)

        # add boxex corners
        boxes_corners = box2coord3d(raw_boxes)
        mesh = gl.get_points_mesh(boxes_corners[:,:3], 10)
        self.viewer.addItem(mesh)

    def show_triples(self):
        self.scene_triples_list.clear()
        words = self.scene_data_dict['encoder']['words']
        for i in range(len(words)):
            word = words[i]
            self.scene_triples_list.addItem(word)

    def add_boxes_to_viewer(self, box_info):
        # keep points
        # self.reset_viewer(only_viewer=True)
        # self.viewer.addItem(self.current_mesh)

        for box_item, l1_item, l2_item in zip(box_info['box_items'], box_info['l1_items'],\
                                                        box_info['l2_items']):
            self.viewer.addItem(box_item)
            self.viewer.addItem(l1_item)
            self.viewer.addItem(l2_item)
            
        if len(box_info['score_items']) > 0:
            for score_item in box_info['score_items']:
                self.viewer.addItem(score_item)

        if len(box_info['text_items']) > 0:
            for text_item in box_info['text_items']:
                self.viewer.addItem(text_item)

    def add_boxes_to_secene_graph(self, box_info):
        self.scene_graph_viewer.items = []
        for box_item, l1_item, l2_item in zip(box_info['box_items'], box_info['l1_items'],\
                                                        box_info['l2_items']):
            self.scene_graph_viewer.addItem(box_item)
            self.scene_graph_viewer.addItem(l1_item)
            self.scene_graph_viewer.addItem(l2_item)
            
        if len(box_info['score_items']) > 0:
            for score_item in box_info['score_items']:
                self.scene_graph_viewer.addItem(score_item)

        if len(box_info['text_items']) > 0:
            for text_item in box_info['text_items']:
                self.scene_graph_viewer.addItem(text_item)
        self.scene_graph_viewer.add_coordinate_system() 
        
    def reset_viewer(self, only_viewer=False):

        self.viewer.items = []
        if not only_viewer:
            self.sample_index_info.setText("")

    def show_sample(self):
        self.reset_viewer()
        self.scene_data_dict = self.dataset.scene_dataset.__getitem__(self.sample_index)
        self.viewer.add_coordinate_system()
        self.scene_graph_viewer.add_coordinate_system()
        self.sample_index_info.setText(f"{self.sample_index}/{self.dataset.__len__}")
        self.show_points()
        self.show_boxes_3d()
        self.show_triples()

    def check_index_overflow(self) -> None:

        if self.sample_index == -1:
            self.sample_index = self.dataset.__len__ - 1

        if self.sample_index >= self.dataset.__len__:
            self.sample_index = 0

    def decrement_index(self) -> None:

        self.sample_index -= 1
        self.check_index_overflow()
        self.show_sample()

    def increment_index(self) -> None:

        self.sample_index += 1
        self.check_index_overflow()
        self.show_sample()

    def generate_layout(self):
        if not hasattr(self, 'lidar_scene'):
            self.lidar_scene = LidarScene(ckpt_file_path='/home/alan/AlanLiang/Projects/AlanLiang/LiDAR-Layout/models/layout/nuscenes/last.ckpt', 
                                          dataset=self.dataset.scene_dataset)
            self.lidar_scene.build_model()
            self.logger.info('Initial Model Successfully!')

        input_data = self.dataset.scene_dataset.collate_fn([self.scene_data_dict])
        gene_boxes = self.lidar_scene.inference_sample(input_data)
        box_names = [self.dataset.scene_dataset.classes_r[idx] for idx in self.scene_data_dict['encoder']['objs'].numpy()]
        box_info = gl.create_boxes(bboxes_3d=gene_boxes, box_texts=box_names)
        self.add_boxes_to_secene_graph(box_info)

    def generate_points_w_layout(self):
        if not hasattr(self, 'lidar_diffusion'):
            self.lidar_diffusion = LiDM_Sampler(ckpt_path='/home/alan/AlanLiang/Projects/AlanLiang/LiDAR-Layout/models/lidm/nuscenes/layout2lidar/last.ckpt')
            self.lidar_diffusion.build_model()
            self.dataset.build_box_lidar_dataset(self.lidar_diffusion.data_config)
            self.logger.info('Initial Model Successfully!')

        batch_dict = self.dataset.box_lidar_dataset.__getitem__(self.sample_index)
        batch = self.dataset.box_lidar_dataset.collate_fn([batch_dict])
        sample_points = self.lidar_diffusion.sample_from_cond(batch)
        self.add_custom_points(sample_points, self.scene_graph_viewer)

    def show_scene_graph(self):
        generate_graph(self.dataset.scene_dataset, self.scene_data_dict)

    def load_nusc_dataset(self):
        data_root = '/home/alan/AlanLiang/Projects/AlanLiang/CentralScene/data/nuscenes'
        self.dataset = NUSC_PCDet(data_root)
        self.logger.info('Load Nuscenes successfully!')
        self.show_sample()
    