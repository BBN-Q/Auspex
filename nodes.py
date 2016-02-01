# coding: utf-8

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from functools import partial
import json

rad = 5

class Node(QGraphicsRectItem):
    """docstring for Node"""
    def __init__(self, name, scene):
        super(Node, self).__init__()
        self.name = name
        self.scene = scene
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        self.outputs = {}
        self.inputs = {}
        self.parameters = {}

        self.setRect(0,0,100,30)
        self.setBrush(QBrush(QColor(100,100,100)))
        self.setPen(QPen(QColor(200,200,200), 0.75))

        # Title bar
        self.title_bar = QGraphicsRectItem(parent=self)
        self.title_bar.setRect(0,0,100,20)
        self.title_bar.setBrush(QBrush(QColor(80,80,100)))
        self.title_bar.setPen(QPen(QColor(200,200,200), 0.75))

        self.label = QGraphicsTextItem(self.name, parent=self)
        self.label.setDefaultTextColor(Qt.white)
        # self.label.setTextInteractionFlags(Qt.TextEditable)

    def add_output(self, port):
        port.connector_type = 'output'
        port.setBrush(Qt.blue)
        port.setParentItem(self)
        port.setPos(100,30+15*(len(self.outputs)+len(self.inputs)))
        label = QGraphicsTextItem(port.name, parent=self)
        top_right = label.boundingRect().topRight()
        label.setPos(95-top_right.x(),20+15*(len(self.outputs)+len(self.inputs)))
        self.setRect(0,0,100,self.rect().height()+15)
        label.setDefaultTextColor(Qt.black)
        self.outputs[port.name] = port

    def add_input(self, port):
        port.connector_type = 'input'
        port.setBrush(Qt.green)
        port.setParentItem(self)
        print(len(self.outputs))
        port.setPos(000,30+15*(len(self.inputs)+len(self.outputs)))
        label = QGraphicsTextItem(port.name, parent=self)
        w = label.textWidth()
        label.setPos(5,20+15*(len(self.inputs)+len(self.outputs)))
        self.setRect(0,0,100,self.rect().height()+15)
        label.setDefaultTextColor(Qt.black)
        self.inputs[port.name] = port

    def add_parameter(self, param):
        # label = QGraphicsTextItem(param.name, parent=self)
        # w = label.textWidth()
        # label.setPos(5,20+15*(len(self.inputs)+len(self.outputs))+35*len(self.parameters))
        # new_param = Parameter(param)
        param.setParentItem(self)
        self.setRect(0,0,100,self.rect().height()+42)
        param.setPos(0,30+15*(len(self.inputs)+len(self.outputs))+42*len(self.parameters))
        # label.setDefaultTextColor(Qt.black)
        self.parameters[param.name] = param

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for k, v in self.outputs.items():
                for w in v.wires_out:
                    w.set_start(v.scenePos())
            for k, v in self.inputs.items():
                for w in v.wires_in:
                    w.set_end(v.scenePos())
        return QGraphicsRectItem.itemChange(self, change, value)

    def paint(self, painter, options, widget):
        painter.setBrush(QBrush(QColor(100,100,100)))
        painter.setPen(QPen(QColor(200,200,200), 0.75))
        painter.drawRoundedRect(self.rect(), 5.0, 5.0)

class Wire(QGraphicsPathItem):
    """docstring for Wire"""
    def __init__(self, start_obj, scene):
        self.path = QPainterPath()
        super(Wire, self).__init__(self.path)

        self.scene     = scene
        self.start     = start_obj.scenePos()
        self.end       = self.start
        self.start_obj = start_obj
        self.end_obj   = None
        self.make_path()

        self.setZValue(5)

        self.setPen(QPen(QColor(200,200,200), 0.75))

        # Add endpoint circle
        rad = 5
        self.end_image = QGraphicsEllipseItem(-rad, -rad, 2*rad, 2*rad, parent=self)
        self.end_image.setBrush(Qt.white)
        self.end_image.setPos(self.start)
        self.end_image.setZValue(10)

        # Setup behavior for unlinking the end of the wire, monkeypatch!
        self.end_image.mousePressEvent = lambda e: self.unhook(e)
        self.end_image.mouseMoveEvent = lambda e: self.set_end(e.scenePos())
        self.end_image.mouseReleaseEvent = lambda e: self.decide_drop(e)

    def unhook(self, event):
        self.end_obj.wires_in.remove(self)
        self.start_obj.wires_out.remove(self)

    def decide_drop(self, event):
        self.setVisible(False)
        drop_site = self.scene.itemAt(event.scenePos())
        if isinstance(drop_site, Connector):
            if drop_site.connector_type == 'input':
                print("Good drop!")
                self.set_end(drop_site.scenePos())
                self.setVisible(True)
                self.end_obj = drop_site
                drop_site.wires_in.append(self)
                self.start_obj.wires_out.append(self)
            else:
                print("Can't connect to output")
        else:
            print("Bad drop!")

    def set_start(self, start):
        self.start = start
        self.make_path()

    def set_end(self, end):
        self.end = end
        self.make_path()
        self.end_image.setPos(end)

    def make_path(self):
        self.path = QPainterPath()
        self.path.moveTo(self.start)
        halfway_x = self.start.x() + 0.5*(self.end.x()-self.start.x())
        self.path.cubicTo(halfway_x, self.start.y(), halfway_x, self.end.y(), self.end.x(), self.end.y())
        self.setPath(self.path)
        

class Parameter(QGraphicsEllipseItem):
    """docstring for Parameter"""
    def __init__(self, name, scene):
        self.scene = scene
        rad = 5
        super(Parameter, self).__init__(-rad, -rad, 2*rad, 2*rad)
        self.name = name
        self.setBrush(QBrush(QColor(200,200,240)))
        self.setPen(Qt.blue)
        self.setZValue(1)

        # Text label and area
        self.label = QGraphicsTextItem(self.name, parent=self)
        self.label.setPos(5,-10)
        # self.text_area = QGraphicsTextItem("0", parent=self)
        # self.text_area.setPos(5,5)
        # self.text_area.setTextInteractionFlags(Qt.TextEditable)
        # self.text_area.setZValue(10)

        # Background
        # self.text_area_bg = QGraphicsRectItem(4,7,92,16, parent=self)
        # self.text_area_bg.setZValue(9)
        # self.text_area_bg.setBrush(QBrush(QColor(200,200,200)))

        # Proxy widget for editing
        self.spin_box = QDoubleSpinBox()
        self.proxy_widget = QGraphicsProxyWidget(self)
        self.proxy_widget.setFocusPolicy(Qt.StrongFocus)
        self.proxy_widget.setWidget(self.spin_box)
        self.proxy_widget.setGeometry(QRectF(4,7,92,16))

class Connector(QGraphicsEllipseItem):
    """docstring for Connector"""
    def __init__(self, name, scene, connector_type='output'):
        self.scene = scene
        rad = 5
        super(Connector, self).__init__(-rad, -rad, 2*rad, 2*rad)
        self.name = name
        self.setBrush(Qt.white)
        self.setPen(Qt.blue)

        self.setZValue(1)

        self.temp_wire = None
        self.wires_in  = []
        self.wires_out = []

        self.connector_type = connector_type

    def mousePressEvent(self, event):
        self.temp_wire = Wire(self, self.scene)
        self.scene.addItem(self.temp_wire)

    def mouseMoveEvent(self, event):
        if self.temp_wire is not None:
            self.temp_wire.set_end(event.scenePos())

    def mouseReleaseEvent(self, event):
        self.temp_wire.decide_drop(event)
   

class NodeCanvas(QGraphicsScene):
    """docstring for NodeCanvas"""
    def __init__(self):
        super(NodeCanvas, self).__init__()
        self.menu = QMenu()
        self.sub_menus = []
        self.generate_menus()
        self.menu.addSeparator()
        capture = QAction('Export Experiment', self)
        capture.triggered.connect(self.export_experiment)
        self.menu.addAction(capture)

    def contextMenuEvent(self, event):
        self.last_click = event.scenePos()
        self.menu.exec_(event.screenPos())

    def generate_menus(self, json_file='nodes.json'):
        with open(json_file) as data_file:
            try:
                data = json.load(data_file)
                for cat_name, cat_items in data.items():
                    print("Found node type {:s}".format(cat_name))
                    sm = self.menu.addMenu(cat_name)
                    self.sub_menus.append(sm)
                    for item in cat_items:
                        print("Adding node {}".format(item['name']))
                        
                        # Create a QAction and add to the menu
                        action = QAction(item['name'], self)
                        
                        # Create function for dropping node on canvas
                        def create(the_item):
                            print("I am {}".format(self))
                            node = Node(the_item['name'], self)
                            for op in the_item['outputs']:
                                node.add_output(Connector(op, self))
                            for ip in the_item['inputs']:
                                node.add_input(Connector(ip, self))
                            for p in the_item['parameters']:
                                node.add_parameter(Parameter(p, self))
                            node.setPos(self.last_click)
                            self.addItem(node)
                            
                        # # Add to class
                        name = "create_"+("".join(item['name'].split()))
                        print("Adding {} method to class.".format(name))

                        setattr(self, name, create)
                        func = getattr(self, name)
                        print("Getattr yields {}.".format(getattr(self, name)))

                        # Connect trigger for action
                        action.triggered.connect(partial(func, item))
                        self.sub_menus[-1].addAction(action)

            except:
                print("Error processing JSON file.")

    def export_experiment(self):
        wires = [i for i in self.items() if isinstance(i, Wire)]
        print(wires)


if __name__ == "__main__":

    app = QApplication([])

    scene = NodeCanvas()
    scene.setBackgroundBrush(QBrush(QColor(60,60,60)))

    node = Node("Convolve", scene)
    node.add_output(Connector("Output", scene))
    node.add_input(Connector("Waveform", scene))
    node.add_input(Connector("Kernel", scene))
    node.setPos(200,0)
    scene.addItem(node)

    node = Node("Decimate", scene)
    node.add_output(Connector("Output", scene))
    node.add_input(Connector("Waveform", scene))
    node.add_input(Connector("Factor", scene))
    node.setPos(0,0)
    scene.addItem(node)

    node = Node("Data Taker", scene)
    node.add_output(Connector("Output", scene))
    node.setPos(-200,0)
    scene.addItem(node)

    view = QGraphicsView(scene)
    view.setRenderHint(QPainter.Antialiasing)
    view.resize(800, 600)
    view.show()

    # view.window().activateWindow()
    view.window().raise_()
    app.exec_()