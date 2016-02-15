# coding: utf-8

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from functools import partial
import json
import sys
import glob
import os.path

class Node(QGraphicsRectItem):
    """docstring for Node"""
    def __init__(self, name, parent=None):
        super(Node, self).__init__(parent=parent)
        self.name = name
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        self.outputs = {}
        self.inputs = {}
        self.parameters = {}

        self.bg_color = QColor(240,240,240)
        self.setRect(0,0,100,30)
        self.setBrush(QBrush(self.bg_color))
        self.setPen(QPen(QColor(200,200,200), 0.75))

        # Title bar
        self.title_bar = QGraphicsRectItem(parent=self)
        self.title_bar.setRect(0,0,100,20)
        self.title_color = QColor(80,80,100)
        self.title_bar.setBrush(QBrush(self.title_color))
        self.title_bar.setPen(QPen(QColor(200,200,200), 0.75))
        self.label = QGraphicsTextItem(self.name, parent=self)
        self.label.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.label.setDefaultTextColor(Qt.white)

        if self.label.boundingRect().topRight().x() > 80:
            self.min_width = self.label.boundingRect().topRight().x()+20
            self.setRect(0,0,self.label.boundingRect().topRight().x()+20,30)
        else:
            self.min_width = 80.0

        self.min_height = 30

        # Resize Handle
        self.resize_handle = ResizeHandle(parent=self)
        self.resize_handle.setPos(self.rect().width()-8, self.rect().height()-8)

        # Remove box
        self.remove_box = RemoveBox(parent=self)
        self.remove_box.setPos(self.rect().width()-13, 5)

        # Disable box
        self.disable_box = None

        # Make sure things are properly sized
        self.itemResize(QPointF(0.0,0.0))

    def set_title_color(self, color):
        self.title_color = color
        self.title_bar.setBrush(QBrush(color))

    def set_bg_color(self, color):
        self.bg_color = color
        self.setBrush(QBrush(color))

    def add_output(self, connector):
        connector.setParentItem(self)
        connector.parent = self
        connector.setPos(self.rect().width(),30+15*(len(self.outputs)+len(self.inputs)))
        self.setRect(0,0,self.rect().width(),self.rect().height()+15)
        self.min_height += 15
        self.outputs[connector.name] = connector
        self.itemResize(QPointF(0.0,0.0))

    def add_input(self, connector):
        connector.setParentItem(self)
        connector.parent = self
        connector.setPos(0,30+15*(len(self.inputs)+len(self.outputs)))
        self.setRect(0,0,self.rect().width(),self.rect().height()+15)
        self.min_height += 15
        self.inputs[connector.name] = connector
        self.itemResize(QPointF(0.0,0.0))

    def add_parameter(self, param):
        param.setParentItem(self)
        param.parent = self
        self.setRect(0,0,self.rect().width(),self.rect().height()+42)
        self.min_height += 42
        param.setPos(0,30+15*(len(self.inputs)+len(self.outputs))+42*len(self.parameters))
        self.parameters[param.name] = param
        self.itemResize(QPointF(0.0,0.0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for k, v in self.outputs.items():
                v.setX(self.rect().width())
                for w in v.wires_out:
                    w.set_start(v.pos()+value)
            for k, v in self.inputs.items():
                for w in v.wires_in:
                    w.set_end(v.pos()+value)
            for k, v in self.parameters.items():
                for w in v.wires_in:
                    w.set_end(v.pos()+value)
        return QGraphicsRectItem.itemChange(self, change, value)

    def itemResize(self, delta):
        # Keep track of actual change
        actual_delta = QPointF(0,0)

        r = self.rect()
        if r.width()+delta.x() >= self.min_width:
            r.adjust(0, 0, delta.x(), 0)
            actual_delta.setX(delta.x())

        if r.height()+delta.y() >= self.min_height:
            r.adjust(0, 0, 0, delta.y())
            actual_delta.setY(delta.y())

        self.setRect(r)
        delta.setY(0.0)

        if hasattr(self, 'resize_handle'):
            self.resize_handle.setPos(self.rect().width()-8, self.rect().height()-8)
        if hasattr(self, 'title_bar'):
            self.title_bar.setRect(0,0,self.rect().width(),20)
        if hasattr(self, 'remove_box'):
            self.remove_box.setPos(self.rect().width()-13, 5)

        conn_delta = actual_delta.toPoint()
        conn_delta.setY(0.0)

        # Move the outputs
        for k, v in self.outputs.items():
            v.setX(self.rect().width())
            for w in v.wires_out:
                w.set_start(v.scenePos()+conn_delta)

        # Resize the parameters
        for k, v in self.parameters.items():
            v.set_box_width(self.rect().width())

        return actual_delta

    def disconnect(self):
        for k, v in self.outputs.items():
            for w in v.wires_out:
                w.start_obj = None
        for k, v in self.inputs.items():
            for w in v.wires_in:
                w.end_obj = None
        for k, v in self.parameters.items():
            for w in v.wires_in:
                w.end_obj = None
        self.scene().clear_wires(only_clear_orphaned=True)

    def paint(self, painter, options, widget):
        painter.setBrush(QBrush(self.bg_color))
        painter.setPen(QPen(QColor(200,200,200), 0.75))
        painter.drawRoundedRect(self.rect(), 5.0, 5.0)

    def dict_repr(self):
        dat = {}
        dat['label'] = self.label.toPlainText()
        dat['name'] = self.name
        params = {}
        for k, v in self.parameters.items():
            params[k] = v.value()
        dat['params'] = params
        dat['pos'] = [self.scenePos().x(), self.scenePos().y()]
        return dat

class ResizeHandle(QGraphicsRectItem):
    """docstring for ResizeHandle"""
    def __init__(self, parent=None):
        super(ResizeHandle, self).__init__()
        self.dragging = False
        self.parent = parent
        self.drag_start = None
        self.setParentItem(parent)
        self.setRect(0,0,5,5)
        self.setBrush(QColor(20,20,20))

    def mousePressEvent(self, event):
        self.dragging = True
        self.drag_start = event.scenePos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.scenePos() - self.drag_start
            actual_delta = self.parent.itemResize(delta)
            self.drag_start = self.drag_start + actual_delta

    def mouseReleaseEvent(self, event):
        self.dragging = False

class RemoveBox(QGraphicsRectItem):
    """docstring for RemoveBox"""
    def __init__(self, parent=None):
        super(RemoveBox, self).__init__(parent=parent)
        self.parent = parent
        self.setRect(0,0,10,10)
        self.setBrush(QColor(60,60,60))
        self.setPen(QPen(Qt.black, 1.0))
        self.close_started = False
    
    def mousePressEvent(self, event):
        self.close_started = True

    def mouseReleaseEvent(self, event):
        # if event.pos()
        self.parent.disconnect()
        self.scene().removeItem(self.parent)

class Wire(QGraphicsPathItem):
    """docstring for Wire"""
    def __init__(self, start_obj, parent=None):
        self.path = QPainterPath()
        super(Wire, self).__init__(self.path, parent=parent)

        self.parent    = parent
        self.start     = start_obj.scenePos()
        self.end       = self.start
        self.start_obj = start_obj
        self.end_obj   = None
        self.make_path()

        self.setZValue(0)
        self.set_start(self.start)

        # Add endpoint circle
        rad = 5
        self.end_image = QGraphicsEllipseItem(-rad, -rad, 2*rad, 2*rad, parent=self)
        self.end_image.setBrush(Qt.white)
        self.end_image.setPos(self.start)
        # self.end_image.setZValue(10)

        # Setup behavior for unlinking the end of the wire, monkeypatch!
        self.end_image.mousePressEvent = lambda e: self.unhook(e)
        self.end_image.mouseMoveEvent = lambda e: self.set_end(e.scenePos())
        self.end_image.mouseReleaseEvent = lambda e: self.decide_drop(e)

    def unhook(self, event):
        print("Unhooking")
        self.end_obj.wires_in.remove(self)
        self.start_obj.wires_out.remove(self)
        self.end_obj = None

    def decide_drop(self, event):
        self.setVisible(False)
        drop_site = self.scene().itemAt(event.scenePos(), QTransform())
        if isinstance(drop_site, Connector):
            if drop_site.connector_type == 'input':
                print("Connecting to data-flow connector")
                self.set_end(drop_site.scenePos())
                self.end_obj = drop_site
                drop_site.wires_in.append(self)
                self.start_obj.wires_out.append(self)
            else:
                print("Can't connect to output")
        elif isinstance(drop_site, Parameter):
            print("Connecting to parameter connector")
            self.set_end(drop_site.scenePos())
            self.end_obj = drop_site
            drop_site.wires_in.append(self)
            self.start_obj.wires_out.append(self)
        else:
            print("Bad drop!")

        self.setVisible(True)
        self.scene().clear_wires(only_clear_orphaned=True)

    def set_start(self, start):
        self.start = start
        self.make_path()

    def set_end(self, end):
        self.end = end
        self.make_path()
        self.end_image.setPos(end)

    def make_path(self):
        self.path = QPainterPath()
        self.path.moveTo(self.start.x()+5, self.start.y()+1)
        halfway_x = self.start.x() + 0.5*(self.end.x()-self.start.x())
        self.path.cubicTo(halfway_x, self.start.y(), halfway_x, self.end.y()+3, self.end.x(), self.end.y()+3)
        self.path.lineTo(self.end.x(), self.end.y()-3)
        self.path.cubicTo(halfway_x, self.end.y(), halfway_x, self.start.y()-3, self.start.x()+5, self.start.y()-1)
        self.path.lineTo(self.start.x()+5, self.start.y()+1)
        self.setPath(self.path)

        linearGradient = QLinearGradient(self.start, self.end)
        linearGradient.setColorAt(0, QColor(128, 128, 128))
        linearGradient.setColorAt(1.0, Qt.white)
        self.setBrush(QBrush(linearGradient))
        self.setPen(QPen(QColor(128, 128, 128), 0.25))

    def dict_repr(self):
        dat = {}
        dat['start'] = {'node': self.start_obj.parent.label.toPlainText(), 'connector_name': self.start_obj.name}
        dat['end'] = {'node': self.end_obj.parent.label.toPlainText(), 'connector_name': self.end_obj.name}
        return dat

class Parameter(QGraphicsEllipseItem):
    """docstring for Parameter"""
    def __init__(self, name, parent=None):
        self.name = name
        rad = 5
        super(Parameter, self).__init__(-rad, -rad, 2*rad, 2*rad, parent=parent)
        
        self.setBrush(QBrush(QColor(200,200,240)))
        self.setPen(Qt.black)
        self.setZValue(1)

        self.temp_wire = None
        self.wires_in  = []
        self.wires_out = []

        # Text label and area
        self.label = QGraphicsTextItem(self.name, parent=self)
        self.label.setDefaultTextColor(Qt.black)
        self.label.setPos(5,-10)

        # SliderBox
        self.spin_box = SliderBox(parent=self)

    def set_box_width(self, width):
        self.spin_box.set_box_width(width)

    def value(self):
        return self.spin_box.value()

    def set_value(self, value):
        self.spin_box.set_value(float(value))

class SliderBox(QGraphicsRectItem):
    """docstring for SliderBox"""
    def __init__(self, parent):
        super(SliderBox, self).__init__(parent=parent)
        self.parent = parent
        self.dragging = False

        self._value = 5.0
        self.min_value = 0.0
        self.max_value = 10.0
        self.increment = 0.5

        self.height = 14
        self.rect_radius = 7.0
        self.control_distance = 0.55228*self.rect_radius
        self.setRect(3,15,94,self.height)

        self.label = QGraphicsTextItem(str(self._value), parent=self)
        label_width = self.label.boundingRect().topRight().x()
        self.label.setPos(3+0.5*self.rect().width()-0.5*label_width,15-5)

    def paint(self, painter, options, widget):
        # Background object is a rounded rectangle
        painter.RenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(220,220,220)))
        painter.setPen(QPen(QColor(200,200,200), 0.75))
        painter.drawRoundedRect(self.rect(), self.rect_radius, self.rect_radius)

        # Draw the bar using a round capped line
        painter.setPen(QPen(QColor(160,200,220), self.height, cap=Qt.RoundCap))
        path = QPainterPath()
        path.moveTo(3+self.rect_radius, 15 + 0.5*self.height)
        fill_size = (self.rect().width()-2*self.rect_radius)*(self._value-self.min_value)/self.max_value
        path.lineTo(3+self.rect_radius+fill_size, 7.5 + 0.5+self.height)
        painter.drawPath(path)

    def set_value(self, value):
        if value <= self.max_value and value >= self.min_value:
            self._value = value
            self.label.setPlainText(str(value))
            self.update()

    def value(self):
        return self._value

    def set_box_width(self, width):
        self.setRect(3,15, width-6, self.height)
        label_width = self.label.boundingRect().topRight().x()
        self.label.setPos(3+0.5*self.rect().width()-0.5*label_width,15-5)

    def mousePressEvent(self, event):
        self.dragging = True
        self.original_value = self._value
        self.drag_start = event.scenePos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.scenePos() - self.drag_start
            value_change = self.increment*int(delta.x()/10.0)
            self.set_value(self.original_value + value_change)

    def mouseReleaseEvent(self, event):
        self.dragging = False
    
class Connector(QGraphicsEllipseItem):
    """docstring for Connector"""
    def __init__(self, name, connector_type, parent=None):
        rad = 5
        super(Connector, self).__init__(-rad, -rad, 2*rad, 2*rad, parent=parent)
        self.name = name
        self.parent = parent
        self.connector_type = connector_type
        self.setZValue(1)

        self.temp_wire = None
        self.wires_in  = []
        self.wires_out = []

        # Text label and area
        self.label = QGraphicsTextItem(self.name, parent=self)
        self.label.setDefaultTextColor(Qt.black)

        if self.connector_type == 'output':
            self.label.setPos(-5-self.label.boundingRect().topRight().x(),-10)      
            self.setBrush(Qt.white)
            self.setPen(Qt.blue)
        else:
            self.label.setPos(5,-10)      
            self.setBrush(Qt.white)
            self.setPen(Qt.blue)

    def mousePressEvent(self, event):
        self.temp_wire = Wire(self)
        self.scene().addItem(self.temp_wire)

    def mouseMoveEvent(self, event):
        if self.temp_wire is not None:
            self.temp_wire.set_end(event.scenePos())

    def mouseReleaseEvent(self, event):
        self.temp_wire.decide_drop(event)
        self.scene().clear_wires(only_clear_orphaned=True)

class NodeScene(QGraphicsScene):
    """docstring for NodeScene"""
    def __init__(self):
        super(NodeScene, self).__init__()
        self.backdrop = QGraphicsRectItem()
        self.backdrop.setRect(-10000,-10000,20000,20000)
        self.backdrop.setZValue(-100)
        self.setBackgroundBrush(QBrush(QColor(60,60,60)))

        self.addItem(self.backdrop)
        self.view = None

        self.menu = QMenu()
        self.sub_menus = {}
        self.generate_menus()

        self.menu.addSeparator()
        clear_wires = QAction('Clear Wires', self)
        clear_wires.triggered.connect(self.clear_wires)
        self.menu.addAction(clear_wires)

        self.last_click = self.backdrop.pos()

    def clear_wires(self, only_clear_orphaned=False):
        wires = [i for i in self.items() if isinstance(i, Wire)]
        for wire in wires:
            if only_clear_orphaned:
                if wire.end_obj is None:
                    self.removeItem(wire)
                elif wire.start_obj is None:
                    self.removeItem(wire)
            else:
                self.removeItem(wire)

    def contextMenuEvent(self, event):
        self.last_click = event.scenePos()
        self.menu.exec_(event.screenPos())

    def generate_menus(self):
        def parse_node_file(filename):
            with open(filename) as data_file:
                cat  = os.path.basename(os.path.dirname(filename))
                data = json.load(data_file)
                
                # Create a QAction and add to the menu
                action = QAction(data['name'], self)
                
                # Create function for dropping node on canvas
                def create(the_data, cat_name):
                    node = Node(the_data['name'])
                    for op in the_data['outputs']:
                        node.add_output(Connector(op, 'output'))
                    for ip in the_data['inputs']:
                        node.add_input(Connector(ip, 'input'))
                    for p in the_data['parameters']:
                        node.add_parameter(Parameter(p))
                    # Custom coloring
                    if cat_name == "Inputs":
                        node.set_title_color(QColor(80,100,70))
                    elif cat_name == "Outputs":
                        node.set_title_color(QColor(120,70,70))

                    node.setPos(self.backdrop.mapFromParent(self.last_click))
                    node.setPos(self.last_click)
                    self.addItem(node)
                    return node
                    
                # Add to class
                name = "create_"+("".join(data['name'].split()))
                setattr(self, name, partial(create, data, cat))
                func = getattr(self, name)

                # Connect trigger for action
                action.triggered.connect(func)
                self.sub_menus[cat].addAction(action)

        node_files = sorted(glob.glob('nodes/*/*.json'))
        categories = set([os.path.basename(os.path.dirname(nf)) for nf in node_files])
        
        for cat in categories:
            sm = self.menu.addMenu(cat)
            self.sub_menus[cat] = sm

        for nf in node_files:
            parse_node_file(nf)

    def load(self, filename):
        with open(filename, 'r') as df:

            # Clear scene
            nodes = [i for i in self.items() if isinstance(i, Node)]
            wires = [i for i in self.items() if isinstance(i, Wire)]
            for o in nodes+wires:
                self.removeItem(o)

            data = json.load(df)
            nodes = data['nodes']
            wires = data['wires']

            new_nodes = {} # Keep track of nodes we create

            for n in nodes:
                create_node_func_name = "create_"+("".join(n['name'].split()))
                if hasattr(self, create_node_func_name):
                    if n['label'] not in new_nodes.keys():
                        new_node = getattr(self, create_node_func_name)()
                        new_node.setPos(float(n['pos'][0]), float(n['pos'][1]))
                        for k, v in n['params'].items():
                            new_node.parameters[k].set_value(v)
                        new_node.label.setPlainText(n['label'])
                        new_nodes[n['label']] = new_node
                    else:
                        print("Node cannot be named {}, label already in use".format(n['label']))
                else:
                    print("Could not load node of type {}, please check nodes directory.".format(n['name']))

            for w in wires:
                # Instantiate a little later
                new_wire = None

                start_node_name = w['start']['node']
                end_node_name   = w['end']['node']
                start_conn_name = w['start']['connector_name']
                end_conn_name   = w['end']['connector_name']

                start_node = new_nodes[start_node_name]
                end_node   = new_nodes[end_node_name]

                # Find our beginning connector
                if start_conn_name in start_node.outputs.keys():
                    new_wire = Wire(start_node.outputs[start_conn_name])
                    self.addItem(new_wire)
                    new_wire.set_start(start_node.outputs[start_conn_name].scenePos())
                    start_node.outputs[start_conn_name].wires_out.append(new_wire)
                    
                    # Find our end connector
                    if end_conn_name in end_node.inputs.keys():
                        new_wire.end_obj = end_node.inputs[end_conn_name]
                        new_wire.set_end(end_node.inputs[end_conn_name].scenePos())
                        end_node.inputs[end_conn_name].wires_in.append(new_wire)
                    elif end_conn_name in end_node.parameters.keys():
                        new_wire.end_obj = end_node.parameters[end_conn_name]
                        new_wire.set_end(end_node.parameters[end_conn_name].scenePos())
                        end_node.parameters[end_conn_name].wires_in.append(new_wire)
                    else:
                        print("Could not find input {} on node {}.".format(end_conn_name, end_node_name))
                else:
                    print("Could not find output {} on node {}.".format(start_conn_name, start_node_name))

    def save(self, filename):
        with open(filename, 'w') as df:
            nodes = [i for i in self.items() if isinstance(i, Node)]
            wires = [i for i in self.items() if isinstance(i, Wire)]
            
            data = {}
            data['nodes'] = [n.dict_repr() for n in nodes]
            data['wires'] = [n.dict_repr() for n in wires]
            json.dump(data, df, sort_keys=True, indent=4, separators=(',', ': '))


class NodeView(QGraphicsView):
    """docstring for NodeView"""
    def __init__(self, scene):
        super(NodeView, self).__init__(scene)
        self.scene = scene        
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) 
        self.setRenderHint(QPainter.Antialiasing)
        self.current_scale = 1.0

    def wheelEvent(self, event):
        change = 0.001*event.angleDelta().y()/2.0
        self.scale(1+change, 1+change)
        self.current_scale *= 1+change

    def mousePressEvent(self, event):
        if (event.button() == Qt.MidButton) or (event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            return super(NodeView, self).mousePressEvent(fake)
        else:
            return super(NodeView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.MidButton) or (event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier):
            self.setDragMode(QGraphicsView.NoDrag)
            fake = QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            return super(NodeView, self).mouseReleaseEvent(fake)
        else:
            return super(NodeView, self).mouseReleaseEvent(event)

class NodeWindow(QMainWindow):
    """docstring for NodeWindow"""
    def __init__(self, parent=None):
        super(NodeWindow, self).__init__(parent=parent)
        self.setWindowTitle("Nodes")
        self.setGeometry(50,50,800,600)
        
        # Setup graphics
        self.scene = NodeScene()
        self.view  = NodeView(self.scene)

        # Setup menu
        self.statusBar()

        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QApplication.instance().quit)

        saveAction = QAction(QIcon('save.png'), '&Save', self)        
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save')
        saveAction.triggered.connect(self.save)

        openAction = QAction(QIcon('open.png'), '&Open', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open')
        openAction.triggered.connect(self.load)


        fileMenu = self.menuBar().addMenu('&File')
        helpMenu = self.menuBar().addMenu('&Help')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(openAction)

        # Setup layout
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.view)
        self.hbox.setContentsMargins(0,0,0,0)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.hbox)

        self.setCentralWidget(self.main_widget)

        # Create the pipeline start node if possible
        if hasattr(self.scene, 'create_PipelineStart'):
            ps = self.scene.create_PipelineStart()
            ps.setPos(-300,0)

    def load(self):
        path = os.path.dirname(os.path.realpath(__file__))
        fn = QFileDialog.getOpenFileName(self, 'Load Graph', path)
        if fn:
            self.scene.load(fn[0])

    def save(self):
        path = os.path.dirname(os.path.realpath(__file__))
        fn = QFileDialog.getSaveFileName(self, 'Save Graph', path)
        if fn:
            self.scene.save(fn[0])

    def cleanup(self):
        # Have to manually close proxy widgets
        nodes = [i for i in self.scene.items() if isinstance(i, Node)]
        for n in nodes:
            for k, v in n.parameters.items():
                # v.proxy_widget.close()
                pass

if __name__ == "__main__":

    app = QApplication([])
    window = NodeWindow()
    app.aboutToQuit.connect(window.cleanup)
    window.show()

    sys.exit(app.exec_())