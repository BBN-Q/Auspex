# coding: utf-8

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from functools import partial
import json


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
        self.label.setDefaultTextColor(Qt.white)

        if self.label.boundingRect().topRight().x() > 100:
            self.setRect(0,0,self.label.boundingRect().topRight().x(),30)
        
        # Resize Handle
        self.resize_handle = ResizeHandle(parent=self)
        self.resize_handle.setPos(self.rect().width()-8, self.rect().height()-8)

    def set_title_color(self, color):
        self.title_color = color
        self.title_bar.setBrush(QBrush(color))

    def set_bg_color(self, color):
        self.bg_color = color
        self.setBrush(QBrush(color))

    def add_output(self, connector):
        connector.setParentItem(self)
        connector.setPos(self.rect().width(),30+15*(len(self.outputs)+len(self.inputs)))
        self.setRect(0,0,self.rect().width(),self.rect().height()+15)
        self.outputs[connector.name] = connector

    def add_input(self, connector):
        connector.setParentItem(self)
        connector.setPos(0,30+15*(len(self.inputs)+len(self.outputs)))
        self.setRect(0,0,self.rect().width(),self.rect().height()+15)
        self.inputs[connector.name] = connector

    def add_parameter(self, param):
        param.setParentItem(self)
        self.setRect(0,0,self.rect().width(),self.rect().height()+42)
        param.setPos(0,30+15*(len(self.inputs)+len(self.outputs))+42*len(self.parameters))
        self.parameters[param.name] = param

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for k, v in self.outputs.items():
                v.setX(self.rect().width())
                for w in v.wires_out:
                    w.set_start(v.scenePos())
            for k, v in self.inputs.items():
                for w in v.wires_in:
                    w.set_end(v.scenePos())
            for k, v in self.parameters.items():
                v.set_box_width(self.rect().width())
                for w in v.wires_in:
                    w.set_end(v.scenePos())
        if hasattr(self, 'resize_handle'):
            self.resize_handle.setPos(self.rect().width()-8, self.rect().height()-8)
        if hasattr(self, 'title_bar'):
            self.title_bar.setRect(0,0,self.rect().width(),20)
        return QGraphicsRectItem.itemChange(self, change, value)

    def paint(self, painter, options, widget):
        painter.setBrush(QBrush(self.bg_color))
        painter.setPen(QPen(QColor(200,200,200), 0.75))
        painter.drawRoundedRect(self.rect(), 5.0, 5.0)

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
            r = self.parent.rect()
            r.adjust(0,0,delta.x(), delta.y())
            self.parent.setRect(r)
            self.parent.itemChange(QGraphicsItem.ItemPositionChange, None)
            self.drag_start = event.scenePos()

    def mouseReleaseEvent(self, event):
        self.dragging = False

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

        self.setZValue(5)
        self.set_start(self.start)

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
        print("Unhooking")
        self.end_obj.wires_in.remove(self)
        self.start_obj.wires_out.remove(self)
        self.end_obj = None

    def decide_drop(self, event):
        self.setVisible(False)
        drop_site = self.scene().itemAt(event.scenePos())
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
        self.path.moveTo(self.start)
        halfway_x = self.start.x() + 0.5*(self.end.x()-self.start.x())
        self.path.cubicTo(halfway_x, self.start.y(), halfway_x, self.end.y(), self.end.x(), self.end.y())
        self.setPath(self.path)

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

        # Proxy widget for editing
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setStyleSheet("background-color:transparent;")
        self.proxy_widget = QGraphicsProxyWidget(self)
        self.proxy_widget.setFocusPolicy(Qt.StrongFocus)
        self.proxy_widget.setWidget(self.spin_box)
        self.proxy_widget.setGeometry(QRectF(4,7,92,16))

    def __del__(self):
        # These don't die on their own...
        self.proxy_widget.close()

    def set_box_width(self, width):
        self.proxy_widget.setGeometry(QRectF(4,7,width-6,16))

class Connector(QGraphicsEllipseItem):
    """docstring for Connector"""
    def __init__(self, name, connector_type, parent=None):
        rad = 5
        super(Connector, self).__init__(-rad, -rad, 2*rad, 2*rad, parent=parent)
        self.name = name
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

        self.addItem(self.backdrop)
        self.view = None

        self.menu = QMenu()
        self.sub_menus = []
        self.generate_menus(json_file='nodes.json')
        
        self.menu.addSeparator()
        capture = QAction('Export Experiment', self)
        capture.triggered.connect(self.export_experiment)
        self.menu.addAction(capture)

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
            else:
                self.removeItem(wire)

    def contextMenuEvent(self, event):
        self.last_click = event.scenePos()
        self.menu.exec_(event.screenPos())

    def generate_menus(self, json_file='nodes.json'):
        with open(json_file) as data_file:
            try:
                data = json.load(data_file)
                for cat_name, cat_items in data.items():
                    sm = self.menu.addMenu(cat_name)
                    self.sub_menus.append(sm)
                    for item in cat_items:
                        
                        # Create a QAction and add to the menu
                        action = QAction(item['name'], self)
                        
                        # Create function for dropping node on canvas
                        def create(the_item, cat_name):
                            node = Node(the_item['name'])
                            for op in the_item['outputs']:
                                node.add_output(Connector(op, 'output'))
                            for ip in the_item['inputs']:
                                node.add_input(Connector(ip, 'input'))
                            for p in the_item['parameters']:
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
                            
                        # # Add to class
                        name = "create_"+("".join(item['name'].split()))
                        setattr(self, name, partial(create, item, cat_name))
                        func = getattr(self, name)

                        # Connect trigger for action
                        action.triggered.connect(func)
                        self.sub_menus[-1].addAction(action)

            except:
                print("Error processing JSON file.")

    def export_experiment(self):
        wires = [i for i in self.items() if isinstance(i, Wire)]
        print(wires)

class NodeView(QGraphicsView):
    """docstring for NodeView"""
    def __init__(self, scene):
        super(NodeView, self).__init__(scene)
        self.scene = scene        
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) 
        self.backdrop = None
        self.current_scale = 1.0

    def wheelEvent(self, event):
        change = 0.001*event.delta()
        self.scale(1+change, 1+change)
        self.current_scale *= 1+change

    def mousePressEvent(self, event):
        if (event.button() == Qt.MidButton):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            return super(NodeView, self).mousePressEvent(fake)
        else:
            return super(NodeView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.MidButton):
            self.setDragMode(QGraphicsView.NoDrag)
            fake = QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            return super(NodeView, self).mouseReleaseEvent(fake)
        else:
            return super(NodeView, self).mouseReleaseEvent(event)

if __name__ == "__main__":

    app = QApplication([])

    scene = NodeScene()
    scene.setBackgroundBrush(QBrush(QColor(60,60,60)))

    view = NodeView(scene)
    view.backdrop = scene.backdrop

    view.setRenderHint(QPainter.Antialiasing)
    view.resize(800, 600)
    view.show()

    ps = scene.create_PipelineStart()
    ps.setPos(-300,0)

    view.window().raise_()
    app.exec_()