import os
import xml.etree.ElementTree as ET
from PIL import Image

def read_anno(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    defect_boxes = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'crazing':
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            defect_boxes.append((xmin, ymin, xmax, ymax))
    
    return defect_boxes