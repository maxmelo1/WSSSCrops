import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from shutil import copy
import xml.etree.ElementTree as ET
from dict2xml import dict2xml
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import json



DSN = 'crops_dataset'
CLASSES = {'Background': 0, 'Citrus': 1, 'Corn': 2}
PALETTE = [0,0,0, 128,0,0, 0, 128,0 ]
PATH_CITRUS = './Dataset_citrus/'
PATH_CORN = './Dataset_corn/'
SETS = ['train', 'val', 'test']


def get_class_id(patch, cls):
    return 0 if np.sum(patch)==0 else cls


class PascalXML:
    last_id = 0
    def __init__(self, args) -> None:
        self.args = args

        
    def setFile(self, image_path: str, cls=0):
        self.image = Image.open(image_path)
        self.gt = Image.open(image_path.replace('rgb', 'labels'))
        samgt = np.array(Image.open(image_path.replace('rgb', 'labels_instance')))

        self.samgt = Image.fromarray(np.where(samgt > 0, cls, samgt))
        self.samgt.putpalette(PALETTE)

        self.label = get_class_id(self.gt, cls)
    
    def writeXML(self):
        count = 0 if self.label==0 else np.unique(self.gt, return_counts=True)[1][1]

        data = {
                
            'folder': DSN,
            'filename': str(PascalXML.last_id)+'.png',
            'size': {
                'width': self.image.size[0],
                'height': self.image.size[1],
                'depth': 3
            },
            'segmented': 1,
                
        }

        if self.label > 0:
            data['object'] = {
                'name': list(CLASSES.keys())[list(CLASSES.values()).index(self.label)],
                'pose': 'Unspecified',
                'truncated': 0,
                'difficult': 0,
                'count': count
            }

        # xml = dict2xml(data)
        xml = dicttoxml(data, custom_root='annotation', attr_type=False)

        # import xml.dom.minidom

        # print(xml)
        xml_decode = xml.decode()
        dom = parseString(xml_decode)
        pretty_xml_as_string = dom.toprettyxml()
 
        xmlfile = open(os.path.join(args.output_path, DSN, 'Annotations', str(PascalXML.last_id)+'.xml' ), "w")
        xmlfile.write(pretty_xml_as_string)
        xmlfile.close()
        

    def writeFiles(self):
        self.image.save(os.path.join(args.output_path, DSN, 'JPEGImages', str(PascalXML.last_id)+'.png'))
        self.samgt.save(os.path.join(args.output_path, DSN, 'SAMSegmentationClass', str(PascalXML.last_id)+'.png'))
        self.gt.save(os.path.join(args.output_path, DSN, 'Labels', str(PascalXML.last_id)+'.png'))

        self.writeXML()


        PascalXML.last_id += 1


def copy_files(files, writer):
    citrus = 0
    corn = 0
    names = []
    for el in files[0]:
        writer.setFile(el, CLASSES['Citrus'])
        names.append(PascalXML.last_id)
        writer.writeFiles()

        if writer.label>0:
            citrus += 1
    
    for el in files[1]:
        writer.setFile(el, CLASSES['Corn'])
        names.append(PascalXML.last_id)
        writer.writeFiles()

        if writer.label>0:
            corn += 1
    
    return citrus, corn, names

def main(args):

    print(f'The new dataset will be stored at: {args.output_path}')
    if not os.path.exists(os.path.join(args.output_path, DSN)):
        os.makedirs(os.path.join(args.output_path, DSN, 'data'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, DSN, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, DSN, 'Labels'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, DSN, 'SAMSegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, DSN, 'Annotations'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, DSN, 'ImageSets/Segmentation'), exist_ok=True)

    files_citrus_train = [os.path.join(PATH_CITRUS, 'train', 'rgb', el) for el in  os.listdir( os.path.join(PATH_CITRUS, 'train', 'rgb'))]
    files_citrus_val = [os.path.join(PATH_CITRUS, 'val', 'rgb', el) for el in  os.listdir( os.path.join(PATH_CITRUS, 'val', 'rgb'))]
    files_citrus_test = [os.path.join(PATH_CITRUS, 'test', 'rgb', el) for el in  os.listdir( os.path.join(PATH_CITRUS, 'test', 'rgb'))]

    files_citrus = files_citrus_train + files_citrus_val + files_citrus_test

    files_corn_train = [os.path.join(PATH_CORN, 'train', 'rgb', el) for el in  os.listdir( os.path.join(PATH_CORN, 'train', 'rgb'))]
    files_corn_val = [os.path.join(PATH_CORN, 'val', 'rgb', el) for el in  os.listdir( os.path.join(PATH_CORN, 'val', 'rgb'))]
    files_corn_test = [os.path.join(PATH_CORN, 'test', 'rgb', el) for el in  os.listdir( os.path.join(PATH_CORN, 'test', 'rgb'))]

    files_corn = files_corn_train + files_corn_val + files_corn_test

    files = files_citrus + files_corn

    print(f'Total of samples found: {len(files)}')

    
    pascalWriter = PascalXML(args)
    
    summary = {
        "classes": 2,
        "class_names": 
            [
                "Citrus", "Corn"
            ],
        "class_dic": 
            {
                "Citrus": 0,
                "Corn": 1,

            },
        "color_dict": 
            {
                "background": [
                    0,
                    0,
                    0
                ],
                "Citrus": [
                    128,
                    0,
                    0
                ],
                "Corn": [
                    0,
                    128,
                    0
                ],
            }
    }

    train_names = []
    corn, citrus, names = copy_files([files_citrus_train, files_corn_train], pascalWriter )

    train_names += names

    summary['train'] = {
        'Citrus': citrus,
        'Corn': corn,
    }
    
    with open( os.path.join(args.output_path, DSN, 'ImageSets/Segmentation', 'train.txt' ), 'w') as f:
        for el in train_names:
            f.write(str(el)+'\n')

    
    val_names = []
    corn, citrus, names = copy_files([files_citrus_val, files_corn_val], pascalWriter )

    val_names += names

    summary['val'] = {
        'Citrus': citrus,
        'Corn': corn,
    }
    
    with open( os.path.join(args.output_path, DSN, 'ImageSets/Segmentation', 'val.txt' ), 'w') as f:
        for el in val_names:
            f.write(str(el)+'\n')

    
    test_names = []
    corn, citrus, names = copy_files([files_citrus_test, files_corn_test], pascalWriter )

    test_names += names

    summary['test'] = {
        'Citrus': citrus,
        'Corn': corn,
    }
    
    with open( os.path.join(args.output_path, DSN, 'ImageSets/Segmentation', 'test.txt' ), 'w') as f:
        for el in test_names:
            f.write(str(el)+'\n')

    
    
    with open( os.path.join(args.output_path, DSN, 'data', 'VOC_2012.json') , "w") as outfile:
        json.dump(summary, outfile, indent=4) 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='dataset_merged_v3')

    args = parser.parse_args()

    main(args)