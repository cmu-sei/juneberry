#! /usr/bin/env python3

# this script will update a labels.txt file produced from yymnist text detection test 
# and output a new file that suitable for use in our juneberry environment."""

import json
import logging
import argparse
from pathlib import Path

def main():
    # Set up some logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Setup and parse all arguments from command line
    parser = argparse.ArgumentParser(description="Converts rareplanes metadata files into the Juneberry coco metadata specification.")
    parser.add_argument('data_root', type=str, help='data root to use for resolutions')
    parser.add_argument('labels_file', type=str, help='annotations file to convert')
    parser.add_argument('output_file', type=str, help='where to write converted file')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise EnvironmentError(data_root, "Data root does not exist!")
    label_path = Path(args.labels_file)
    if not label_path.exists():
        raise EnvironmentError("Labels file does not exist!")

    #updates the path to relative. removes the data_root
    #rel_label_path = label_path.relative_to(data_root)

    #open file in read mode 
    label = open(label_path, 'r') 

    category_dict = [ 
        {"id" : 0, "name" : 'Zero'}, 
        {"id" : 1, "name" : 'One'}, 
        {"id" : 2, "name" : 'Two'}, 
        {"id" : 3, "name" : 'Three'}, 
        {"id" : 4, "name" : 'Four'},
        {"id" : 5, "name" : 'Five'},
        {"id" : 6, "name" : 'Six'},
        {"id" : 7, "name" : 'Seven'},
        {"id" : 8, "name" : 'Eight' },
        {"id" : 9, "name" : 'Nine'}]
    result = {'annotations': [], 'categories': category_dict, 'images': []} 

    #iterate through one line/image at a time 
    count = 0
    a_count = 0
    while True:
        line = label.readline()
        if not line: 
            break
        sect = line.split() # (path, box1, box2, ...)
       
        #----do for each line/image-----------
        
        #images[] 
        image_path = Path(sect[0]) # /yymnist_system_test/yymnist/Images10/000001.jpg
        rel_image_path = image_path.relative_to(data_root) #yymnist/Images10/000001.jpg
        image = {
            "id" : int(count),
            "file_name" : str(rel_image_path),  #same as first string in the line 
            "height" : 416, 
            "width" : 416 
        }
        result['images'].append(image)

        #lienses[]

        #info[]

        #----do for each box on the image----

        #annotations[] 
        for b in range(1, len(sect)):
            numb = sect[b].split(",") # (xmin, ymin, xmax, ymax, class)
            xmin = int(numb[0])
            ymin = int(numb[1])
            xmax = int(numb[2])
            ymax = int(numb[3])
            width = (xmax - xmin) 
            height = (ymax - ymin)
            area = height * width
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            box = [int(x_center), int(y_center), width, height]
            
            anno = {
                "id" : int(a_count), #num annotation for this image is what i was told but looks like it is # annotations overall 
                "image_id" : int(count), #link to image id
                "category_id" : int(numb[4]), #5th num of sect. the actual letter category id
                "area" : int(area), #width * height 
                "bbox" : box, #[x,y,width,height] x,y are the center coordinates 
                "iscrowd" : 0
            }
            result['annotations'].append(anno)
            a_count += 1
        #end for
        count += 1
    #end while

    label.close()

    #writing to the output file
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=4)

    logging.info("Done")
    
if __name__ == '__main__':
    main()