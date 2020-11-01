"""
This script reads the Category and Attribute Prediction Benchmark from the DeepFashion dataset and splits the data into train/val/test groups and saves the img_path,bbox vector,category vector,
attribute vector for each image in all the 3 groups
"""
import os
import numpy as np
import pandas as pd


class create_DeepFashion:

    def __init__(self, dataset_path):

        # The constants
        img_folder_name = "dataset/img"
        eval_folder_name = "dataset/Eval"
        anno_folder_name = "dataset/Anno"
        list_eval_partition_file = "list_eval_partition.txt"
        list_attr_img_file = "list_attr_img.txt"
        list_category_img_file = "list_category_img.txt"
        list_category_cloth_file = "list_category_cloth.txt"
        list_bbox_file = "list_bbox.txt"
        # The data structures
        self.train = pd.DataFrame(columns = ["img_path","bbox","category","attributes"])
        self.val = pd.DataFrame(columns = ["img_path","bbox","category","attributes"])
        self.test = pd.DataFrame(columns = ["img_path","bbox","category","attributes"])

        # Construct the paths
        self.path = dataset_path
        self.img_dir = os.path.join(self.path, img_folder_name)
        self.eval_dir = os.path.join(self.path, eval_folder_name)
        self.anno_dir = os.path.join(self.path, anno_folder_name)

        self.list_eval_partition = os.path.join(self.eval_dir, list_eval_partition_file)
        self.list_attr_img = os.path.join(self.anno_dir, list_attr_img_file)
        self.list_category_img = os.path.join(self.anno_dir, list_category_img_file)
        self.list_category_cloth = os.path.join(self.anno_dir, list_category_cloth_file)
        self.list_bbox = os.path.join(self.anno_dir, list_bbox_file)
 

    def read_imgs_and_split(self):

        #Declating the names of the csv where the split-data would be stored
        train_file = "train.csv"
        val_file = "val.csv"
        test_file = "test.csv"

        if os.path.exists("/split-data/" + train_file):
            print("Reading data structures from: ", train_file)
            temp = pd.read_csv("/split-data/" + train_file, usecols = [0])
            print("Training images", int(temp.shape[0]))
            if os.path.exists("/split-data/" + val_file):
                print("Reading data structures from: ", val_file)
                temp = pd.read_csv("/split-data/" + val_file, usecols = [0])
                print("Val images", int(temp.shape[0]))
            if os.path.exists("/split-data/" + test_file):
                print("Reading data structures from: ", test_file)
                temp = pd.read_csv("/split-data/" + test_file, usecols = [0])
                print("Testing images", int(temp.shape[0]))    
            
            return

        # Read in the category index to category name mapping from the DeepFashion dataset
        category_to_name = {}

        with open(self.list_category_cloth) as f:
            count = int(f.readline().strip()) #Read the first line
            _ = f.readline().strip()  # read and throw away the header

            i = 0
            for line in f:
                words = line.split()
                category_to_name[i] = str(words[0])
                i = i + 1
            
        assert(count == 50)
    

        # Read in the image to category mapping from the DeepFashion dataset
        image_to_category = {}
        with open(self.list_category_img) as f:
            imgs_count = int(f.readline().strip()) #Read the first line
            _ = f.readline().strip()  # read and throw away the header

            #Read each line and split the words and store the data
            for line in f:
                words = line.split()

                #image_to_category[img_path] = category_index
                image_to_category[words[0].strip()] = int(words[1].strip())

        assert(imgs_count == len(image_to_category))

	    # Read in the image to bbox mapping
        image_to_bbox = {}
        with open(self.list_bbox) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # read and throw away the header

            #Read each line and split the words and store the data
            for line in f:
                words = line.split()

                #Collecting the tuple of bbox : (x1,y1,x2,y2)
                data = (words[1],words[2],words[3],words[4])

                #image_to_bbox[img_path] = tuple of bbox
                image_to_bbox[words[0]] = data

        assert(imgs_count == len(image_to_bbox))

        # Read in the images
        with open(self.list_eval_partition) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # read and throw away the header

            for line in f:
                words = line.split()
                img = words[0].strip()
                category_idx = image_to_category[img]
                category = str(category_to_name[category_idx - 1]) 
                bbox = np.asarray(image_to_bbox[img],dtype = np.int16)

                #Divide and save the data into train/val/test dataframe
                #Save a tuple of (img_path,bbox_list,category_onehot,attributes_vector)
                if words[1].strip() == "train":
                    self.train = self.train.append({"img_path" : img, "bbox" : bbox, "category" : category},ignore_index = True )
                if words[1].strip() == "val":
                    self.val = self.val.append({"img_path" : img, "bbox" : bbox, "category" : category},ignore_index = True)
                if words[1].strip() == "test":
                    self.test = self.test.append({"img_path" : img, "bbox" : bbox, "category" : category},ignore_index = True)

        print("Training images", int(self.train.shape[0]))
        print("Validation images", int(self.val.shape[0]))
        print("Test images", int(self.test.shape[0]))
        assert(imgs_count == int(self.train.shape[0])+int(self.test.shape[0])+int(self.val.shape[0]))

        # Store the data structures
        self.train.to_csv( self.path + "/split-data/train_new.csv", index = False)
        self.val.to_csv( self.path + "/split-data/val_new.csv", index = False)
        self.test.to_csv( self.path +"/split-data/test_new.csv", index = False)
        print("Storage done")


if __name__ == "__main__":
    df = create_DeepFashion("D:/Flipkart Grid/Fashion-Intelligence-Systems/dataset/Category and Attribute Prediction")
    df.read_imgs_and_split()