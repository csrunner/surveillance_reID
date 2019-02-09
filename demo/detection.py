import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import os 
import os.path as osp
import warnings
from darknet import Darknet
from preprocess import prep_image, prep_image_cpu, inp_to_image
import random 
import itertools
import bbox

warnings.filterwarnings('ignore')
CUDA = torch.cuda.is_available()
DEBUG = False
RESO = 416

class Detection:
    def __init__(self, gpuID, cfgfile, weightsfile):
        start = 0
        #check the params
        if not os.path.isfile(cfgfile):
            print("class Detection: cfgfile: {0} does not exist.exit()".format(cfgfile))
            exit()
        if not os.path.isfile(weightsfile):
            print("class Detection: weightsfile: {0} does not exist.exit()".format(weightsfile))
            exit()
        
        scales = "1,2,3"
        batch_size = 1
        confidence = 0.5
        nms_thesh = 0.4
        self.gpuID = gpuID
        if CUDA:
            torch.cuda.set_device(gpuID)

        if DEBUG:
            print(scales)
            print(images)
            print(batch_size)
            print(confidence)
            print(nms_thresh)
            print(cfgfile)
            print(weightsfile)

        self.num_classes = 80
        self.reso = RESO

        #Set up the neural network
        print("Loading detection network.....")
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        print("Detection network successfully loaded")
    
        self.model.net_info["height"] = self.reso
        inp_dim = int(self.model.net_info["height"])
        if DEBUG:
            print("inp_dim: {0}".format(inp_dim))
        assert inp_dim % 32 == 0 
        assert inp_dim > 32

        #If there's a GPU availible, put the model on GPU
        if CUDA:
            self.model.cuda()
    
    
        #Set the model in evaluation mode
        self.model.eval()
    
    def detect_one_frame(self,img):
        #start = time.time()
        if CUDA:
            torch.cuda.set_device(self.gpuID)
        batch_size = 1
        inp_dim = self.reso 
        num_classes = self.num_classes 

        #Detection phase
        imlist = [img]
        batches = list(map(prep_image_cpu, imlist, [inp_dim for x in range(len(imlist))]))
        #print("from start to preprocess finish. time: {0:3f}s".format(time.time()-start))
        if DEBUG:
            print("batches.shape");print(len(batches))

        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
         
        if CUDA:
            im_dim_list = im_dim_list.cuda()
    
        leftover = 0
    
        if (len(im_dim_list) % batch_size):
            leftover = 1
        
        
        if batch_size != 1:
            num_batches = len(imlist) // batch_size + leftover            
            im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        


        i = 0
        write = False
        objs = {}
        for batch in im_batches:
            #load the image 
            #start = time.time()
            #cptogpu = time.time()
            if CUDA:
                batch = batch.cuda()
        
            #Apply offsets to the result predictions
            #Tranform the predictions as described in the YOLO paper
            #flatten the prediction vector 
            # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
            # Put every proposed box as a row.
            #inference = time.time() 
            with torch.no_grad():
                prediction = self.model(Variable(batch), CUDA)
             
            #print("from cptogpu   to inference finish. time: {0:3f}s".format(time.time()-cptogpu))
            #print("from inference to inference finish. time: {0:3f}s".format(time.time()-inference))
            #print("from start     to inference finish. time: {0:3f}s".format(time.time()-start))
    #        prediction = prediction[:,scale_indices]

        
            #get the boxes with object confidence > threshold
            #Convert the cordinates to absolute coordinates
            #perform NMS on these boxes, and save the results 
            #I could have done NMS and saving seperately to have a better abstraction
            #But both these operations require looping, hence 
            #clubbing these ops in one loop instead of two. 
            #loops are slower than vectorised operations. 
        
            #prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            confidence = 0.5
            num_classes = 80
            nms = True
            nms_thresh = 0.4
         
            prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thresh)
            #print("from start     to write_results finish. time: {0:3f}s".format(time.time()-start))
            
        
            if type(prediction) == int:
                i += 1
                continue

            #end = time.time()
            prediction[:,0] += i*batch_size
          
            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output,prediction))

                #if DEBUG:
                    #print("predict time:{0:.3f}s".format((end - start)/batch_size))
                    #print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            i += 1

            if CUDA:
                torch.cuda.synchronize()
        try:
            output
        except NameError:
            #end = time.time()
            #if DEBUG:
                #print("No detections were made")
                #print("time cost:{0:.3f}".format(end -  start))
            return 0 
        
        #print("from start     to cudasynchronize() finish. time: {0:3f}s".format(time.time()-start))
        #beginscale = time.time()
        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
        
    
    
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
        output[:,1:5] /= scaling_factor
    
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
        
        coords = np.zeros([len(output),4],dtype = int) 
        #print("coords=");print(coords)
        idx=0
        for x in output:
            coords[idx] = x[1:5].int().cpu().numpy()
            idx+=1
        
        #end = time.time()
        #print("from beginscale to coord finish. time: {0:3f}s".format(time.time()-beginscale))
        #torch.cuda.empty_cache()
        #print("from start      to scaling finish. time: {0:3f}s".format(time.time()-start))
        #print("from beginscale to scaling finish. time: {0:3f}s".format(time.time()-beginscale))

        #if DEBUG:
            #print("time cost:{0:.3f}".format(end -  start))
        return coords 
