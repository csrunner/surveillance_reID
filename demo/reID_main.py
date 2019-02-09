import detection
import reid_api
detection.Detection(0,'./cfg/yolov3.cfg','./cfg/yolov3.weights')
print('yolo weight files loaded')

model_path = '/home/user/lyj_ReID/tyrant_0110/Person_reID_baseline_pytorch-master/model/ft_ResNet50'
reid_api.ReID('0', model_path)
print('reid weight files loaded')

def main():
    detection.Detection(0,'./detection/cfg/yolov3.cfg','./detection/cfg/yolov3.weights')
    #detection.detect_one_frame()
    
    print('weight files loaded')

if '__name__' == '__main__':
    main()
