
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_object_detection_model(num_classes=11):
    # Load a pre-trained model for classification and adjust it
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    # num_classes which is user-defined (NuScenes has ~10-23 classes, simplified to 10 + background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
