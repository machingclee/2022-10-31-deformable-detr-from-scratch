import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import config
from argument_parser import ParsedArgument, get_args_parser
from datasets.face.wider_face_dataset import FacialLandmarkValidationDataset, FacialLandmarkTrainingDataset
from datasets.rust.rust_dataset import RustAnnotationDataset, torch_img_transform
from datasets.common_utils import torch_img_denormalization_to_pil, draw_dots, draw_box, cxcywh_to_xyxy
from glob import glob
from torch.utils.data import DataLoader
from PIL import ImageDraw, Image, ImageFont
from copy import deepcopy
from deformable_detr import build
from device import device

from datasets.rust.rust_dataset import resize_and_padding, cxcywh_to_xyxy
from torchvision.ops import nms
from typing import TypedDict, List, Optional, Tuple


class DetectionResult(TypedDict, total=False):
    label: List[str]
    bbox: List[List[float]]
    score: List[List[float]]


def inference(fast_rcnn: nn.Module = None, pillow_img=None):
    # type: (Optional[nn.Module], Optional[Image.Image]) -> Tuple[float, List[float]]
    """
    return:
        max_score: float
        max_score_box: List[float] in terms of [xmin, ymin, xmax, ymax]
    """
    img = pillow_img

    img, padding_window, (ori_w, ori_h) = resize_and_padding(img, return_window=True)
    x_scale = ori_w / padding_window[0]
    y_scale = ori_h / padding_window[1]
    box_scaling = torch.as_tensor([x_scale, y_scale, x_scale, y_scale]).to(device)
    img = torch_img_transform(img)
    scores, boxes, cls_idxes, rois = fast_rcnn(img[None, ...])
    boxes = boxes * box_scaling

    if len(boxes) == 0:
        return 0, []
    # draw = ImageDraw.Draw(img_original_size)
    # for score, box, cls_idx in zip(scores, boxes, cls_idxes):
    #     xmin, ymin, xmax, ymax=box
    #     draw.rectangle(((xmin, ymin), (xmax, ymax)), outline = 'blue', width = 3)
    #     draw.text(
    #         (xmin, max(ymin - 20, 4)),
    #         "{}: {:.2f}".format(cls_names[cls_idx.item()], score.item()), "blue",
    #         font=font
    #     )
    # img_original_size.show()

    max_score_index = torch.argmax(scores).detach().cpu().item()
    max_score = scores[max_score_index].detach().cpu().item()
    max_score_box = boxes[max_score_index].detach().cpu().numpy()

    return max_score, list(max_score_box)

def collate_fn(batch):
    imgs = torch.cat([img  for img, _ in batch])
    return imgs, {
        "boxes":  [targets["boxes"] for _, targets in batch],
        "labels": [targets["labels"] for _, targets in batch]
    }
    

def visualize_training_data(n_images: int):
    training_data_loader = DataLoader(dataset=FacialLandmarkTrainingDataset(),
                                      batch_size=1,
                                      collate_fn=collate_fn,
                                      shuffle=True)
    train_iter = iter(training_data_loader)
    for i in range(n_images):
        img, targets = next(train_iter)
        boxes = targets["boxes"]
        boxes = cxcywh_to_xyxy(boxes.squeeze(0)) * config.input_height
        # landmarks = landmarks.squeeze(0)
        pil_img = torch_img_denormalization_to_pil(img)
        draw_box(pil_img, boxes, color=(0, 0, 255, 150))
        # draw_dots(pil_img, bboxes, landmarks)
        pil_img.save("dataset_check/{}.jpg".format(str(i).zfill(3)))


def visualize(model: nn.Module = None, image_name: Optional[str] = None, cls_names=["face"]):
    font = ImageFont.truetype(config.font_path, size=16)
    # type: (...) -> None
    val_data_loader = DataLoader(dataset=FacialLandmarkValidationDataset(),
                                 batch_size=1,
                                 shuffle=True)
    img, target_bboxes, file_path = next(iter(val_data_loader))
    target_bboxes = target_bboxes.to(device).squeeze(0)
    
    img = img.squeeze(0)
    pil_img = Image.fromarray(img.cpu().numpy())
    img_ori = pil_img
    
    img_for_predict = torch_img_transform(deepcopy(pil_img)).to(device).unsqueeze(0)
    gt_boxes = target_bboxes
    

    draw=ImageDraw.Draw(img_ori)

    for box_info in gt_boxes:
        x1, y1, x2, y2=box_info
        draw.rectangle(((x1, y1)
                        , (x2, y2)), outline = (0, 255, 0, 255), width = 1)
        draw.text(
            (x1, max(y1 - 20, 4)),
            "{}".format(cls_names[0]), (0, 255, 0),
            font = font
        )
    draw.text(
        (10, config.input_height - 40),
        file_path[0],
        font=font
    )
    
    if model is not None:
        with torch.no_grad():
            model.eval()
            out = model(img_for_predict)
            pred_logits = out["pred_logits"].squeeze(0)
            pred_boxes = out["pred_boxes"].squeeze(0)
            
            scores = pred_logits.softmax(-1)[:, -1]
            cls_idxes = pred_logits.argmax(-1) - 1
            pred_boxes = cxcywh_to_xyxy(pred_boxes) * config.input_height
            keep = scores > config.pred_thres
 
            for score, box, cls_idx in zip(scores[keep], pred_boxes[keep], cls_idxes[keep]):
                xmin, ymin, xmax, ymax = box
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue', width=1)
                draw.text(
                    (xmin, max(ymin - 20, 4)),
                    "{}: {:.2f}".format(cls_names[cls_idx.item()], score.item()), "blue",
                    font=font
                )
            model.train()

    # img_ori = img_ori.crop(padding_window)
    # img_ori.resize(original_wh)
    
    if image_name is not None:
        img_ori.save(image_name)
        
    img_ori.save("performance_check/latest.jpg")

   

    
def denormalize(tensor_img):
    mean = torch.as_tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.expand_as(tensor)
    std = std.expand_as(tensor)
    tensor = tensor * std + mean
    tensor = tensor * 255
    tensor = tensor.detach().cpu().numpy().astype("uint8")
    img = Image.fromarray(tensor)
    return img
    
    

if __name__=="__main__":
    # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    # args: ParsedArgument = parser.parse_args()
    # model, criterion, postprocessors = build(args)
    # model = model.to(device)
    # visualize(model)
    visualize_training_data(100)
        
        
        