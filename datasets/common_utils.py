import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
from device import device
from typing import Tuple


def collate_fn(batch):

    imgs = torch.cat([img.unsqueeze(0) for img, _ in batch], dim=0)
    return imgs, {
        "boxes": [targets["boxes"] for _, targets in batch],
        "labels": [targets["labels"] for _, targets in batch]
    }


def draw_box(pil_img: Image.Image, bboxes, confs=None, color=(255, 255, 255, 150)):
    draw = ImageDraw.Draw(pil_img)
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
        if confs is not None:
            conf = confs[i]
            draw.text(
                (xmin, max(ymin - 10, 4)),
                "{:.2f}".format(conf.item()),
                color
            )


def draw_dots(pil_img: Image.Image, pred_boxes, pred_landmarks: Tuple[float], r=2, constrain_pts=False):
    draw = ImageDraw.Draw(pil_img)
    for bbox, landmark in zip(pred_boxes, pred_landmarks):
        xmin, ymin, xmax, ymax = bbox
        for x, y in np.array_split(landmark, 5):
            if not constrain_pts:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
            else:
                if xmin <= x and x <= xmax and ymin <= y and y <= ymax:
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))


def torch_img_denormalization_to_pil(img: torch.Tensor) -> Image.Image:
    mean = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
    img = (img * std + mean) * 255
    img = Image.fromarray(img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8"))
    return img


def xyxy_to_cxcywh(bboxes):
    if len(bboxes) == 0:
        return bboxes
    cxcy = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2
    wh = (bboxes[:, 2:4] - bboxes[:, 0:2])

    if isinstance(bboxes, torch.Tensor):
        def cat_func(arr_to_concat): return torch.cat(arr_to_concat, dim=-1)
    else:
        def cat_func(arr_to_concat): return np.concatenate(arr_to_concat, axis=-1)

    out = cat_func([cxcy, wh])
    return out


def cxcywh_to_xyxy(bboxes):

    if len(bboxes) == 0:
        return bboxes
    xmin_ymin = bboxes[:, 0:2] - bboxes[:, 2:4] / 2
    xmax_ymax = bboxes[:, 0:2] + bboxes[:, 2:4] / 2

    if isinstance(bboxes, torch.Tensor):
        def cat_func(arr_to_concat): return torch.cat(arr_to_concat, dim=-1)
    else:
        def cat_func(arr_to_concat): return np.concatenate(arr_to_concat, axis=-1)

    out = cat_func([xmin_ymin, xmax_ymax])

    return out


torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
