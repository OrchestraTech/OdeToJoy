#Датасет EasyPortait для сегментации человеческого лица в кадре
#0	BACKGROUND, 1	PERSON, 2	SKIN, 3	LEFT BROW, 4	RIGHT_BROW, 5	LEFT_EYE, 6	RIGHT_EYE, 7	LIPS, 8	TEETH
from email.mime import image
import os
from turtle import forward
import torch
from PIL import Image
from torch.nn.modules import BatchNorm2d
import torch.utils.data as data
import torchvision.transforms.v2 as tfs  
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torchmetrics import JaccardIndex

class PortraitDataset(data.Dataset):
    def __init__(self, path, train=True, validation=False, image_transform=None, annotation_transform=None):
        self.path = path
        self.image_transform = image_transform
        self.annotation_transform = annotation_transform
        self.samples_images = [] #массив состоит из картинок
        self.samples_annotations = [] #массив состоит из масок        

        images_path = os.path.join(self.path, "images")
        if validation:
            images_path = os.path.join(images_path, "val")
        else:
            images_path = os.path.join(images_path, "train" if train else "test")

        annotations_path = os.path.join(self.path, "annotations")
        if validation:
            annotations_path = os.path.join(annotations_path, "val")
        else:
            annotations_path = os.path.join(annotations_path, "train" if train else "test")


        if os.path.exists(images_path):
            for img_name in sorted(os.listdir(images_path)):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = os.path.splitext(img_name)[0]
                ann_name = stem + ".png"
                ann_name_full = os.path.join(annotations_path, ann_name)
                if not os.path.exists(ann_name_full):
                    continue
                self.samples_images.append(os.path.join(images_path, img_name))
                self.samples_annotations.append(ann_name_full)

        assert len(self.samples_images) == len(self.samples_annotations), \
        f"images={len(self.samples_images)}, annotations={len(self.samples_annotations)}"
        assert len(self.samples_images) > 0, f"пусто: {images_path}"
                
        
    def __getitem__(self, index):
        image_path, annotation_path = self.samples_images[index], self.samples_annotations[index]

        image_ = Image.open(image_path).convert("RGB") #на всякий случай
        annotation_ = Image.open(annotation_path).convert("L") #grayscale

        if self.image_transform:
            image_ = self.image_transform(image_)

        if self.annotation_transform:
            annotation_ = self.annotation_transform(annotation_)
            annotation_ = annotation_.squeeze(0)
        annotation_ = annotation_.long()

        return image_, annotation_

    def __len__(self):
        return len(self.samples_images)



class UNetNN(nn.Module):

    class TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                )
        def forward(self, x):
            return self.model(x)

    class EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetNN.TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)
        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNetNN.TwoConvLayers(in_channels, out_channels)
        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x,y], dim=1)
            u = self.block(u)
            return u

    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        
        self.enc_block1 = self.EncoderBlock(in_channels, 64)
        self.enc_block2 = self.EncoderBlock(64, 128)
        self.enc_block3 = self.EncoderBlock(128, 256)
        self.enc_block4 = self.EncoderBlock(256, 512)

        self.bottleneck = self.TwoConvLayers(512, 1024)

        self.dec_block1 = self.DecoderBlock(1024, 512)
        self.dec_block2 = self.DecoderBlock(512, 256)
        self.dec_block3 = self.DecoderBlock(256, 128)
        self.dec_block4 = self.DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)

#Так как у нас дисбаланс классов, сейчас напишем свою функцию потерь ( по факту это что то типо mIoU)
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=9, smooth=1.0, ignore_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background
    def forward(self, logits, targets):
        probs = torch.nn.functional.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        interseption = (probs * targets_one_hot).sum(dim=(0,2,3))
        prob_sum = probs.sum(dim=(0,2,3))
        target_sum = targets_one_hot.sum(dim=(0,2,3))
        dice_per_class = (2.0 * interseption + self.smooth) / (prob_sum + target_sum + self.smooth)
        if self.ignore_background:
            dice_per_class = dice_per_class[1:]
        return 1.0 - dice_per_class.mean()

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    image_transforms = tfs.Compose([tfs.Resize((256, 256), interpolation=tfs.InterpolationMode.BILINEAR), tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)]) #Среднее из 4 ближайших соседей
    annotation_transforms = tfs.Compose([tfs.Resize((256, 256), interpolation=tfs.InterpolationMode.NEAREST), tfs.PILToTensor()]) #Просто ближайший сосед
    path_to_my_dataset = r"D:\EasyPortrait"

    d_train = PortraitDataset(path_to_my_dataset, image_transform=image_transforms, annotation_transform=annotation_transforms)
    train_data = data.DataLoader(d_train, batch_size=2, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)

    d_val = PortraitDataset(path_to_my_dataset, train=False, validation=True, image_transform=image_transforms, annotation_transform=annotation_transforms)
    val_data = data.DataLoader(d_val, batch_size=2, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)

    d_test = PortraitDataset(path_to_my_dataset, train=False, validation=False, image_transform=image_transforms, annotation_transform=annotation_transforms)
    test_data = data.DataLoader(d_test, batch_size=2, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)

    model = UNetNN()
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)

    loss1 = torch.nn.CrossEntropyLoss()
    loss2 = SoftDiceLoss()

    #Для графика валидации
    loss_lst_val = []
    loss_lst_train = []


    epochs = 25

    # for _e in range(epochs):
    #     model.train()

    #     loss_mean = 0
    #     lm_count = 0

    #     train_tqdm = tqdm(train_data)
    #     for x_train, y_train in train_tqdm:
    #         x_train = x_train.to(device)
    #         y_train = y_train.to(device)
    #         predict = model(x_train)
    #         loss = loss1(predict, y_train) + loss2(predict, y_train)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         lm_count += 1
    #         loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
    #         train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean}")

    #     #Валидация модели
    #     model.eval()
    #     Q_val = 0
    #     count_val = 0

    #     for x_val, y_val in val_data:
    #         x_val = x_val.to(device)
    #         y_val = y_val.to(device)
    #         with torch.no_grad():
    #             predict = model(x_val)
    #             loss = loss1(predict, y_val) + loss2(predict, y_val)
    #             Q_val += loss.item()
    #             count_val += 1
    #     Q_val /= count_val

    #     loss_lst_train.append(loss_mean)
    #     loss_lst_val.append(Q_val)
    #     print(f"validation_loss={Q_val}")

    #Сохранение модели
    # st = model.state_dict()
    # torch.save(st, 'odetojoy.tar')
    #Загрузка модели
    #------
    st = torch.load("odetojoy.tar", weights_only=True)
    model.load_state_dict(st)
    #-----
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, epochs + 1), loss_lst_train, label="train", marker="o")
    # plt.plot(range(1, epochs + 1), loss_lst_val, label="val", marker="s")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Train vs Val Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(range(1, epochs + 1))
    # plt.show()


    #Тестирование
    model.eval()

    miou = JaccardIndex(task="multiclass", num_classes=9, average="macro").to(device)
    with torch.no_grad():
        for x_test, y_test in test_data:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            logits = model(x_test)
            preds = logits.argmax(dim=1)
            miou.update(preds, y_test)
    miou = miou.compute()
    print(f"mIoU: {miou.item() * 100} %")



    img = Image.open(r"D:\testimg.jpg").convert("RGB")
    img = image_transforms(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        logits = model(img)

    probs = torch.nn.functional.softmax(logits, dim=1)          
    pred = probs.argmax(dim=1)                
    pred = pred.squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(pred, cmap="tab10", vmin=0, vmax=8)  # tab10 — 10 цветов, хватит на 9
    plt.colorbar(ticks=range(9), label="Класс")
    plt.title("Предсказанная маска (9 классов)")
    plt.axis("off")
    plt.show()






