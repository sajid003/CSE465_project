import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
from torch import optim, nn
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from UNet import *


class BrainTumorDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.images = sorted([root_path+"/images/"+i for i in os.listdir(root_path+"/images/")])
        self.masks = sorted([root_path+"/masks/"+i for i in os.listdir(root_path+"/masks/")])


        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("L")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)


    def __len__(self):
        return len(self.images)






if __name__ == "__main__":
    # Hyper params for training
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8 # Keeping batch size low. Otherwise GPU will be out of memory.
    EPOCHS = 15


    DATA_PATH = "./train"
    MODEL_SAVE_PATH = "./unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = BrainTumorDataset(DATA_PATH)


    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)


    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
          # Training epoch
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)


        # Validation epoch
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)