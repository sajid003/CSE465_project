import torch
from torchvision import transforms
from UNet import *
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(1,1).to(device) # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('Final_Unet_1000_images.pth', weights_only=True))
model.eval()



transform = transforms.Compose([
    # transforms.Resize((512, 512)),
    transforms.ToTensor()])

img = transform(Image.open("/content/brisc2025_train_00028_gl_ax_t1.jpg").convert("L")).float().to(device)
mask = transform(Image.open("/content/brisc2025_train_00028_gl_ax_t1.png").convert("L")).float().to(device)
print(img.shape)
print(mask.shape)
img = img.unsqueeze(0)
mask = mask.unsqueeze(0)
print(img.shape)
print(mask.shape)





class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, model_output):
        return (model_output[0, :, : ] * self.mask).sum()


target_layers = [model.up_convolution_4.conv.conv_op[1]]
targets = [SemanticSegmentationTarget(mask)]
with GradCAM(model=model,
             target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=img,
                        targets=targets)[0, :]
    # cam_image = show_cam_on_image(img, grayscale_cam)


colormap = plt.get_cmap('coolwarm')  # Choose a colormap: 'jet', 'viridis', 'plasma', etc.
rgb_image = colormap(grayscale_cam)
print(grayscale_cam)
plt.imshow(rgb_image)
plt.show()
# Image.fromarray(cam_image)