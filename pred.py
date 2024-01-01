import tqdm
from torch.utils.data import DataLoader

from utils import net as network
from utils.net import *
from utils.utils import *


val_dataset = dataset(r"dataset/valid/*")

H = 224
W = 224
batch_size = 1
num_workers = 1
dataloaders = {
    'valid': DataLoader(dataload(H=H, W=W, data_path=val_dataset, aug=False), batch_size=batch_size, shuffle=True, num_workers=num_workers)
}

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_classes = 19


if __name__ == '__main__':
    model = network.UNet(1, num_classes).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(r"net_0.009118677116930485_E_899_evimo.pth", map_location=torch.device(device)))
    model.eval()

    pbar = tqdm.tqdm(dataloaders['valid'], unit='batch')

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        input = inputs[0][0].detach().cpu().numpy()
        label = labels[0].detach().cpu().numpy()
        label = np.sum(label, axis=0)

        class_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        result = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(num_classes):
            color_map[i][0], color_map[i][2] = color_map[i][2], color_map[i][0]
            result[class_labels[0] == i] = color_map[i]

        visualization(input, cmap='gray', title='Input')
        visualization(label, cmap='gray', title='Ground Truth')
        visualization(result, title='Prediction')
