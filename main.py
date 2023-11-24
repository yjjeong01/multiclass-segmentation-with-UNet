import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.optim as optim

from utils.net import *
from utils import net as network
from utils.loss import *
from utils.utils import *

train_dataset = dataset(r"dataset/train/*")
val_dataset = dataset(r"dataset/valid/*")

H = 224
W = 224
batch_size = 16
num_workers = 8

dataloaders = {
    'train': DataLoader(dataload(H=H, W=W, data_path=train_dataset, aug=True), batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'valid': DataLoader(dataload(H=H, W=W, data_path=val_dataset, aug=False), batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

device_txt = "cuda:0"
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
num_classes = 19


if __name__ == '__main__':
    model = network.UNet(1, num_classes).to(device)
    model = nn.DataParallel(model, output_device='cuda:0')

    num_epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    dice_loss = DiceLoss()

    print("****************************GPU : ", device)

    best_loss = 1e10

    loss_L = []
    for epoch in range(1, num_epochs + 1):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('------------------------' * 10)
        phases = ['train', 'valid'] if epoch % 20 == 0 else ['train']

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0
            pbar = tqdm.tqdm(dataloaders[phase], unit='batch')

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    LOSS = dice_loss(outputs, labels)
                    metrics['Jointloss'] += LOSS

                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            if phase == 'valid':
                pred = outputs[0].cpu().detach().numpy()
                pred = np.sum(pred, axis=0)
                visualization(pred, title=f'Epochs {epoch}', cmap='gray')

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples
            loss_L.append(epoch_Jointloss.cpu().detach().numpy())

            for param_group in optimizer.param_groups:
                lr_rate = param_group['lr']
            print(phase, "Joint loss :", epoch_Jointloss.item(), 'lr rate', lr_rate)

            savepath = 'model/net_{}_E_{}.pth'
            if phase == 'valid' and epoch_Jointloss < best_loss:
                print("model saved")
                best_loss = epoch_Jointloss
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))

    plt.plot(loss_L)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
