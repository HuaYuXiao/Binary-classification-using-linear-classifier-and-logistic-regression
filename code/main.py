import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import copy

file_path = "/Users/hyx13701490089/Library/Mobile Documents/com~apple~CloudDocs/课程/EE271/ass6/"

with open(file_path+'val.txt', 'a+') as f:
    files = os.listdir(file_path+'data_train/empty')
    for each in files:
        filetype = os.path.split(each)[1]
        if filetype == '.txt':
            continue
        name ='/empty'+'/'+ each + ' 0\n'   
        f.write(name)
    files = os.listdir(file_path+'data_train/occupied')
    for each in files:
        filetype = os.path.split(each)[1]
        if filetype == '.txt':
            continue
        name = '/occupied' + '/' + each + ' 1\n'   
        f.write(name)

print('label finished')

data_transform = transforms.Compose([transforms.Resize((100,100)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),])

train_dataset = datasets.ImageFolder(root=file_path+'data_train',transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset+'data_train',batch_size=16,shuffle=True,num_workers=0)

val_dataset = datasets.ImageFolder(root=file_path+'data_val',transform=data_transform)
val_dataloader = DataLoader(dataset=val_dataset+'data_val',batch_size=16,shuffle=True, num_workers=0)

test_dataset = datasets.ImageFolder(root=file_path+'data_test',transform=data_transform)
test_dataloader = DataLoader(dataset=val_dataset+'data_test',batch_size=16,shuffle=True, num_workers=0)

print('dataset finished')

device = torch.device('cpu')

# 载入训练模型 
model = models.squeezenet1_1(weights=True)
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2
model.to(device)

# 优化方法、损失函数
optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
loss_fc = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer,10, 0.1)

# 训练
total_epoch = 2
# 训练日志保存
logfile_dir = file_path

acc_best_wts = model.state_dict()
best_acc = 0
iter_count = 0

for each in range(total_epoch):
    train_loss,train_acc = 0.0,0.0
    train_correct,train_total = 0,0

    val_loss,val_acc = 0.0,0.0
    val_correct,val_total = 0,0

    for i, sample_batch in enumerate(train_dataloader):
        inputs = sample_batch[0].to(device)
        labels = sample_batch[1].to(device)
    
        # 模型设置为train
        model.train()

        # forward
        outputs = model(inputs)

        loss = loss_fc(outputs, labels)

        # forward update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        train_loss += loss.item()
        train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        train_total += labels.size(0)

        print('iter:{}'.format(i))
        
        if i % 10 == 9:
            for sample_batch in val_dataloader:
                inputs = sample_batch[0].to(device)
                labels = sample_batch[1].to(device)

                model.eval()
                outputs = model(inputs)
                loss = loss_fc(outputs, labels)
                _, prediction = torch.max(outputs, 1)
                val_correct += ((labels == prediction).sum()).item()
                val_total += inputs.size(0)
                val_loss += loss.item()

            val_acc = val_correct / val_total

            print('[{},{}] train_loss = {:.5f} train_acc = {:.5f} val_loss = {:.5f} val_acc = {:.5f}'.format(
                each + 1, i + 1, train_loss / 100,train_correct / train_total, val_loss/len(val_dataloader),
                val_correct / val_total))
            if val_acc > best_acc:
                best_acc = val_acc
                acc_best_wts = copy.deepcopy(model.state_dict())

            with open(logfile_dir +'train_loss.txt','a') as f:
                f.write(str(train_loss / 100) + '\n')
            with open(logfile_dir +'train_acc.txt','a') as f:
                f.write(str(train_correct / train_total) + '\n')
            with open(logfile_dir +'val_loss.txt','a') as f:
                f.write(str(val_loss/len(val_dataloader)) + '\n')
            with open(logfile_dir +'val_acc.txt','a') as f:
                f.write(str(val_correct / val_total) + '\n')

            iter_count += 200
            
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            val_correct = 0
            val_total = 0
            val_loss = 0
    scheduler.step()

print('train finished')

# 保存模型
model_file =file_path
torch.save(acc_best_wts, model_file+'/model.pth')
print('model saved')

#测试
model = models.squeezenet1_1(weights=True)
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2
model.load_state_dict(torch.load(model_file+'/model.pth', map_location='cpu'))

model.eval()

test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=0)

correct = 0
total = 0
acc = 0.0
for i, sample in enumerate(test_dataloader):
    inputs, labels = sample[0], sample[1]

    outputs = model(inputs)

    _, prediction = torch.max(outputs, 1)
    correct += (labels == prediction).sum().item()
    total += labels.size(0)

acc = correct / total
print('test finished')

print('total:',total) 
print('correct:',correct)
print('acc:',acc)
