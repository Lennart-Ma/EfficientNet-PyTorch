import torch

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def training_loop(model, device, train_loader, optimizer, loss_function, epoch, num_epochs):

    model.train()

    n_total_steps = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.type(torch.DoubleTensor)
        
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            output = model(data)

            loss = loss_function(torch.flatten(output), target)

            loss.backward()

            optimizer.step()

            if batch_idx % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}')


def val_loop(model, device, val_loader, loss_function):

    model.eval()

    with torch.no_grad():
        for i, (data,target) in enumerate(val_loader):
            
            target = target.type(torch.DoubleTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            curr_loss = loss_function(torch.flatten(output),target)
            
            # Is this the correct solution for calculating the 
            output = torch.sigmoid(output)
            output = output.cpu().numpy() #this numpy array needs to look like this np([1, 0, 1, 0]) instead of np([0.2, -0.1, 0.8, 0.7])
            
            for j in range(len(output)):
                if output[j] > 0.5:
                    output[j] = 1
                else:
                    output[j] = 0

            if i==0:
                predictions = output
                targets = target.data.cpu().numpy()
                loss = np.array([curr_loss.data.cpu().numpy()])

            else:
                predictions = np.concatenate((predictions, output))
                targets = np.concatenate((targets, target.data.cpu().numpy()))
                loss = np.concatenate((loss, np.array([curr_loss.data.cpu().numpy()])))

    accuracy = accuracy_score(targets, predictions)
    conf_mat = confusion_matrix(targets, predictions)

    sensitivity = conf_mat.diagonal()/conf_mat.sum(axis=1)

    print("Test Accuracy: ", accuracy, "Test Sensitivity (Overall): ", np.mean(sensitivity), "Test loss: ", np.mean(loss))

    return targets, predictions, accuracy