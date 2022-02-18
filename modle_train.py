import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def modle_train(net, epoch, train_loader, test_loader, loss_func, optimizer, min_testloss):
    y1 = []
    y2 = []
    net_epoch1 = "0"

    for epoch in range(epoch):
        train_loss = []
        test_loss = []
        for data in train_loader:
            onebatch, onebatchlables = data
            onebatch = onebatch.type(torch.FloatTensor).to(device)
            onebatchlables = onebatchlables.type(torch.LongTensor).to(device).view(-1, 1)
            output = net(onebatch, onebatch[:, -30:, :])
            loss = loss_func(output, torch.max(onebatchlables, 1)[1])  # cross entropy loss
            train_loss.append(loss.sum())
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()

        with torch.no_grad():
            for testdata in test_loader:
                onetestbatch, onetestbatchlables = testdata
                onetestbatch = onetestbatch.type(torch.FloatTensor).to(device)
                onetestbatchlables = onetestbatchlables.type(torch.FloatTensor).to(device).view(-1, 1)
                test_output = net(onetestbatch, onetestbatch[:, -30:, :])

                testloss = loss_func(test_output, torch.max(onetestbatchlables, 1)[1])
                test_loss.append(testloss.sum())
            # print(sum(test_loss))


        y1.append((sum(train_loss).cpu().detach().numpy()) / len(train_loss))
        y2.append((sum(test_loss).cpu().detach().numpy()) / len(test_loss))
    if epoch % 5 == 0:
        # torch.save(net, "goodnet.pt")
        net_epoch1 = str(epoch)
        print("epoch:{},train loss:{},test loss:{}".format(net_epoch1, y1[-1], y2[-1]))
    return y1, y2, net_epoch1


        # with torch.no_grad():
        #     for validata in validation_loader:
        #         onetestbatch,onetestbatchlables=validata
        #         onetestbatch=onetestbatch.type(torch.FloatTensor).to(device)
        #         onetestbatchlables=onetestbatchlables.type(torch.LongTensor).to(device)
        #         test_output=net(onetestbatch,onetestbatch[:,-30:,:])
        #
        #
        #         testloss =loss_func(test_output,torch.max(onetestbatchlables,1)[1])
        #         test_loss.append(testloss.sum())
        #     # print(sum(test_loss))
        # y1.append(sum(train_loss).cpu().detach().numpy())
        # y2.append(sum(test_loss))
        # # y2.append(sum(test_loss).cpu().detach().numpy())



def modle_train2(net,epoch,train_loader,validation_loader,loss_func,optimizer,min_testloss):
    y1 = []
    y2 = []
    net_epoch1="0"

    for epoch in range(epoch):
        train_loss = []
        test_loss = []
        for data in train_loader:
            onebatch,onebatchlables=data
            onebatch=onebatch.type(torch.FloatTensor).to(device)
            onebatchlables =onebatchlables.type(torch.FloatTensor).to(device)
            output=net(onebatch,onebatch)
            loss = loss_func(output, onebatchlables)  # cross entropy loss
            train_loss.append(loss.sum())
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()


        with torch.no_grad():
            for validata in validation_loader:
                onetestbatch,onetestbatchlables=validata
                onetestbatch=onetestbatch.type(torch.FloatTensor).to(device)
                onetestbatchlables=onetestbatchlables.type(torch.FloatTensor).to(device)
                test_output=net(onetestbatch,onetestbatch)


                testloss =loss_func(test_output,onetestbatchlables)
                test_loss.append(testloss.sum())
            # print(sum(test_loss))
        y1.append((sum(train_loss).cpu().detach().numpy())/len(train_loss))
        y2.append((sum(test_loss).cpu().detach().numpy())/len(test_loss))


        if y2[-1].item()<min_testloss:
            min_testloss=y2[-1].item()
            torch.save(net,"goodnet.pt")

            net_epoch1=str(epoch)
            print("epoch:{},train loss:{},test loss:{}".format(net_epoch1,y1[-1],y2[-1]))
    return y1,y2,net_epoch1