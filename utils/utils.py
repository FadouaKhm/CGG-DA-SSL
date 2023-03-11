import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
from sklearn.metrics import *
from utils.plot_cf import *
import torch
import numpy as np
def get_vgg_features(net, image):
    x = net.features(image)    
    x = net.avgpool(x)
    x = torch.flatten(x, 1)
    x = net.classifier[:4](x)
    return x.cpu().detach().numpy()
def eval_model(eval_loader, net, exp_id, plt_cf = True):
    acc = 0.0
    y_true = []
    y_pred = []
    y_logits = []
    #feats = []
    with torch.no_grad():
        for _, image, target in eval_loader:

            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            

            
            y_true.extend(target.cpu().tolist())
            y_pred.extend(torch.max(logit, dim=-1)[1].cpu().tolist()) 
            y_logits.extend(torch.softmax(logit, dim=-1).cpu().tolist())
            
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
            #feats.append(get_vgg_features(net, image))

    print('Accuracy = ' ,accuracy_score(y_true, y_pred))  
    if plt_cf:
            eval_viz(y_true, y_pred, ['C'+str(i) for i in range(max(y_true))], exp_id)
    return y_true, y_pred, y_logits
def eval_model_feat(eval_loader, net, exp_id, plt_cf = True):
    acc = 0.0
    y_true = []
    y_pred = []
    y_logits = []
    feats = []
    all_x = []
    with torch.no_grad():
        for _, image, target in eval_loader:

            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            

            
            y_true.extend(target.cpu().tolist())
            y_pred.extend(torch.max(logit, dim=-1)[1].cpu().tolist()) 
            y_logits.extend(torch.softmax(logit, dim=-1).cpu().tolist())
            
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
            feats.append(get_vgg_features(net, image))
            # all_x.append(image.cpu())

    print('Accuracy = ' ,accuracy_score(y_true, y_pred))  
    if plt_cf:
            eval_viz(y_true, y_pred, ['C'+str(i) for i in range(max(y_true+[1]))], exp_id)
    return y_true, y_pred, y_logits, np.concatenate(feats,axis=0)#, all_x
def eval_viz(y_true, y_pred, target_names, title):
    acc = accuracy_score(y_true, y_pred)
    cf_mat = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cf_mat,
                              target_names,
                              title=title,
                              cmap=None,
                              normalize=True)
    # print(cf_mat)
    print(title, acc)
import h5py
def get_vae_data(txt_data_dir,files):
    dataset = {"images": [], "labels": []}
    
    for line in files:
        file = h5py.File(txt_data_dir+'/{}.hdf5'.format(line), 'r')
        if len(dataset["images"]) == 0:
            dataset["images"] = file["images"][...]
            dataset["labels"] = file["labels"][...]
        else:
            dataset["images"] = np.concatenate([dataset["images"][...], file["images"][...]], axis=0)
            dataset["labels"] = np.concatenate([dataset["labels"][...], file["labels"][...]], axis=0)
        file.close()
    
    dataset_ = dataset
    
    data, targets = dataset_["images"].transpose([0,2,3,1]), dataset_["labels"]
    data, targets = data.astype(np.uint8), targets.astype(np.int64)
    #X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
    return data, targets
def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'

    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c': 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]

    else:
        if net_name == 'WideResNet':
            import models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'WideResNetVar':
            import models.nets.wrn_var as net
            builder = getattr(net, 'build_WideResNetVar')()
        elif net_name == 'ResNet50':
            import models.nets.resnet50 as net
            builder = getattr(net, 'build_ResNet50')(is_remix)
        else:
            assert Exception("Not Implemented Error")

        if net_name != 'ResNet50':
            setattr_cls_from_kwargs(builder, net_conf)
        return builder.build


def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
