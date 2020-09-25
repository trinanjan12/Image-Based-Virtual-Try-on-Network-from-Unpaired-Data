import torch


def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'ov_pix2pixHD':
        from .ov_pix2pixHD_model import Pix2PixHDModel , InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'ov_pix2pixHD_online':
        from .ov_pix2pixHD_model_online import Pix2PixHDModel 
        if opt.isTrain:
            model = Pix2PixHDModel()
    else:
        from .ui_model import UIModel
        model = UIModel()

    print('--------- model used ---------',opt.model) #TODO  add type of the model with inference or train
    model.initialize(opt)

    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
