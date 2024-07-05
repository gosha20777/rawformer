from rawformer.base.weight_init import init_weights
from rawformer.torch.funcs      import prepare_model
from rawformer.torch.spectr_norm import apply_sn

def default_model_init(model, model_config, device):
    model = prepare_model(model, device)
    init_weights(model, model_config.weight_init)

    if model_config.spectr_norm:
        apply_sn(model)

    return model

