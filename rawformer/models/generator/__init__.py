from rawformer.base.networks    import select_base_generator
from rawformer.models.funcs     import default_model_init

from .vit       import ViTGenerator
from .vitunet   import ViTUNetGenerator
from .vitmodnet import ViTModNetGenerator
from .vitrawnet import ViTRawNetGenerator

def select_generator(name, **kwargs):
    if name == 'vit-v0':
        return ViTGenerator(**kwargs)

    if name == 'vit-unet':
        return ViTUNetGenerator(**kwargs)

    if name == 'vit-modnet':
        return ViTModNetGenerator(**kwargs)
    
    if name == 'vit-rawnet':
        return ViTRawNetGenerator(**kwargs)

    input_shape  = kwargs.pop('input_shape')
    output_shape = kwargs.pop('output_shape')

    assert input_shape == output_shape
    return select_base_generator(name, image_shape = input_shape, **kwargs)

def construct_generator(model_config, input_shape, output_shape, device):
    model = select_generator(
        model_config.model,
        input_shape  = input_shape,
        output_shape = output_shape,
        **model_config.model_args
    )

    return default_model_init(model, model_config, device)

