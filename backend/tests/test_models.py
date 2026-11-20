import pytest
import torch

from ml.training.train import SUPPORTED_MODELS, build_model


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_every_model_builds_and_outputs_one_logit_per_class(model_name):
    model = build_model(5, model_name=model_name, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 5)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model(5, model_name="not_a_model", pretrained=False)