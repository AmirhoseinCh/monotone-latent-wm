# Third-party code

This repository is released under the
[CC BY-NC 4.0](LICENSE) license. It includes code adapted from the following
projects, which keep their original copyright notices:

| Path | Source | License |
|---|---|---|
| `lmwm/models/AdaLN_vit.py`, `lmwm/models/modules.py`, `lmwm/models/tensors.py`, `lmwm/models/vit_predictor_wm.py` | [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) (modified) | CC BY-NC 4.0 |
| `lmwm/models/vit.py` | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) via jepa-wms, itself adapted from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) | MIT |
| `lmwm/data/` (RLDS pipeline: `dataset.py`, `obs_transforms.py`, `traj_transforms.py`, `spec.py`, `utils/`) | [octo-models/octo](https://github.com/octo-models/octo) and its PyTorch port | MIT |

Frozen image encoders are loaded at runtime and are not redistributed here:
[DINOv2](https://github.com/facebookresearch/dinov2) (Apache 2.0),
[DINOv3](https://github.com/facebookresearch/dinov3) (DINOv3 License),
[V-JEPA 2](https://github.com/facebookresearch/vjepa2).

## MIT license text (Octo)

```
MIT License

Copyright (c) 2023 Robotic AI & Learning Lab Berkeley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
