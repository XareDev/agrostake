# References

The project builds on public datasets, model architectures, and deployment
tools. The reference PDF that circulated with the original workspace is not
redistributed here; the DOI links below point to the canonical records.

## Dataset and plant-disease background

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
   plant health for the development of mobile and web applications.* Zenodo.
   <https://doi.org/10.5281/zenodo.1204916>

2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for
   image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
   <https://doi.org/10.3389/fpls.2016.01419>

The AgroStake classes (`Black Rot`, `ESCA`, `Healthy`, `Leaf Blight`) correspond
to the grape subset of PlantVillage. The exact augmented dataset used for the
reported experiments is not included; see [`../data/README.md`](../data/README.md).

## Model architecture

3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
   MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the
   IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*,
   4510–4520. <https://doi.org/10.1109/CVPR.2018.00474>

   MobileNetV2 is the teacher backbone in the preserved grayscale distillation
   notebook. The deployed RGB v2 and grayscale GAP models are custom
   convolutional networks, not MobileNetV2.

## Knowledge distillation

4. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a
   neural network. *NIPS 2014 Deep Learning Workshop*.
   <https://arxiv.org/abs/1503.02531>

5. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y.
   (2015). FitNets: Hints for thin deep nets. *International Conference on
   Learning Representations (ICLR)*. <https://arxiv.org/abs/1412.6550>

The preserved AgroStake KD recipe uses hard-label CE weight `alpha = 0.5`,
temperature `T = 4.0`, and a MobileNetV2 teacher. It is documented as a
negative result for the grayscale student; see [`results.md`](results.md).

## Deployment and quantisation

6. Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., &
   Kalenichenko, D. (2018). Quantization and training of neural networks for
   efficient integer-arithmetic-only inference. *CVPR*, 2704–2713.
   <https://doi.org/10.1109/CVPR.2018.00286>

7. Espressif Systems. *ESP-DL: Deep learning library for ESP32 series*.
   <https://github.com/espressif/esp-dl>
   Version used: `3.3.2`.

8. Espressif Systems. *ESP-IDF Programming Guide*.
   <https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/>

9. TensorFlow. *Post-training quantization*.
   <https://www.tensorflow.org/lite/performance/post_training_quantization>

10. TensorFlow. *TensorFlow Lite converter*.
    <https://www.tensorflow.org/lite/models/convert>

## Software and data licences

- ESP-IDF and ESP-DL: Apache License 2.0 (upstream).
- TensorFlow / TensorFlow Lite: Apache License 2.0 (upstream).
- PyTorch and torchvision: BSD-style licences (upstream).
- PlantVillage and derived datasets: upstream terms; see the dataset record.
- This repository's code: MIT (`LICENSE`).
- Documentation, figures, notebooks, and released model weights/metadata:
  CC BY-NC-SA 4.0 (`LICENSE-DOCS`). The weights are shared for research and
  non-commercial use; contact the authors for other terms.

## Citation of this repository

Use [`../CITATION.cff`](../CITATION.cff). A plain-text form:

```text
Harmouni, Y. (2026). AgroStake: Edge AI for Grapevine Disease
Detection on the ESP32-S3 (Version 0.1.1) [Computer software].
https://github.com/XareDev/agrostake
```
