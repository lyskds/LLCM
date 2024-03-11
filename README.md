# LLCM
**</h2>Leapfrog Latent Consistency Model for Medical Images Generation</h2>**

The scarcity of accessible medical image data poses a significant obstacle in effectively training deep learning models for medical diagnosis, as hospitals refrain from sharing their data due to privacy concerns. In response, we gathered a diverse dataset named MedImgs, which comprises over 250,127 images spanning 61 disease types and 159 classes of both humans and animals from open-source repositories. We propose a Leapfrog Latent Consistency Model (LLCM) that is distilled from a retrained diffusion model based on the collected MedImgs dataset, which enables our model to generate real-time high-resolution images. You may find belwo some sample medical images generated by our LLCM at several inference steps

![image](https://github.com/lyskds/LLCM/assets/162650359/6192f441-50e2-453c-a966-36c54f77c32b)

**</h2>How to fine-tune with your dataset</h2>**
- Install these libraries: 
pip install diffusers transformers accelerate
- You may launch this script 'train_text_to_img_llcm_launch.py' after loading our model weight and train it to obtain the fine-tuned weight.
  
**</h2>How to generate image with fine-tuned weight</h2>**
- You may run 'inference.py' after loading the fine-tuned weight to genetate images by specifying the prompt.
