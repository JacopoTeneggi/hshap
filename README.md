# Interpretability
This repo contains my work for a Biomedical Engineering Undergraduate Research course at Johns Hopkins University.
## Aim 
Study and compare various methods to produce saliency maps on a synthetic dataset. These methods are ueful in understanding what parts of an image are important for predictions made by a neural network. 
Present a modified version of the SHAP method, which functions in a recursive and hierarchical manner to produce exact Shapley coefficients. 
## Content 
### Generating 
Contains a single ipynb, which can generate the desired dataset, which I would run on jupyter lab. 
Once generated, you will have to upload the zip file to a drive account to go on with the training on colab. 
### Training 
There are different ipynb, all on google colab, which correspond to the different datasets. 
### Interpreting 
### Hierarchical Shapley
### Utils 

## References 
### Repositories 
- SHAP: https://github.com/slundberg/shap
- Flashtorch: https://github.com/MisaOgura/flashtorch
- GradCAM and GradCAM++: https://github.com/vickyliin/gradcam_plus_plus-pytorch
- Smooth GradCAM++: https://github.com/yiskw713/SmoothGradCAMplusplus
### Publications 
#### General
- Adebayo J et al. (2018), Sanity Checks for Saliency Maps, [online], https://arxiv.org/abs/1810.03292
- Lundberg SM et al. (2018), Explainable machine-learning predictions for the prevention of hypoxaemia during surgery. Nature Biomedical Engineering 2(10):749–760, https://www.nature.com/articles/s41551-018-0304-0
- Glorot X & Yoshu B (2010), Understanding the difficulty of training deep feedforward neural networks, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, http://proceedings.mlr.press/v9/glorot10a.html
#### Gradient-based methods 
- Selvaraju RR et al. (2016), Grad-cam: Visual explanations from deep networks via gradient-based localization, [online],  https://arxiv.org/abs/1610.02391v3
- Chattopadhyay A et al. (2017), Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks, [online], https://arxiv.org/abs/1710.11063
- Smilkov D et al. (2017), SmoothGrad: removing noise by adding noise, https://arxiv.org/abs/1706.03825
- Chattopadhyay A et al. (2018), Grad-cam++: Generalized gradient-based visual explana-tions for deep convolutional networks, WACV, https://www.researchgate.net/publication/320727679_Grad-CAM_Generalized_Gradient-based_Visual_Explanations_for_Deep_Convolutional_Networks
#### Shapley-based methods 
- **Lundberg S & Lee SI (2017), A Unified Approach to Interpreting Model Predictions, NIPS, https://arxiv.org/abs/1705.07874**
- Lunberg S, Lee SI & Erion G (2018), Consistent Individualized Feature Attribution for Tree Ensembles, https://arxiv.org/abs/1802.03888 
- Chen J et al. (2018), L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data, [online], https://arxiv.org/abs/1808.02610
- Wang J et al. (2019), Shapley Q-value: A Local Reward Approach to Solve Global Reward Games, https://arxiv.org/abs/1907.05707

