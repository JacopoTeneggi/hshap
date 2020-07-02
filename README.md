# Interpretability
This repo contains a project done for a Biomedical Engineering Undergraduate Research course at the Johns Hopkins University.
For any questions, reach out to Alexandre Luster: alexandre.luster@epfl.ch.

## Aim 
Study and compare various methods to produce saliency maps on a synthetic dataset. These methods are ueful in understanding what parts of an image are important for predictions made by a neural network. 
Present a modified version of the SHAP method, which functions in a recursive and hierarchical manner to produce exact Shapley coefficients. 

## Content 

### Generating 
Contains an Ipython notebook to generate the desired dataset.
Once generated, you will have to upload the zip file to a drive account to go on with the training on colab. 
There are 3 rules to chose from: 
- 2: presence of a cross
- 3: presence of a red triangle
- 4: presence of a black circle and absence of crosses
For rule 2, a csv file is also generated which contains the ground truth positions of the centers of the crosses within an image. This allows to judge the quality of the saliency maps created with SHAP and Hierarchical Shapley.

### Hierarchical Shapley
The presented interpretation method is available in HierarchicalShapley.py. Example: 

    h = HierarchicalShap(net, background) # Initialize the model; net is the torch model to study and background a torch tensor representing background.
    mask = h.saliency_map(image, label, tolerance = [8,12], only_one_run = True, debug=False, max_depth = 30) # Generate the saliency map 
    
There are four different implementations. Saliency_map, saliency_map_optim_tol (improved for running the algorithm when judging with several thresholds), saliency_map_optim_rand (optimized for when using only_one_run = False, meaning computing the saliency map for the original input but also other shifted inputs) and saliency_map_optim_all (optimized for when using several thresholds and shifted inputs). 

In ShapleyComparison.ipynb, an Ipython notebook hosted on google colab, the different implementations are compared, and this module is compared to other shapley-based methods.

### Training 
The differet Ipython notebooks, hosted on google colab, are used to train a CNN to classify the different types of data. 

### Interpreting 

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

