# Face Super Resolution Guided by 3D Facial Priors
![image](https://github.com/HUuxiaobin/Face-Super-Resolution-Guided-by-3D-Facial-Priors/blob/main/images/github_1.jpg)
State-of-the-art face super-resolution methods employ deep convolutional neural networks to learn a mapping between low- and high-resolution facial patterns by exploring local appearance knowledge. However, most of these methods do not well exploit facial structures and identity information, and struggle to deal with facial images that exhibit large pose variations. In this paper, we propose a novel face super-resolution method that explicitly incorporates 3D facial priors which grasp the sharp facial structures. Our work is the first to explore 3D morphable knowledge based on the fusion of parametric descriptions of face attributes (e.g., identity, facial expression, texture, illumination, and face pose). Furthermore, the priors can easily be incorporated into any network and are extremely efficient in improving the performance and accelerating the convergence speed. Firstly, a 3D face rendering branch is set up to obtain 3D priors of salient facial structures and identity knowledge. Secondly, the Spatial Attention Module is used to better exploit this hierarchical information (i.e., intensity similarity, 3D facial structure, and identity content) for the super-resolution problem. Extensive experiments demonstrate that the proposed 3D priors achieve superior face super-resolution results over the state-of-the-arts. <br>


The basic implementation of ECCV spotlight paper Face Super-Resolution Guided by 3D Facial Priors https://arxiv.org/pdf/2007.09454.pdf, please cite this paper if it is helpful for you. </br>
3D Priors Extraction: </br>
Simply one, Refer to the https://github.com/davisking/dlib </br>
Basic Model:we uploaded a basic model, but in the further, we will further polish this repository.   </br>

After obtaining five facial key landmarks, we move the bbox.txt into the dataset/[]/face. To get the 3D render priors, please enter into 3Dface_priors folder.   </br>

run demo.py to generate the 3D facial relevant vector under training_set/[videos_folder_list]/face/.  </br>
run demo_render.py to generate the rendered face results under dataset/[videos_folder_list]/face_render/. </br>

![image](https://github.com/HUuxiaobin/Face-Super-Resolution-Guided-by-3D-Facial-Priors/blob/main/images/github2.JPG)

Then,

run main.py to train face super-resolution models.


@inproceedings{hu2020face,
  title={Face super-resolution guided by 3d facial priors},
  author={Hu, Xiaobin and Ren, Wenqi and LaMaster, John and Cao, Xiaochun and Li, Xiaoming and Li, Zechao and Menze, Bjoern and Liu, Wei},
  booktitle={European Conference on Computer Vision},
  pages={763--780},
  year={2020},
  organization={Springer}
}
