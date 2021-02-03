# Face Super Resolution Guided by 3D Facial Priors
The basic implementation of ECCV spotlight paper Face Super-Resolution Guided by 3D Facial Priors https://arxiv.org/pdf/2007.09454.pdf, please cite this paper if it is helpful for you. </br>
3D Priors Extraction: </br>
Simply one, Refer to the https://github.com/davisking/dlib </br>
Basic Model:we uploaded a basic model, but in the further, we will further polish this repository.   </br>

After obtaining five facial key landmarks, we move the bbox.txt into the dataset/[]/face. To get the 3D render priors, please enter into 3Dface_priors folder.   </br>

run demo.py to generate the 3D facial relevant vector under training_set/[videos_folder_list]/face/.  </br>
run demo_render.py to generate the rendered face results under dataset/[videos_folder_list]/face_render/. </br>

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
