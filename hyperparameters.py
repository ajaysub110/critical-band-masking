import os
import random

import numpy as np
from matplotlib import colors as mcolors
import torch
import torchvision.models as tvmodels
from vonenet import get_model
from torchvision import transforms

# set random seeds
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# set system and device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# set experiment constants
NOISE_SDS = [0, 0.02, 0.04, 0.08, 0.16]
SNRS = [0.625, 1.25, 2.5, 5, 10]
N_FREQS = 7
N_NOISES = len(NOISE_SDS)
CONTRAST = 0.2
IMAGENET_MEAN = 0.449
EPSILON = 0.1 # minimum gap between pixel limit and pixel value
IMAGE_SIZE = 224

LEAVE_OUT_DFS = ['3043', '3029']

# set paths
ROOT = os.path.dirname(os.path.realpath(__file__))
STIMULI_ROOT = os.path.join(ROOT, 'stimuli/CBMSplit')
GRAY_ROOT = os.path.join(ROOT, 'stimuli/GraySplit')

IMAGENET_ROOT = os.path.join(ROOT, 'stimuli/val')
IMAGENET_CLASSES = os.path.join(ROOT, 'stimuli/imagenet_classes.txt')
IMAGENET_SYNSETS = os.path.join(ROOT, 'stimuli/imagenet_synsets.txt')
BBOX_ANNS_ROOT = os.path.join(ROOT, "stimuli/bboxes_annotations")

MY_ROOT = ROOT # SET THIS TO YOUR ROOT
NETWORK_DATA_ROOT = os.path.join(MY_ROOT, 'data/network_data')
os.makedirs(NETWORK_DATA_ROOT, exist_ok=True)

# categories and mappings
CATEGORIES16 = ["airplane", 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 
	       'elephant', 'keyboard', 'knife', 'oven', 'truck']
CAT2SNET = {
    "knife" :    ["n03041632"],
    "keyboard" : ["n03085013", "n04505470"],
    "elephant" : ["n02504013", "n02504458"],
    "bicycle" :  ["n02835271", "n03792782"],
    "airplane" : ["n02690373", "n03955296", "n13861050", "n13941806"],
    "clock" :    ["n02708093", "n03196217", "n04548280"],
    "oven" :     ["n03259401", "n04111414", "n04111531"],
    "chair" :    ["n02791124", "n03376595", "n04099969", "n00605023", "n04429376"],
    "bear" :     ["n02132136", "n02133161", "n02134084", "n02134418"],
    "boat" :     ["n02951358", "n03344393", "n03662601", "n04273569", "n04612373", "n04612504"],
    "cat" :      ["n02122878", "n02123045", "n02123159", "n02126465", "n02123394", "n02123597", "n02124075", "n02125311"],
    "bottle" :   ["n02823428", "n03937543", "n03983396", "n04557648", "n04560804", "n04579145", "n04591713"],
    "truck" :    ["n03345487", "n03417042", "n03770679", "n03796401", "n00319176", "n01016201", "n03930630", "n03930777", "n05061003", "n06547832", "n10432053", "n03977966", "n04461696", "n04467665"],
    "car" :      ["n02814533", "n03100240", "n03100346", "n13419325", "n04285008"],
    "bird" :     ["n01321123", "n01514859", "n01792640", "n07646067", "n01530575", "n01531178", "n01532829", "n01534433", "n01537544", "n01558993", "n01562265", "n01560419", "n01582220", "n10281276", "n01592084", "n01601694", "n01614925", "n01616318", "n01622779", "n01795545", "n01796340", "n01797886", "n01798484", "n01817953", "n01818515", "n01819313", "n01820546", "n01824575", "n01828970", "n01829413", "n01833805", "n01843065", "n01843383", "n01855032", "n01855672", "n07646821", "n01860187", "n02002556", "n02002724", "n02006656", "n02007558", "n02009229", "n02009912", "n02011460", "n02013706", "n02017213", "n02018207", "n02018795", "n02025239", "n02027492", "n02028035", "n02033041", "n02037110", "n02051845", "n02056570"],
    "dog" :      ["n02085782", "n02085936", "n02086079", "n02086240", "n02086646", "n02086910", "n02087046", "n02087394", "n02088094", "n02088238", "n02088364", "n02088466", "n02088632", "n02089078", "n02089867", "n02089973", "n02090379", "n02090622", "n02090721", "n02091032", "n02091134", "n02091244", "n02091467", "n02091635", "n02091831", "n02092002", "n02092339", "n02093256", "n02093428", "n02093647", "n02093754", "n02093859", "n02093991", "n02094114", "n02094258", "n02094433", "n02095314", "n02095570", "n02095889", "n02096051", "n02096294", "n02096437", "n02096585", "n02097047", "n02097130", "n02097209", "n02097298", "n02097474", "n02097658", "n02098105", "n02098286", "n02099267", "n02099429", "n02099601", "n02099712", "n02099849", "n02100236", "n02100583", "n02100735", "n02100877", "n02101006", "n02101388", "n02101556", "n02102040", "n02102177", "n02102318", "n02102480", "n02102973", "n02104029", "n02104365", "n02105056", "n02105162", "n02105251", "n02105505", "n02105641", "n02105855", "n02106030", "n02106166", "n02106382", "n02106550", "n02106662", "n02107142", "n02107312", "n02107574", "n02107683", "n02107908", "n02108000", "n02108422", "n02108551", "n02108915", "n02109047", "n02109525", "n02109961", "n02110063", "n02110185", "n02110627", "n02110806", "n02110958", "n02111129", "n02111277", "n08825211", "n02111500", "n02112018", "n02112350", "n02112706", "n02113023", "n02113624", "n02113712", "n02113799", "n02113978"],
}

SNET2CAT = {}
for cat, v in CAT2SNET.items():
	for snet in v:
		SNET2CAT[snet] = CATEGORIES16.index(cat)

# Load class id to key mapping and synsets
with open(IMAGENET_CLASSES, 'r') as f:
	CLASS_ID_TO_KEY = f.readlines()
CLASS_ID_TO_KEY = [x.strip() for x in CLASS_ID_TO_KEY]

with open(IMAGENET_SYNSETS, 'r') as f:
	synsets = f.readlines()

# 1001 synsets, 0th is background
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
KEY_TO_CLASSNAME = {spl[0]:' '.join(spl[1:]) for spl in splits}

# for network analysis
IMAGE_TRANSFORMS = transforms.Compose([
	transforms.ToTensor(),
])

NETWORK_NAME = 'resnet50'
TV_NETWORK_NAMES = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
'inception_v3', 'mobilenet_v2', 'voneresnet50', 'squeezenet1_0', 'squeezenet1_1', 'shufflenet_v2_x0_5', 'mnasnet0_5', 
'vgg11_bn', 'vgg19_bn', 'wide_resnet50_2', 'resnext50_32x4d', 'densenet121', 'wide_resnet101_2', 'resnext101_32x8d', 
'densenet169', 'densenet201', 'mnasnet1_0', 'vgg13_bn', 'vgg16_bn']

# plotting constants
COLORS = list(mcolors.CSS4_COLORS.keys())[8:]

COLOR_DICT = {
	'alexnet': 'lightgray',
	'resnet18': 'lightgray',
	'resnet34': 'lightgray',
	'resnet50': 'lightgray',
	'resnet101': 'lightgray',
	'resnet152': 'lightgray',
	'inception_v3': 'lightgray',
	'mobilenet_v2': 'lightgray',
	'voneresnet50': 'tan',
	'squeezenet1_0': 'lightgray',
	'squeezenet1_1': 'lightgray',
	'shufflenet_v2_x0_5': 'lightgray',
	'mnasnet0_5': 'lightgray',
	'vgg11_bn': 'lightgray',
	'vgg19_bn': 'lightgray',
	'wide_resnet50_2': 'lightgray',
	'resnext50_32x4d': 'lightgray',
	'densenet121': 'lightgray',
	'wide_resnet101_2': 'lightgray',
	'resnext101_32x8d': 'lightgray',
	'densenet169': 'lightgray',
	'densenet201': 'lightgray',
	'mnasnet1_0': 'lightgray',
	'vgg13_bn': 'lightgray',
	'vgg16_bn': 'lightgray',
	'resnet50_trained_on_SIN': 'darkorchid',
	'resnet50_trained_on_SIN_and_IN': 'darkorchid',
	'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'darkorchid',
	'bagnet9': 'chocolate',
	'bagnet17': 'chocolate',
	'bagnet33': 'chocolate',
	'simclr_resnet50x1_supervised_baseline': 'darkgoldenrod',
	'simclr_resnet50x4_supervised_baseline': 'darkgoldenrod',
	'simclr_resnet50x1': 'goldenrod',
	'simclr_resnet50x2': 'goldenrod',
	'simclr_resnet50x4': 'goldenrod',
	'InsDis': 'goldenrod',
	'MoCo': 'goldenrod',
	'MoCoV2': 'goldenrod',
	'PIRL': 'goldenrod',
	'InfoMin': 'goldenrod',
	'resnet50_l2_eps0': 'lightcyan',
	'resnet50_l2_eps0_01': 'paleturquoise',
	'resnet50_l2_eps0_03': 'lightblue',
	'resnet50_l2_eps0_05': 'lightskyblue',
	'resnet50_l2_eps0_1': 'cornflowerblue',
	'resnet50_l2_eps0_25': 'dodgerblue',
	'resnet50_l2_eps0_5': 'royalblue',
	'resnet50_l2_eps1': 'blue',
	'resnet50_l2_eps3': 'mediumblue',
	'resnet50_l2_eps5': 'darkblue',
	'efficientnet_b0': 'darkslategray',
	'efficientnet_es': 'darkslategray',
	'efficientnet_b0_noisy_student': 'darkslategray',
	'efficientnet_l2_noisy_student_475': 'darkslategrey',
	'transformer_B16_IN21K': 'yellowgreen',
	'transformer_B32_IN21K': 'yellowgreen',
	'transformer_L16_IN21K': 'yellowgreen',
	'transformer_L32_IN21K': 'yellowgreen',
	'vit_small_patch16_224': 'yellowgreen',
	'vit_base_patch16_224': 'yellowgreen',
	'vit_large_patch16_224': 'yellowgreen',
	'cspresnet50': 'lightgray',
	'cspresnext50': 'lightgray',
	'cspdarknet53': 'lightgray',
	'darknet53': 'lightgray',
	'dpn68': 'tomato',
	'dpn68b': 'tomato',
	'dpn92': 'tomato',
	'dpn98': 'tomato',
	'dpn131': 'tomato',
	'dpn107': 'tomato',
	'hrnet_w18_small': 'coral',
	'hrnet_w18_small': 'coral',
	'hrnet_w18_small_v2': 'coral',
	'hrnet_w18': 'coral',
	'hrnet_w30': 'coral',
	'hrnet_w40': 'coral',
	'hrnet_w44': 'coral',
	'hrnet_w48': 'coral',
	'hrnet_w64': 'coral',
	'selecsls42': 'coral',
	'selecsls84': 'coral',
	'selecsls42b': 'coral',
	'selecsls60': 'coral',
	'selecsls60b': 'coral',
	'clip': '',
	'clipRN50': '',
	'resnet50_swsl': 'plum',
	'ResNeXt101_32x16d_swsl': 'plum',
	'BiTM_resnetv2_50x1': 'mediumpurple',
	'BiTM_resnetv2_50x3': 'mediumpurple',
	'BiTM_resnetv2_101x1': 'mediumpurple',
	'BiTM_resnetv2_101x3': 'mediumpurple',
	'BiTM_resnetv2_152x2': 'mediumpurple',
	'BiTM_resnetv2_152x4': 'mediumpurple',
	'resnet50_clip_hard_labels': 'lightgray',
	'resnet50_clip_soft_labels': 'lightgray',
	'swag_regnety_16gf_in1k': 'palevioletred',
	'swag_regnety_32gf_in1k': 'palevioletred',
	'swag_regnety_128gf_in1k': 'palevioletred',
	'swag_vit_b16_in1k': 'palevioletred',
	'swag_vit_l16_in1k': 'palevioletred',
	'swag_vit_h14_in1k': 'palevioletred',
}
