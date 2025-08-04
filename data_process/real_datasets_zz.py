import os
import pickle as pkl
import torch
import numpy as np
import rawpy
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset
from utils.utils import dataload, log
from .process import raw_wb_aug, get_aug_param_torch, get_camera_noisy_params_max


class RealBase_Dataset(Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        
        self.clip_low = 0 if args['clip_low'] else float('-inf')
        self.clip_high = 1 if args['clip_high'] else float('inf')
        # self.initialization() # 基类不调用
    
    def default_args(self):
        self.args = {}
        self.args['crop_per_image'] = 8
        self.args['crop_size'] = 512
        self.args['ori'] = False
        self.args['dstname'] = 'SID'
        self.args['camera_type'] = 'SonyA7S2'
        self.args['mode'] = 'train'
        self.args['command'] = ''
        self.args['wp'] = 16383
        self.args['bl'] = 512
    
    def initialization(self):
        self.suffer = 'ARW'
        self.dataset_file = f'SID_{self.args["mode"]}.info'
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos = pkl.load(info_file)
            print(f">> Sucessfully Load {self.dataset_file} (Length: {len(self.infos)})")
        
        self.length = len(self.infos)
        self.get_shape()
        self.get_darkshading_infos()
    
    def lr_idremap_table_init(self):
        self.lr_idremap_table = [None] * self.length
        for idx in range(len(self.infos)):
            self.get_lr_id(idx)
        log('Successfully finish id_remap')

    def get_lr_id(self, idx):
        if 'idremap' in self.args["command"]:
            # 没有的话就构建remap table
            if self.lr_idremap_table[idx] is None:
                ratio_dict = {}
                for i, ratio in enumerate(self.infos[idx]['ratio']):
                    if ratio not in ratio_dict:
                        ratio_dict[ratio] = [i]
                    else:
                        ratio_dict[ratio].append(i)
                self.lr_idremap_table[idx] = []
                for ratio in ratio_dict:
                    self.lr_idremap_table[idx].append(ratio_dict[ratio])
            
            # 选择100, 250, 300
            ratio_id = np.random.randint(len(self.lr_idremap_table[idx]))
            id = np.random.randint(len(self.lr_idremap_table[idx][ratio_id]))
            lr_id = self.lr_idremap_table[idx][ratio_id][id]
        else:
            lr_id = np.random.randint(len(self.infos[idx]['ratio']))
        return lr_id

    def __len__(self):
        return self.length
    
    def get_darkshading_infos(self):
        ds_folder_path = f"./{self.args['ds_dir']}/{self.args['camera_type']}/"
        with open(os.path.join(ds_folder_path, "darkshading_BLE.pkl"), "rb") as f:
            self.pmn_ble = pkl.load(f)
        self.pmn_dsk_high = np.load(os.path.join(ds_folder_path, "darkshading_highISO_k.npy"))
        self.pmn_dsk_low = np.load(os.path.join(ds_folder_path, "darkshading_lowISO_k.npy"))
        self.pmn_dsb_high = np.load(os.path.join(ds_folder_path, "darkshading_highISO_b.npy"))
        self.pmn_dsb_low = np.load(os.path.join(ds_folder_path, "darkshading_lowISO_b.npy"))
        self.darkframe_num_dict = np.loadtxt("./darkframes/darkframenum.txt", delimiter=',', skiprows=1, dtype=int)
        self.darkframe_num_dict = {iso: num for iso, num in self.darkframe_num_dict}
    
    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def init_random_crop_point(self, mode='non-overlapped', raw_crop=False):
        self.h_start = []
        self.w_start = []
        self.h_end = []
        self.w_end = []

        self.aug = np.random.randint(4, size=self.args['crop_per_image'])
        h, w = self.h, self.w

        if raw_crop:
            h, w = self.H, self.W
        if mode== 'non-overlapped':
            nh = h // self.args['patch_size']
            nw = w // self.args['patch_size']
            h_start = np.random.randint(0, h - nh * self.args['patch_size'] + 1)
            w_start = np.random.randint(0, w - nw * self.args['patch_size'] + 1)

            for i in range(nh):
                for j in range(nw):
                    self.h_start.append(h_start + i * self.args['patch_size'])
                    self.w_start.append(w_start + j * self.args['patch_size'])
                    self.h_end.append(h_start + (i + 1) * self.args['patch_size'])
                    self.w_end.append(w_start + (j + 1) * self.args['patch_size'])
        else: # random crop
            for i in range(self.args['crop_per_image']):
                h_start = np.random.randint(0, h - self.args['patch_size'] + 1)
                w_start = np.random.randint(0, w - self.args['patch_size'] + 1)
                self.h_start.append(h_start)
                self.w_start.append(w_start)
                self.h_end.append(h_start + self.args['patch_size'])
                self.w_end.append(w_start + self.args['patch_size'])
    
    def data_aug(self, data, mode=0):
        if mode == 0: 
            return data
        
        rot = mode % 2
        flip = mode // 2
        if rot:
            data = np.rot90(data, k=2, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]

        return data
        
    def eval_crop(self, data, base=64):
        crop_size = self.args['patch_size']
        d = base // 2
        l = crop_size - base
        nh = self.h // l + 1
        nw = self.w // l + 1
        data = F.pad(data, (d, d, d, d), mode='reflect')
        croped_data = torch.empty((nh, nw, self.c, crop_size, crop_size), dtype=data.dtype, device=data.device)
        # chunk main part
        for i in range(nh - 1):
            for j in range(nw - 1):
                croped_data[i][j] = data[..., i * l:i * l + crop_size, j * l: j * l + crop_size]
        # pad side
        for i in range(nh-1):
            j = nw - 1
            croped_data[i][j] = data[..., i*l:i*l+crop_size,-crop_size:]
        for j in range(nw-1):
            i = nh - 1
            croped_data[i][j] = data[..., -crop_size:,j*l:j*l+crop_size]
        # pad angle
        croped_data[nh-1][nw-1] = data[..., -crop_size:,-crop_size:]
        # 整合为tensor
        croped_data = croped_data.view(-1, self.c, crop_size, crop_size)

        return croped_data

    def eval_merge(self, croped_data, base=64):
        crop_size = self.args["patch_size"]
        data = torch.empty((1, self.c, self.h, self.w), dtype=croped_data.dtype, device=croped_data.device)
        # crop setting
        d = base//2
        l = crop_size - base
        nh = self.h // l + 1
        nw = self.w // l + 1
        croped_data = croped_data.view(nh, nw, self.c, crop_size, crop_size)
        # 分块crop主体区域
        for i in range(nh-1):
            for j in range(nw-1):
                data[..., i*l:i*l+l,j*l:j*l+l] = croped_data[i, j, :, d:-d, d:-d]
        # 补边
        for i in range(nh-1):
            j = nw - 1
            data[..., i*l:i*l+l, -l:] = croped_data[i, j, :, d:-d, d:-d]
        for j in range(nw-1):
            i = nh - 1
            data[..., -l:, j*l:j*l+l] = croped_data[i, j, :, d:-d, d:-d]
        # 补角
        data[..., -l:, -l:] = croped_data[nh-1, nw-1, :, d:-d, d:-d]
        
        return data
    
    def random_crop(self, img):
        # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
        c, h, w = img.shape
        # 创建空numpy做画布, [crops, h, w]
        crops = np.empty((self.args["crop_per_image"], c, self.args["patch_size"], self.args["patch_size"]), dtype=np.float32)
        # 往空tensor的通道上贴patchs
        for i in range(self.args["crop_per_image"]):
            crop = img[:, self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i]]
            crop = self.data_aug(crop, mode=self.aug[i])
            crops[i] = crop

        return crops

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=0)
        out = (out - self.args['bl']) / (self.args['wp'] - self.args['bl']) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def get_darkshading(self, iso):
        if iso <= 1600:
            return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
        else:
            return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]
    

class SIDSysTrainDataset(RealBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()

        self.cache = {}
        for idx in tqdm(range(len(self.infos))):
            hr_raw = np.array(rawpy.imread(self.infos[idx]["long"]).raw_image_visible).astype(np.float32)
            self.cache[idx] = hr_raw
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
        self.length = len(self.infos)

    def systhesis_shot_noise(self, img):
        K = np.random.uniform(0., 1.) * (25.6 - 0.05) + 0.05

        return np.random.poisson(img / K) * K
    
    def __getitem__(self, idx): 
        # load data
        hr_raw = self.cache[idx]

        # get random dgain
        dgain = np.random.randint(100, 301)
        iso = self.infos[idx]['ISO']

        # get dark frame
        ds = self.get_darkshading(iso)
        sample_id = np.random.randint(1, self.darkframe_num_dict[iso])
        sample_dark_frame = np.load(os.path.join(self.args['darkframe_path'], f"ISO{iso}/{sample_id:04d}_ISO{iso}.npy"))

        # get shot noise
        shot_noise = self.systhesis_shot_noise(hr_raw / dgain)
        lr_raw = shot_noise + sample_dark_frame - ds
        
        lr_raw = lr_raw - ds
        # SID配对数据训练的时候都要减
        hr_raw = hr_raw - ds

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [c, h, w]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [c, h, w]

        self.init_random_crop_point(mode=self.args['croptype'])
        hr_crops = self.random_crop(hr_imgs)
        lr_crops = self.random_crop(lr_imgs)

        lr_crops = torch.FloatTensor(lr_crops)
        hr_crops = torch.FloatTensor(hr_crops)
        lr_crops *= dgain

        data = {
            "name": f"{self.infos[idx]['name'][:5]}_{self.infos[idx]['ratio']}",
            "wb": torch.FloatTensor(self.infos[idx]['wb']),
            "ccm": torch.FloatTensor(self.infos[idx]['ccm']),
            "iso": self.infos[idx]['ISO'],
            "ExposureTime": self.infos[idx]['ExposureTime'],
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * dgain,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data


class SID_Dataset(RealBase_Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
        if self.args['mode'] == 'train':
            self.length = len(self.infos)
            self.lr_idremap_table_init()
        else:
            self.evaltest_remap()
            self.change_eval_ratio(ratio=250)
            self.length = len(self.infos)

    def evaltest_remap(self):
        self.infos_all = [self.infos[:40], self.infos[40:80], self.infos[80:]]
        # 镀一层包装，就不用getitem改代码了
        for rid in range(3):
            for i in range(len(self.infos_all[rid])):
                self.infos_all[rid][i]['short'] = [self.infos_all[rid][i]['short']]
                self.infos_all[rid][i]['ratio'] = [self.infos_all[rid][i]['ratio']]

    def change_eval_ratio(self, idx=None, ratio=None):
        ratio_list = [100, 250, 300]
        assert idx is not None or ratio is not None, 'Check please!'
        if idx is not None:
            assert idx in [0,1,2], 'idx must in [0,1,2]'
            ratio = ratio_list[idx]
        elif ratio is not None:
            assert int(ratio) in ratio_list, 'ratio must in [100,250,300]'
            idx = int(ratio) // 100 - 1

        self.infos = self.infos_all[idx]
        self.length = len(self.infos)
        log(f'Eval ratio {ratio}')
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = torch.FloatTensor(self.infos[idx]['wb'])
        data['ccm'] = torch.FloatTensor(self.infos[idx]['ccm'])
        data['name'] = f"{self.infos[idx]['name'][:5]}_{self.infos[idx]['ratio']}"
        data['ISO'] = self.infos[idx]['ISO']
        data['ExposureTime'] = self.infos[idx]['ExposureTime']
        
        hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H, self.W)
        lr_id = self.get_lr_id(idx) if self.args['mode'] == 'train' else 0
        lr_raw = np.array(dataload(self.infos[idx]['short'][lr_id])).reshape(self.H, self.W)
        data['ratio'] = self.infos[idx]['ratio'][lr_id]

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(data['ISO'])
            if 'darkshading2' in self.args['command'] and self.args["mode"] == 'train':
                # SID配对数据训练的时候都要减
                hr_raw = hr_raw - self.get_darkshading(data['ISO'])
                # lr_raw += np.random.randn() * self.noiseparam[data['ISO']]['biassig']
        elif 'blc' in self.args['command'] and 'HB' not in self.args['command']:
            # 只使用均值矫正
            lr_raw = lr_raw - self.pmn_ble[data['ISO']]

        # pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [c, h, w]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [c, h, w]

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'])
            hr_crops = self.random_crop(hr_imgs)
            lr_crops = self.random_crop(lr_imgs)
        else:
            hr_crops = hr_imgs
            lr_crops = lr_imgs

        lr_crops = torch.FloatTensor(lr_crops).unsqueeze(0)
        hr_crops = torch.FloatTensor(hr_crops).unsqueeze(0)
        lr_crops *= data['ratio']

        data['ratio'] = torch.ones(hr_crops.shape[0]) * data['ratio']
        data['rgb_gain'] = torch.ones(hr_crops.shape[0])
        data["lr"] = torch.clamp(lr_crops, self.clip_low, self.clip_high)
        data["hr"] = torch.clamp(hr_crops, 0, 1)
        
        return data


class ELD_Dataset(RealBase_Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()
        self.args['ori'] = False
        self.args['dstname'] = 'ELD'
        self.args['mode'] = 'eval'

    def initialization(self):
        # 获取数据地址
        self.suffer = 'ARW'
        self.dataset_file = f'ELD_SonyA7S2.info'
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos = pkl.load(info_file)
            print(f'>> Successfully load "{self.dataset_file}" (Length: {len(self.infos)})')
        self.iso_list = self.args['iso_list']
        self.ratio_list = self.args['ratio_list']
        self.imgs_per_scene = len(self.iso_list) * len(self.ratio_list)
        self.length = len(self.infos) * len(self.iso_list) * len(self.ratio_list)
        self.get_shape()
        self.get_darkshading_infos()

    def __len__(self):
        return self.length

    def get_raw_id(self, scene_id, iso, ratio):
        for i in range(len(self.infos[scene_id])):
            raw_iso = self.infos[scene_id][i]['ISO']
            raw_ratio = self.infos[scene_id][i]['ratio']
            if raw_iso == iso and raw_ratio == ratio:
                img_id = i + 1
                break
        # 就近选gt
        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(img_id - gt_ids))
        gt_id = gt_ids[ind]
        return img_id-1, gt_id-1

    def fast_eval(self, on=True):
        if on:
            # self.iso_list = self.args['iso_list'][-1:]
            self.infos_backup = self.infos.copy()
            self.infos = [self.infos[-3], self.infos[-1]]
            self.ratio_list = self.args['ratio_list'][-1:]
            self.recheck_length()
        else:
            self.infos = self.infos_backup.copy()
            self.iso_list = self.args['iso_list']
            self.ratio_list = self.args['ratio_list']
            self.recheck_length()

    def recheck_length(self):
        self.imgs_per_scene = len(self.iso_list) * len(self.ratio_list)
        self.length = len(self.infos) * len(self.iso_list) * len(self.ratio_list)

    def __getitem__(self, idx):
        data = {}
        # 划分数据，get id
        scene_id = idx // self.imgs_per_scene
        img_idx = idx % self.imgs_per_scene
        iso_idx = img_idx // len(self.ratio_list)
        ratio_idx = img_idx % len(self.ratio_list)
        data['ISO'] = self.iso_list[iso_idx]
        data['ratio'] = self.ratio_list[ratio_idx]
        lr_id, hr_id = self.get_raw_id(scene_id, data['ISO'], data['ratio'])
        # 读取数据
        data['wb'] = torch.FloatTensor(self.infos[scene_id][hr_id]['wb'])
        data['ccm'] = torch.FloatTensor(self.infos[scene_id][hr_id]['ccm'])
        data['name'] = f"scene-{scene_id+1:02d}_{self.infos[scene_id][lr_id]['name']}"
        data['ExposureTime'] = self.infos[scene_id][hr_id]['ExposureTime']
        
        hr_raw = np.array(dataload(self.infos[scene_id][hr_id]['data'])).reshape(self.H,self.W)
        lr_raw = np.array(dataload(self.infos[scene_id][lr_id]['data'])).reshape(self.H,self.W)

        lr_raw = lr_raw - self.get_darkshading(data['ISO'])

        # transfer data
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [c, h, w]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [c, h, w]

        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0)

        lr_crops *= data['ratio']

        data['ratio'] = torch.ones(hr_crops.shape[0]) * data['ratio']
        data['rgb_gain'] = torch.ones(hr_crops.shape[0])
        data["lr"] = torch.clamp(lr_crops, self.clip_low, self.clip_high)
        data["hr"] = torch.clamp(hr_crops, 0, 1)
        
        return data
    

class SIDEvalDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wl, self.bl = args['wp'], args['bl']
        self.clip_low = 0 if args['clip_low'] else float("-inf")
        self.clip_high = 1 if args['clip_high'] else float("inf")

        ## load pmn's darkshading
        with open(f"./resources/SonyA7S2/darkshading_BLE.pkl", "rb") as f:
            self.pmn_ble = pkl.load(f)
        self.pmn_dsk_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_k.npy")
        self.pmn_dsk_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_k.npy")
        self.pmn_dsb_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_b.npy")
        self.pmn_dsb_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_b.npy")

        ## format data pairs
        with open(f"./infos/SID_evaltest.info", "rb") as info_file:
            self.data_info = pkl.load(info_file)
        self.evaltest_remap()

        # self.cache = {}
        # for idx in tqdm(range(len(self.data_info))):
        #     hr_raw = np.array(rawpy.imread(self.data_info[idx]["long"]).raw_image_visible).astype(np.float32)
        #     lr_id = 0
        #     lr_raw = np.array(rawpy.imread(self.data_info[idx]["short"][lr_id]).raw_image_visible).astype(np.float32)
        #     self.cache[idx] = (hr_raw, lr_raw, lr_id)

    def __len__(self):
        return len(self.data_info)

    def get_darkshading(self, iso):
        if iso <= 1600:
            return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
        else:
            return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]

    def lr_idremap_table_init(self):
        self.lr_idremap_table = [None] * len(self.data_info)
        for idx in range(len(self.data_info)):
            self.get_lr_id(idx)

    def get_lr_id(self, idx):
        if self.lr_idremap_table[idx] is None:
            ratio_dict = {}
            for i, ratio in enumerate(self.data_info[idx]["ratio"]):
                if ratio not in ratio_dict:
                    ratio_dict[ratio] = [i]
                else:
                    ratio_dict[ratio].append(i)
            self.lr_idremap_table[idx] = []
            for ratio in ratio_dict:
                self.lr_idremap_table[idx].append(ratio_dict[ratio])

        # 选择100, 250, 300
        ratio_id = np.random.randint(len(self.lr_idremap_table[idx]))
        id = np.random.randint(len(self.lr_idremap_table[idx][ratio_id]))
        lr_id = self.lr_idremap_table[idx][ratio_id][id]
        return lr_id

    def evaltest_remap(self):
        self.data_info_all = [self.data_info[:40], self.data_info[40:80], self.data_info[80:]]
        # wrapping to avoid change getitem
        for rid in range(3):
            for i in range(len(self.data_info_all[rid])):
                self.data_info_all[rid][i]["short"] = [self.data_info_all[rid][i]["short"]]
                self.data_info_all[rid][i]["ratio"] = [self.data_info_all[rid][i]["ratio"]]
    
    def change_eval_ratio(self, idx=None, ratio=None):
        ratio_list = [100, 250, 300]
        assert idx is not None or ratio is not None, 'Check please!'
        if idx is not None:
            assert idx in [0,1,2], 'idx must in [0,1,2]'
            ratio = ratio_list[idx]
        elif ratio is not None:
            assert int(ratio) in ratio_list, 'ratio must in [100,250,300]'
            idx = int(ratio) // 100 - 1

        self.data_info = self.data_info_all[idx]
        self.length = len(self.data_info)
        log(f'Eval ratio {ratio}')

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def __getitem__(self, idx):
        ## load data
        hr_raw = np.array(rawpy.imread(self.data_info[idx]["long"]).raw_image_visible).astype(np.float32)
        lr_id = 0
        lr_raw = np.array(rawpy.imread(self.data_info[idx]["short"][lr_id]).raw_image_visible).astype(np.float32)
        dgain = self.data_info[idx]["ratio"][lr_id]

        ## subtract dark shading
        lr_raw = lr_raw - self.get_darkshading(self.data_info[idx]["ISO"])

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [h, w, c]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]
        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0).permute(0, 3, 1, 2)

        lr_crops *= dgain

        data = {
            "name": f"{self.data_info[idx]['name'][:5]}_{self.data_info[idx]['ratio']}",
            "wb": torch.FloatTensor(self.data_info[idx]["wb"]),
            "ccm": torch.FloatTensor(self.data_info[idx]["ccm"]),
            "ISO": self.data_info[idx]["ISO"],
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * dgain,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data


class ELDPairEvalDataset(Dataset):
    def __init__(self, wl=16383, bl=512, clip_low=False, clip_high=True, eval_ratio=100):
        super().__init__()
        self.wl, self.bl = wl, bl
        self.clip_low = 0 if clip_low else float("-inf")
        self.clip_high = 1 if clip_high else float("inf")
        self.eval_ratio = eval_ratio
        self.iso_list = [800, 1600, 3200]

        ## load pmn's darkshading
        with open(f"./resources/SonyA7S2/darkshading_BLE.pkl", "rb") as f:
            self.pmn_ble = pkl.load(f)
        self.pmn_dsk_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_k.npy")
        self.pmn_dsk_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_k.npy")
        self.pmn_dsb_high = np.load(f"./resources/SonyA7S2/darkshading_highISO_b.npy")
        self.pmn_dsb_low = np.load(f"./resources/SonyA7S2/darkshading_lowISO_b.npy")

        ## data
        with open("infos/ELD_SonyA7S2.info", "rb") as info_file:
            self.data_info = pkl.load(info_file)

    def __len__(self):
        return len(self.data_info) * len(self.iso_list)

    def get_darkshading(self, iso):
        if iso <= 1600:
            return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
        else:
            return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def get_raw_id(self, scene_id, iso):
        for i in range(len(self.data_info[scene_id])):
            raw_iso = self.data_info[scene_id][i]["ISO"]
            if raw_iso == iso and self.eval_ratio == self.data_info[scene_id][i]["ratio"]:
                img_id = i + 1
                break
        gt_ids = np.array([1, 6, 11, 16])
        gt_id = gt_ids[np.argmin(np.abs(img_id - gt_ids))]
        return img_id - 1, gt_id - 1

    def __getitem__(self, idx):
        scene_idx, iso_idx = idx // len(self.iso_list), idx % len(self.iso_list)
        lr_id, hr_id = self.get_raw_id(scene_idx, self.iso_list[iso_idx])
        hr_raw = np.array(rawpy.imread(self.data_info[scene_idx][hr_id]["data"]).raw_image_visible).astype(np.float32)
        lr_raw = np.array(rawpy.imread(self.data_info[scene_idx][lr_id]["data"]).raw_image_visible).astype(np.float32)

        ## subtract dark shading
        lr_raw = lr_raw - self.get_darkshading(self.iso_list[iso_idx])

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [h, w, c]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]
        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        lr_crops *= self.eval_ratio

        data = {
            "name": f"scene-{idx+1:02d}_{self.data_info[scene_idx][lr_id]['name']}",
            "wb": torch.FloatTensor(self.data_info[scene_idx][hr_id]["wb"]),
            "ccm": torch.FloatTensor(self.data_info[scene_idx][hr_id]["ccm"]),
            "iso": torch.ones(hr_crops.shape[0]) * self.iso_list[iso_idx],
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * self.eval_ratio,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data


class LRIDEvalDataset(Dataset):
    def __init__(
        self, wl=1023, bl=64, clip_low=False, clip_high=True, ratio_list=[1, 2, 4, 8, 16], dataset_name="indoor_x3"
    ):
        super().__init__()
        self.wl, self.bl = wl, bl
        self.clip_low = 0 if clip_low else float("-inf")
        self.clip_high = 1 if clip_high else float("inf")
        self.dataset_name = dataset_name
        self.ratio_list = ratio_list
        self.iso = 6400
        self.darkshading, self.darkshading_hot = {}, {}
     
        ## format data pairs
        with open(f"infos/{dataset_name}_GT_align_ours.info", "rb") as info_file:
            self.infos_gt = pkl.load(info_file)
        with open(f"infos/{dataset_name}_short.info", "rb") as info_file:
            self.infos_short = pkl.load(info_file)
        self.data_info = self.infos_gt
        for i in range(len(self.data_info)):
            self.data_info[i]["hr"] = self.data_info[i]["data"]
            self.data_info[i]["lr"] = {dgain: self.infos_short[dgain][i] for dgain in self.infos_short}
            del self.data_info[i]["data"]

        self.id_remap = self.data_split()

    def __len__(self):
        return len(self.id_remap) * len(self.ratio_list)

    def data_split(self, eval_ids=None):
        id_remap = list(range(len(self.data_info)))
        if self.dataset_name == "indoor_x5":
            eval_ids = [4, 14, 25, 41, 44, 51, 52, 53, 58]
        elif self.dataset_name == "outdoor_x3":
            eval_ids = [9, 21, 22, 32, 44, 51]
        else:
            eval_ids = []

        id_remap = eval_ids
        return id_remap

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def hot_check(self, idx):
        if self.dataset_name == "indoor_x5":
            hot_ids = [6, 15, 33, 35, 39, 46, 37, 59]
        elif self.dataset_name == "outdoor_x3":
            hot_ids = [0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 26, 30, 51, 52, 54, 55, 56]
        else:
            raise NotImplementedError
        return True if idx in hot_ids else False

    def get_bias(self, iso=6400, exp=30, hot=False):
        if hot:
            bias = self.blc_mean_hot[iso][:, 0] * exp + self.blc_mean_hot[iso][:, 1]  # RGGB: (4,)
        else:
            bias = self.blc_mean[iso][:, 0] * exp + self.blc_mean[iso][:, 1]  # RGGB: (4,)
        return bias

    def blc_rggb(self, raw, bias):
        def _bayer2rggb(bayer):
            H, W = bayer.shape
            return bayer.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)

        def _rggb2bayer(rggb):
            H, W, _ = rggb.shape
            return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H * 2, W * 2)

        return _rggb2bayer(_bayer2rggb(raw) + bias.reshape(1, 1, 4))

    def get_darkshading(self, iso=6400, hot=False):
        if iso not in self.darkshading:
            self.darkshading[iso] = np.load(f"./resources/IMX686/ds_{iso}.npy")
            self.darkshading_hot[iso] = np.load(f"./resources/IMX686/ds_{iso}_hot.npy")
        
        ds = self.darkshading_hot[iso] if hot else self.darkshading[iso]
        return ds

    def __getitem__(self, idx):
        dgain = self.ratio_list[idx // len(self.id_remap)]

        idr = self.id_remap[idx % len(self.id_remap)]
        hr_raw = np.load(self.data_info[idr]["hr"])
        lr_id = 0
        lr_raw = np.array(rawpy.imread(self.data_info[idr]["lr"][dgain]["data"][lr_id]).raw_image_visible)
        hr_raw, lr_raw = hr_raw.astype(np.float32), lr_raw.astype(np.float32)

        ## subtract dark shading
        lr_raw = lr_raw - self.get_darkshading(
            iso=self.iso,
            hot=self.hot_check(int(self.data_info[idr]["name"][-3:])),
        )

        ## pack to 4-chans
        lr_imgs = self.pack_raw(lr_raw, norm=True, clip=False)  ## [h, w, c]
        hr_imgs = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]

        ## augmentation and crop to patches
        lr_crops = torch.FloatTensor(lr_imgs).unsqueeze(0).permute(0, 3, 1, 2)
        hr_crops = torch.FloatTensor(hr_imgs).unsqueeze(0).permute(0, 3, 1, 2)

        lr_crops *= dgain

        data = {
            "name": f"{self.data_info[idr]['name']}_x{dgain:02d}",
            "wb": torch.FloatTensor(self.data_info[idr]["wb"]),
            "ccm": torch.FloatTensor(self.data_info[idr]["ccm"]),
            "iso": torch.ones(hr_crops.shape[0]) * self.iso,
            "rgb_gain": torch.ones(hr_crops.shape[0]),
            "ratio": torch.ones(hr_crops.shape[0]) * dgain,
            "lr": torch.clamp(lr_crops, self.clip_low, self.clip_high),
            "hr": torch.clamp(hr_crops, 0, 1),
        }

        return data