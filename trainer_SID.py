import os
import time
from torch.optim import Adam, lr_scheduler
from data_process import *
from utils import *
from archs import *
from losses import *
from base_trainer import *
from archs.depthwise_separable_conv import DepthwiseSeparableConv, replace_conv3x3_with_depthwise, count_parameters

class SID_Trainer(Base_Trainer):
    def __init__(self):
        super().__init__()
        # model
        self.net = globals()[self.arch['name']](self.arch)
        
        # ===== 检查配置中是否启用深度可分离卷积 =====
        if self.arch.get('use_depthwise_separable', False):
            # print("启用深度可分离卷积...")
            original_params = count_parameters(self.net)
            print(f"原始参数量: {original_params:,}")
            
            # # 替换3x3卷积为深度可分离卷积
            # replace_conv3x3_with_depthwise(self.net)
            
            # # 统计替换后的参数量
            # new_params = count_parameters(self.net)
            # reduction = (original_params - new_params) / original_params * 100
            # print(f"替换后参数量: {new_params:,}")
            # print(f"参数减少: {reduction:.1f}%")
        # ===== 结束 =====

        # Raw2RGB
        if 'isp' in self.dst['command'].lower():
            self.arch_isp = self.args['arch_isp']
            self.isp = globals()[self.arch_isp['name']](self.arch_isp)
            model_path = os.path.join(f'{self.fast_ckpt}/ISP_CNN.pth')
            model_dict = torch.load(model_path, map_location=self.device)
            self.isp = load_weights(self.isp, model_dict, by_name=False)
            self.isp = self.isp.to(self.device)
            log('Use the ISP_CNN.pth from RViDeNet as ISP...')

        self.current_epoch = self.hyper['last_epoch']

        if torch.cuda.device_count() > 1:
            log("Using PyTorch's nn.DataParallel for multi-gpu...")
            self.multi_gpu = True
            self.net = nn.DataParallel(self.net)
        else:
            self.multi_gpu = False

        self.optimizer = Adam(self.net.parameters(), lr=self.hyper['learning_rate'])

        # Choose Learning Rate
        self.lr_lambda = self.get_lr_lambda_func()
        self.scheduler = LambdaScheduler(self.optimizer, self.lr_lambda)


        self.infos = None
        if self.mode=='train':
            self.dst_train = globals()[self.args['dst_train']['dataset']](self.args['dst_train'])
            self.dataloader_train = DataLoader(self.dst_train, batch_size=self.hyper['batch_size'], worker_init_fn=worker_init_fn,
                                    shuffle=True, num_workers=self.args['num_workers'], pin_memory=False)
            self.change_eval_dst('eval')
            self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)


        self.net = self.net.to(self.device)
        self.loss = Unet_Loss()
        # 添加感知损失
        if 'perceptual' in self.args['loss'] and self.args['loss']['perceptual']:
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        
        # 添加梯度损失
        if 'gradient' in self.args['loss'] and self.args['loss']['gradient']:
            self.gradient_loss = GradientLoss().to(self.device)
        self.corrector = IlluminanceCorrect()
        torch.backends.cudnn.benchmark = True
        # model log
        self.best_psnr = self.hyper['best_psnr'] if 'best_psnr' in self.hyper else 0
        last_eval_epoch = self.hyper['last_epoch'] // self.hyper['plot_freq']
        self.train_psnr = AverageMeter('PSNR', ':2f', last_epoch=self.hyper['last_epoch'])
        self.eval_psnr = AverageMeter('PSNR', ':2f', last_epoch=last_eval_epoch)
        self.eval_ssim = AverageMeter('SSIM', ':4f')
        self.eval_psnr_lr = AverageMeter('PSNR', ':2f')
        self.eval_ssim_lr = AverageMeter('SSIM', ':4f')
        self.eval_psnr_dn = AverageMeter('PSNR', ':2f')
        self.eval_ssim_dn = AverageMeter('SSIM', ':4f')

        # load weight
        if self.hyper['last_epoch']:    # 不是初始化
            try:
                # 优先尝试加载断点
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
                if not os.path.exists(model_path):
                    model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
                    
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 检查是否是完整的训练状态(新格式)还是仅模型权重(旧格式)
                if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                    # 加载完整训练状态
                    self.load_checkpoint(checkpoint)
                    log(f"从epoch {checkpoint['epoch']} 恢复训练状态")
                    # 如果加载的checkpoint与last_epoch不匹配，更新current_epoch
                    if checkpoint['epoch'] != self.hyper['last_epoch']:
                        log(f"注意: YML中的last_epoch为{self.hyper['last_epoch']}，已更新为checkpoint中的{checkpoint['epoch']}")
                        self.current_epoch = checkpoint['epoch']
                else:
                    # 仅加载模型权重(向后兼容)
                    self.net = load_weights(self.net, checkpoint, multi_gpu=self.multi_gpu, by_name=True)
                    log(f"已加载模型权重(仅参数), epoch={self.hyper['last_epoch']}")
            except Exception as e:
                log(f'无法加载checkpoint: {e}')
        else:
            log(f'Initializing {self.arch["name"]}...')
            # initialize_weights(self.net)

        self.logfile = f'./logs/log_{self.model_name}.log'
        log(f'Model Name:\t{self.model_name}', log=self.logfile, notime=True)
        log(f'Architecture:\t{self.arch["name"]}', log=self.logfile, notime=True)
        log(f'TrainDataset:\t{self.args["dst_train"]["dataset"]}', log=self.logfile, notime=True)
        log(f'EvalDataset:\t{self.args["dst_eval"]["dataset"]}', log=self.logfile, notime=True)
        log(f'CameraType:\t{self.dst["camera_type"]}', log=self.logfile, notime=True)
        log(f'num_channels:\t{self.arch["nf"]}', log=self.logfile, notime=True)
        log(f'BatchSize:\t{self.hyper["batch_size"]}', log=self.logfile, notime=True)
        log(f'PatchSize:\t{self.dst["patch_size"]}', log=self.logfile, notime=True)
        log(f'LearningRate:\t{self.hyper["learning_rate"]}', log=self.logfile, notime=True)
        log(f'Epoch:\t\t{self.hyper["stop_epoch"]}', log=self.logfile, notime=True)
        log(f'num_workers:\t{self.args["num_workers"]}', log=self.logfile, notime=True)
        log(f'Command:\t{self.dst["command"]}', log=self.logfile, notime=True)
        log(f"Let's use {torch.cuda.device_count()} GPUs!", log=self.logfile, notime=True)
        # self.device != torch.device(type='cpu') 
        if 'gpu_preprocess' in self.dst and self.dst['gpu_preprocess']:
            log("Using PyTorch's GPU Preprocess...")
            self.use_gpu = True
        else:
            log(f"Using Numpy's CPU Preprocess")
            self.use_gpu = False 

        self.ratiofix = True if 'ratiofix' in self.dst['command'] else False
        
        # 设置信号处理
        # self.setup_signal_handler()
    
    def change_eval_dst(self, mode='eval'):
        self.dst = self.args[f'dst_{mode}']
        self.dstname = self.dst['dstname']
        self.dst_eval = globals()[self.dst['dataset']](self.dst)
        self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)
        self.cache_dir = f'/data/cache/{self.dstname}'

    # 添加信号处理方法
    def setup_signal_handler(self):
        """设置信号处理函数，捕获中断信号并保存训练状态"""
        import sys
        def signal_handler(sig, frame):
            print('\n接收到中断信号，保存训练状态...')
            self.save_checkpoint(self.current_epoch, f'{self.fast_ckpt}/{self.model_name}_interrupt.pth')
            print('训练状态已保存，程序退出')
            sys.exit(0)
        
        import signal
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler) # kill命令

    def train(self):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        start_epoch = self.current_epoch + 1
        for epoch in range(start_epoch, self.hyper['stop_epoch']+1):
            self.current_epoch = epoch
            # log init
            self.net.train()
            self.train_psnr.reset()
            runtime = {'preprocess':0, 'dataloader':0, 'net':0, 'bp':0, 'metric':0, 'total':1e-9}
            time_points = [0] * 10
            time_points[0] = time.time()

            with tqdm(total=len(self.dataloader_train)) as t:                
                for k, data in enumerate(self.dataloader_train):
                    runtime['dataloader'] += timestamp(time_points, 1)
                    # Preprocess
                    imgs_lr, imgs_hr, ratio, noise_map = self.preprocess(data, mode='train', preprocess=True)
                    runtime['preprocess'] += timestamp(time_points, 2)
                    
                    # 训练
                    self.optimizer.zero_grad()
                    if noise_map is not None:
                        outputs = self.net(imgs_lr, noise_map)
                        # 检查输出格式
                        if isinstance(outputs, tuple) and len(outputs) == 4:
                            main_output, texture_mask, detail_output, denoise_output = outputs
                        else:
                            main_output = outputs
                            texture_mask, detail_output, denoise_output = None, None, None
                            
                        # 如果去噪没提前线性提亮，算loss的时候提亮上去
                        if self.dst['ori'] is True:
                            main_output = main_output * ratio
                            if detail_output is not None:
                                detail_output = detail_output * ratio
                            if denoise_output is not None:
                                denoise_output = denoise_output * ratio
                        
                        pred = main_output
                                
                        # 计算多损失
                        loss, loss_values = self.compute_multi_loss(main_output, detail_output, denoise_output, imgs_hr)
                    else:
                        pred = self.net(imgs_lr)
                        # 极暗，乘上去
                        if self.dst['ori'] is True:
                            pred = pred * ratio
                        loss = self.loss(pred.clamp(0,1), imgs_hr)
                    runtime['net'] += timestamp(time_points, 3)
                    loss.backward()
                    self.optimizer.step()
                    runtime['bp'] += timestamp(time_points, 4)

                    # 更新tqdm的参数
                    with torch.no_grad():
                        if self.arch['use_dpsv']: 
                            pred = pred[0]
                        if 'rgb_gain' in data:
                            data['rgb_gain'] = data['rgb_gain'].view_as(ratio)
                            pred = pred / data['rgb_gain']
                            imgs_hr = imgs_hr / data['rgb_gain']
                        pred = torch.clamp(pred, 0, 1)
                        imgs_hr = torch.clamp(imgs_hr, 0, 1)
                        psnr = PSNR_Loss(pred, imgs_hr)
                        self.train_psnr.update(psnr.item())

                    # 格式化损失值用于显示
                    # loss_str = ' '.join([f"{k}:{v:.4f}" for k, v in loss_values.items()])
                    
                    runtime['total'] = runtime['preprocess']+runtime['dataloader']+runtime['net']+runtime['bp']
                    t.set_description(f'Epoch {epoch}')
                    t.set_postfix({'lr':f"{lr:.2e}", 'PSNR':f"{self.train_psnr.avg:.2f}",
                                    # 'loader':f"{100*runtime['dataloader']/runtime['total']:.1f}%",
                                    # 'process':f"{100*runtime['preprocess']/runtime['total']:.1f}%",
                                    # 'net':f"{100*runtime['net']/runtime['total']:.1f}%",
                                    # 'bp':f"{100*runtime['bp']/runtime['total']:.1f}%",
                                    # 'loss': loss_str
                                    })
                    t.update(1)
                    time_points[0] = time.time()

            # 更新学习率
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # 存储模型
            if epoch % self.hyper['save_freq'] == 0:
                # model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                epoch_id = epoch // self.hyper['plot_freq'] * self.hyper['plot_freq']
                save_path = os.path.join(self.model_dir, '%s_e%04d.pth'% (self.model_name, epoch_id))
                self.save_checkpoint(epoch_id, save_path)
                # torch.save(model_dict, save_path)
            
            # 输出过程量，随时看
            savefile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.jpg')
            logfile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.pkl')
            self.train_psnr.plot_history(savefile=savefile, logfile=logfile)
            # if epoch % self.hyper['plot_freq'] == 0:
            wb = data['wb'][0].numpy()
            if self.dst['ori'] is True:
                imgs_lr = imgs_lr * ratio
                pred = pred
                imgs_hr = imgs_hr# * ratio
            
            if self.save_plot:
                inputs = imgs_lr[0].detach().cpu().numpy().clip(0,1)
                output = pred[0].detach().cpu().numpy()
                target = imgs_hr[0].detach().cpu().numpy()
                temp_img = np.concatenate((inputs, output, target),axis=2)[:3]
                temp_img[0] = temp_img[0] * wb[0]
                temp_img[2] = temp_img[2] * wb[2]
                filename = os.path.join(self.sample_dir, 'temp', f'temp_{epoch//10*10:04d}.png')
                temp_img = temp_img.transpose(1,2,0)[:,:,::-1] ** (1/2.2)
                cv2.imwrite(filename, np.uint8(temp_img*255))

            # fast eval
            if epoch % self.hyper['plot_freq'] == 0:
                log(f"learning_rate: {lr:.3e}")
                self.dst_eval.fast_eval(on=True)
                self.eval(epoch=epoch)
                self.dst_eval.fast_eval(on=False)
                # model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                # torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
                self.save_checkpoint(epoch, f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
            
            # reload best model each period
            num_of_epochs = self.hyper['stop_epoch'] - self.hyper['last_epoch']
            T = self.hyper['T'] if 'T' in self.hyper else 1 
            period = num_of_epochs//T
            if (self.hyper['last_epoch']+epoch) % period == 0:
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
                if os.path.exists(model_path):
                    model = torch.load(model_path, map_location=self.device)

                    # 检查加载的文件是新格式还是旧格式
                    if isinstance(model, dict) and 'model' in model:
                        # 新格式：包含完整训练状态
                        model_weights = model['model']
                    else:
                        # 旧格式：仅包含模型权重
                        model_weights = model

                    self.net = load_weights(self.net, model_weights, self.multi_gpu, by_name=True)
                    log(f'Successfully reload best model (Eval PSNR:{self.best_psnr})',
                        log=f'./logs/log_{self.model_name}.log')

    def eval(self, epoch=-1):
        self.net.eval()
        self.eval_psnr.reset()
        self.eval_ssim.reset()
        self.eval_psnr_lr.reset()
        self.eval_psnr_dn.reset()
        self.eval_ssim_lr.reset()
        self.eval_ssim_dn.reset()
        # record every metric
        metrics = {}
        metrics_path = f'./metrics/{self.model_name}_metrics.pkl'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pkl.load(f)
        # multiprocess
        if epoch > 0:
            pool = []
        else:
            pool = ProcessPoolExecutor(max_workers=max(4, self.args['num_workers']))
        task_list = []
        save_plot = self.save_plot
        with tqdm(total=len(self.dataloader_eval)) as t:
            for k, data in enumerate(self.dataloader_eval):
                # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                imgs_lr, imgs_hr, ratio, noise_map = self.preprocess(data, mode='eval', preprocess=False)
                wb = data['wb'][0].numpy()
                ccm = data['ccm'][0].numpy()
                name = data['name'][0]
                ISO = data['ISO'].item()
                exp = data['ExposureTime'].item()
                # print(ISO)

                with torch.no_grad():
                    # # 太大了就用下面这个策略
                    # croped_imgs_lr = self.dst_eval.eval_crop(imgs_lr)
                    # croped_imgs_hr = self.dst_eval.eval_crop(imgs_hr)
                    # croped_imgs_dn = []
                    # for img_lr, img_hr in zip(croped_imgs_lr, croped_imgs_hr):
                    #     img_dn = self.net(img_lr)
                    #     croped_imgs_dn.append(img_dn)
                    # croped_imgs_dn = torch.cat(croped_imgs_dn)
                    # imgs_lr = self.dst_eval.eval_merge(croped_imgs_lr)
                    # imgs_dn = self.dst_eval.eval_merge(croped_imgs_dn)
                    
                    detail_output = None
                    denoise_output = None
                    # 扛得住就pad再crop
                    if imgs_lr.shape[-1] % 16 != 0:
                        p2d = (4,4,4,4)
                        imgs_lr = F.pad(imgs_lr, p2d, mode='reflect')
                        if noise_map is not None:
                            imgs_dn = self.net(imgs_lr, noise_map)
                        else:
                            imgs_dn = self.net(imgs_lr)

                        if isinstance(imgs_dn, tuple) and len(imgs_dn) == 4:
                            imgs_dn, texture_mask, detail_output, denoise_output = imgs_dn
                        else:
                            imgs_dn = imgs_dn

                        imgs_lr = imgs_lr[..., 4:-4, 4:-4]
                        imgs_dn = imgs_dn[..., 4:-4, 4:-4]
                    else:
                        if noise_map is not None:
                            imgs_dn = self.net(imgs_lr, noise_map)
                        else:
                            imgs_dn = self.net(imgs_lr)

                        if isinstance(imgs_dn, tuple) and len(imgs_dn) == 4:
                            imgs_dn, texture_mask, detail_output, denoise_output = imgs_dn
                        else:
                            imgs_dn = imgs_dn
                    
                    # brighten
                    if self.dst['ori']:
                        imgs_lr = imgs_lr * ratio
                        imgs_dn = imgs_dn * ratio
                    imgs_lr = torch.clamp(imgs_lr, 0, 1)
                    imgs_dn = torch.clamp(imgs_dn, 0, 1)

                    # np.save(f'{name}_input.npy', imgs_lr.detach().cpu().numpy())
                    # np.save(f'{name}_denoised.npy', imgs_dn.detach().cpu().numpy())
                    # np.save(f'{name}_gt.npy', imgs_hr.detach().cpu().numpy())
                    
                    # align to ELD's configuration (.=_=.)
                    if self.args['brightness_correct'] and epoch < 0:
                        imgs_dn = self.corrector(imgs_dn, imgs_hr)

                    # PSNR & SSIM (Raw domain)
                    output = tensor2im(imgs_dn)
                    target = tensor2im(imgs_hr)
                    res = quality_assess(output, target, data_range=255)
                    raw_metrics = [res['PSNR'], res['SSIM']]
                    self.eval_psnr.update(res['PSNR'])
                    self.eval_ssim.update(res['SSIM'])
                    metrics[name] = raw_metrics
                    # convert raw to rgb
                    if save_plot:
                        if self.infos is None:
                            inputs = tensor2im(imgs_lr)
                            res_in = quality_assess(inputs, target, data_range=255)
                            raw_metrics = [res_in['PSNR'], res_in['SSIM']] + raw_metrics
                        else:
                            raw_metrics = [self.infos[k]['PSNR_raw'], self.infos[k]['SSIM_raw']] + raw_metrics
                        if epoch > 0:
                            # self.multiprocess_plot(imgs_lr, imgs_dn, imgs_hr, 
                            #         wb, ccm, name, save_plot, epoch, raw_metrics, k)
                            pool.append(threading.Thread(target=self.multiprocess_plot, args=(imgs_lr, imgs_dn, imgs_hr, 
                                    wb, ccm, name, save_plot, epoch, raw_metrics, k, denoise_output, detail_output)))
                            pool[k].start()
                        else:
                            infos = self.infos[k] if self.infos is not None else None
                            # 多进程
                            if infos is None:
                                inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
                                target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
                            else:
                                inputs = np.load(infos['path_npy_in'])
                                target = np.load(infos['path_npy_gt'])

                            if 'isp' not in self.dst['command'].lower():
                                output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
                                detail_rgb = raw2rgb_rawpy(detail_output, wb=wb, ccm=ccm) if detail_output is not None else None
                                denoise_rgb = raw2rgb_rawpy(denoise_output, wb=wb, ccm=ccm) if denoise_output is not None else None

                            # raw_metrics = None # 用RGB metrics

                            # task_list.append(
                            #     pool.submit(plot_dual_path_sample, inputs, output, target, 
                            #         detail_rgb, denoise_rgb,
                            #         filename=name, save_plot=save_plot, epoch=epoch,
                            #         model_name=self.model_name, save_path=self.sample_dir,
                            #         res=raw_metrics
                            #         )
                            #     )
                            
                            task_list.append(
                                pool.submit(plot_sample_V2, inputs, output, target, 
                                    filename=name, save_plot=save_plot, epoch=epoch,
                                    model_name=self.model_name, save_path=self.sample_dir,
                                    res=raw_metrics, detail_output=detail_rgb, denoise_output=denoise_rgb
                                )
                            )

                    t.set_description(f'{name}')
                    t.set_postfix({'PSNR':f"{self.eval_psnr.avg:.2f}"})
                    t.update(1)

        if save_plot:
            if epoch > 0:
                for i in range(len(pool)):
                    pool[i].join()
            else:
                pool.shutdown(wait=True)
                for task in as_completed(task_list):
                    psnr, ssim, name = task.result()
                    metrics[name] = (psnr[1], ssim[1])
                    # if name[0] == '1' or self.dstname=='ELD':
                    self.eval_psnr_lr.update(psnr[0])
                    self.eval_psnr_dn.update(psnr[1])
                    self.eval_ssim_lr.update(ssim[0])
                    self.eval_ssim_dn.update(ssim[1])
        else:
            self.eval_psnr_dn = self.eval_psnr
            self.eval_ssim_dn = self.eval_ssim

        # 超过最好记录才保存
        if self.eval_psnr_dn.avg >= self.best_psnr and epoch > 0:
            self.best_psnr = self.eval_psnr_dn.avg
            log(f"Best PSNR is {self.best_psnr} now!!")
            # model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
            # torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
            self.save_checkpoint(epoch, f'{self.fast_ckpt}/{self.model_name}_best_model.pth')

        log(f"Epoch {epoch}: PSNR={self.eval_psnr.avg:.2f}\n"
            +f"psnrs_lr={self.eval_psnr_lr.avg:.2f}, psnrs_dn={self.eval_psnr_dn.avg:.2f}"
            +f"\nssims_lr={self.eval_ssim_lr.avg:.4f}, ssims_dn={self.eval_ssim_dn.avg:.4f}",
            log=f'./logs/log_{self.model_name}.log')
        if epoch < 0:
            with open(metrics_path, 'wb') as f:
                pkl.dump(metrics, f)
        savefile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.jpg')
        logfile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.pkl')
        if epoch > 0:
            self.eval_psnr.plot_history(savefile=savefile, logfile=logfile)
        del pool
        plt.close('all')
        gc.collect()
        return metrics
    
    def multiprocess_plot(self, imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, raw_metrics, k, denoise_output=None, detail_output=None):
        # if self.infos is None:
        inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
        # else:
        #     inputs = np.load(self.infos[k]['path_npy_in'])
        #     target = np.load(self.infos[k]['path_npy_gt'])
        output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
        denoise_rgb = raw2rgb_rawpy(denoise_output, wb=wb, ccm=ccm) if denoise_output is not None else None
        detail_rgb = raw2rgb_rawpy(detail_output, wb=wb, ccm=ccm) if detail_output is not None else None
        
        # psnr, ssim, _ = plot_sample(inputs, output, target, 
        #                 filename=name, 
        #                 save_plot=save_plot, epoch=epoch,
        #                 model_name=self.model_name,
        #                 save_path=self.sample_dir,
        #                 res=raw_metrics)

        psnr, ssim, _ = plot_sample_V2(inputs, output, target, 
                    filename=name, 
                    save_plot=save_plot, epoch=epoch,
                    model_name=self.model_name,
                    save_path=self.sample_dir,
                    res=raw_metrics,
                    detail_output=detail_rgb,
                    denoise_output=denoise_rgb)
        
        self.eval_psnr_lr.update(psnr[0])
        self.eval_psnr_dn.update(psnr[1])
        self.eval_ssim_lr.update(ssim[0])
        self.eval_ssim_dn.update(ssim[1])

    def predict(self, raw, name='ds'):
        self.net.eval()
        img_lr = raw2bayer(raw+self.dst["bl"])[None, ...]
        img_lr = torch.from_numpy(img_lr)
        img_lr = img_lr.type(torch.FloatTensor).to(self.device)
        with torch.no_grad():
            croped_imgs_lr = self.dst_eval.eval_crop(img_lr)
            croped_imgs_dn = []
            for img_lr in tqdm(croped_imgs_lr):
                img_dn = self.net(img_lr)
                croped_imgs_dn.append(img_dn)
            croped_imgs_dn = torch.cat(croped_imgs_dn)
            img_dn = self.dst_eval.eval_merge(croped_imgs_dn)
            img_dn = img_dn
            img_dn = img_dn[0].detach().cpu().numpy()
        np.save(f'{name}.npy', img_dn)
    
    def preprocess(self, data, mode='train', preprocess=True):
        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
        imgs_hr = tensor_dim5to4(data['hr']).type(torch.FloatTensor).to(self.device)
        imgs_lr = tensor_dim5to4(data['lr']).type(torch.FloatTensor).to(self.device)
        # self.use_gpu = True
        dst = self.dst_train if mode=='train' else self.dst_eval


        if self.use_gpu and mode=='train' and preprocess:
            b = imgs_lr.shape[0]
            if self.args['dst_train']['dataset'] == 'Mix_Dataset':
                data['ratio'] = data['ratio'].view(-1).type(torch.FloatTensor).to(self.device)
                aug_r, aug_g, aug_b = get_aug_param_torch(data, b=b, command=self.dst['command'])
                aug_wbs = torch.stack((aug_r, aug_g, aug_b, aug_g), dim=1)
                data['rgb_gain'] = torch.ones(b) * (aug_g + 1)
                data['wb'] = data['wb'][0].repeat(b, 1)
                noise_params = []       
                for i in range(b):
                    aug_wb = aug_wbs[i].numpy()
                    if data['black_lr'][0]: aug_wb += 1
                    dgain = data['ratio'][i]
                    imgs_lr[i] = imgs_lr[i] if self.dst['ori'] else imgs_lr[i] * dgain
                    if np.abs(aug_wb).max() != 0:
                        data['wb'][i] *= (1+aug_wb[1]) / (1+aug_wb)
                        iso = data['ISO'][i//self.dst['crop_per_image']].item()
                        dn, dy, p = SNA_torch(imgs_hr[i], aug_wb, iso=iso, ratio=dgain, black_lr=data['black_lr'][0],
                            camera_type=self.dst['camera_type'], ori=self.dst['ori'])
                        imgs_lr[i] = imgs_lr[i] + dn 
                        imgs_hr[i] = imgs_hr[i] + dy
                        noise_params.append({
                            'K': p['K'],
                            'sigGs': p['sigGs'], 
                            'wp': p['wp'],
                            'bl': p['bl']
                        })
                    else:
                        # 原始数据：根据ISO估算参数
                        iso = data['ISO'][i//self.dst['crop_per_image']].item()
                        estimated_params = get_camera_noisy_params_max(f'{self.dst["camera_type"]}_{iso}')
                        noise_params.append(estimated_params)
                data['noise_parms'] = noise_params

                # TODO：noisemap
                if self.arch.get('use_noise_map', False):
                    noise_map = self.generate_noise_map_batch(imgs_lr, noise_params)
                    data['noise_map'] = noise_map

                # 处理噪声图(如果存在)
                noise_map = None
                if 'noise_map' in data:
                    noise_map = tensor_dim5to4(data['noise_map']).type(torch.FloatTensor).to(self.device)
                    
            elif self.args['dst_train']['dataset'] == 'Raw_Dataset':
                data['ratio'] = torch.ones(b, device=self.device)
                # 人工加噪声，注意，这里统一时间的视频应该共享相同的噪声参数！！
                for i in range(b):
                    if dst.args['params'] is None:
                        noise_param = sample_params_max(camera_type=self.dst['camera_type'], ratio=None)
                    else:
                        noise_param = dst.args['params']
                    for key in noise_param:
                        if torch.is_tensor(noise_param[key]) is False:
                            noise_param[key] = torch.from_numpy(np.array(noise_param[key], np.float32))
                        noise_param[key] = noise_param[key].to(self.device)
                    data['ratio'][i] = noise_param['ratio']
                    imgs_lr[i] = generate_noisy_torch(imgs_lr[i], param=noise_param,
                                noise_code=self.dst['noise_code'], ori=self.dst['ori'], clip=self.dst['clip'])
        else: # mode == 'eval'
            pass
        
        ratio = data['ratio'].type(torch.FloatTensor).to(self.device)
        ratio = ratio.view(-1,1,1,1)
        if 'rgb_gain' in data:
            data['rgb_gain'] = data['rgb_gain'].type(torch.FloatTensor).to(self.device).view_as(ratio)
        
        if self.dst['clip']:
            lb = -100 if 'HB' in self.dst['command'] else 0
            imgs_lr = imgs_lr.clamp(lb, 1)
            imgs_hr = imgs_hr.clamp(0, 1)
        return imgs_lr, imgs_hr, ratio, noise_map
    
    def compute_multi_loss(self, main_output, detail_output, denoise_output, gt):

        total_loss = 0
        
        # 主输出损失 - 使用普通的L1损失
        main_loss = self.loss(main_output, gt)
        total_loss += main_loss
        
        # 记录详细损失值用于日志（可选）
        loss_values = {'main_loss': main_loss.item()}
        
        # 细节路径中间监督 - 使用perceptual loss和gradient loss
        if detail_output is not None:
            # 可以添加VGG感知损失，需要先初始化
            if hasattr(self, 'perceptual_loss'):
                detail_percep_loss = self.perceptual_loss(detail_output, gt)
                total_loss += detail_percep_loss * 0.1  # 权重可调
                loss_values['detail_percep_loss'] = detail_percep_loss.item()
                
            # 添加梯度损失
            if hasattr(self, 'gradient_loss'):
                detail_grad_loss = self.gradient_loss(detail_output, gt)
                total_loss += detail_grad_loss * 0.5  # 权重可调
                loss_values['detail_grad_loss'] = detail_grad_loss.item()
        
        # 降噪路径中间监督 - 使用L1损失
        if denoise_output is not None:
            denoise_loss = self.loss(denoise_output, gt)
            total_loss += denoise_loss * 0.5  # 权重可调
            loss_values['denoise_loss'] = denoise_loss.item()
        
        return total_loss, loss_values
    
    def save_checkpoint(self, epoch, filepath, is_best=False):
        """保存完整的训练状态"""
        checkpoint = {
            'epoch': epoch,
            'model': self.net.module.state_dict() if self.multi_gpu else self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'train_psnr': {
                'avg': self.train_psnr.avg,
                'history': self.train_psnr.history
            },
            'eval_psnr': {
                'avg': self.eval_psnr.avg,
                'history': self.eval_psnr.history
            },
            'eval_ssim': {
                'avg': self.eval_ssim.avg if hasattr(self.eval_ssim, 'avg') else 0,
                'history': self.eval_ssim.history if hasattr(self.eval_ssim, 'history') else []
            }
            # ,
            # 'random_state': {
            #     'numpy': np.random.get_state(),
            #     'pytorch': torch.get_rng_state(),
            #     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            # }
        }
        
        torch.save(checkpoint, filepath)
        if is_best:
            log(f"保存最佳模型: {filepath} (PSNR: {self.best_psnr:.2f})")
        else:
            log(f"保存训练状态: {filepath}")

    def load_checkpoint(self, checkpoint):
        """加载训练状态断点"""
        # 加载模型权重
        if self.multi_gpu:
            self.net.module.load_state_dict(checkpoint['model'])
        else:
            self.net.load_state_dict(checkpoint['model'])
        
        # 加载优化器状态
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载学习率调度器
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # 加载训练指标
        if 'best_psnr' in checkpoint:
            self.best_psnr = checkpoint['best_psnr']
        
        # 加载PSNR历史
        if 'train_psnr' in checkpoint and isinstance(checkpoint['train_psnr'], dict):
            if 'history' in checkpoint['train_psnr']:
                self.train_psnr.history = checkpoint['train_psnr']['history']
        
        if 'eval_psnr' in checkpoint and isinstance(checkpoint['eval_psnr'], dict):
            if 'history' in checkpoint['eval_psnr']:
                self.eval_psnr.history = checkpoint['eval_psnr']['history']
        
        # 加载随机状态
        # if 'random_state' in checkpoint:
        #     if 'numpy' in checkpoint['random_state']:
        #         np.random.set_state(checkpoint['random_state']['numpy'])
        #     if 'pytorch' in checkpoint['random_state']:
        #         torch.set_rng_state(checkpoint['random_state']['pytorch'])
        #     if 'cuda' in checkpoint['random_state'] and checkpoint['random_state']['cuda'] is not None:
        #         torch.cuda.set_rng_state_all(checkpoint['random_state']['cuda'])

    def generate_noise_map_batch(self, images, noise_params_list):
        """为一个batch生成噪声图"""
        noise_maps = []
        
        for i, (img, params) in enumerate(zip(images, noise_params_list)):
            # 使用实际的噪声参数生成噪声图
            noise_map = generate_noise_map(
                image=img.cpu().numpy(),
                noise_params=params
            )
            noise_maps.append(noise_map)

        # 添加归一化处理
        if noise_maps is not None:
            # 判断是否为PyTorch张量
            is_tensor = torch.is_tensor(noise_maps)
            
            # 遍历每个裁剪样本进行归一化
            # [crop_per_image, C, H, W]
            normalized_maps = []
            for single_map in noise_maps:
                
                if is_tensor:
                    map_min = torch.min(single_map)
                    map_max = torch.max(single_map)
                    
                    # 避免除零错误
                    if map_max - map_min > 1e-6:
                        normalized = (single_map - map_min) / (map_max - map_min)
                    else:
                        normalized = torch.zeros_like(single_map) + 0.5
                else:
                    map_min = np.min(single_map)
                    map_max = np.max(single_map)
                    
                    # 避免除零错误
                    if map_max - map_min > 1e-6:
                        normalized = (single_map - map_min) / (map_max - map_min)
                    else:
                        normalized = np.zeros_like(single_map) + 0.5
                
                normalized_maps.append(normalized)
            
            # 重新组合批次
            if is_tensor:
                noise_maps = torch.stack(normalized_maps, dim=0)
            else:
                noise_maps = np.stack(normalized_maps, axis=0)
        
        return np.stack(noise_maps, axis=0)

def MultiProcessPlot(imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, 
                    raw_metrics, infos, model_name, sample_dir):
    if infos is None:
        inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
    else:
        inputs = np.load(infos['path_npy_in'])
        target = np.load(infos['path_npy_gt'])
    output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
    
    psnr, ssim = plot_sample(inputs, output, target, 
                    filename=name, 
                    save_plot=save_plot, epoch=epoch,
                    model_name=model_name,
                    save_path=sample_dir,
                    res=raw_metrics)
    return psnr, ssim


if __name__ == '__main__':
    trainer = SID_Trainer()
    if trainer.mode == 'train':
        trainer.train()
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        trainer.mode = 'evaltest'
    # best_model
    best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_best_model.pth')
    if os.path.exists(best_model_path) is False: 
        best_model_path = os.path.join(f'{trainer.fast_ckpt}',f'{trainer.model_name}_last_model.pth')
    best_model = torch.load(best_model_path, map_location=trainer.device)

    # 检查加载的文件是新格式还是旧格式
    if isinstance(best_model, dict) and 'model' in best_model:
        # 新格式：包含完整训练状态
        model_weights = best_model['model']
        log(f"加载新格式模型权重用于评估")
        log(f"Epoch{best_model['epoch']}, Best_PSNR{best_model['best_psnr']}")
    else:
        # 旧格式：仅包含模型权重
        model_weights = best_model
        log(f"加载旧格式模型权重用于评估")

    trainer.net = load_weights(trainer.net, model_weights, multi_gpu=trainer.multi_gpu)
    if 'eval' in trainer.mode:
        # ELD
        trainer.change_eval_dst('eval')
        for dgain in trainer.args['dst_eval']['ratio_list']:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'ELD Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.ratio_list=[dgain]
            trainer.dst_eval.recheck_length()
            metrics = trainer.eval(-1)

    if 'test' in trainer.mode:
        # SID
        trainer.change_eval_dst('test')
        SID_ratio_list = [100, 250, 300]
        for dgain in SID_ratio_list:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'SID Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            metrics = trainer.eval(-1)
    log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')