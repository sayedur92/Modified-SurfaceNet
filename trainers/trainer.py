import sys
sys.path.append('/home/hpc/vlgm/vlgm103v/genmatpro/idea/')

import torch
import torch.optim
from torch.nn.functional import l1_loss, mse_loss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from pathlib import Path

from accelerate import Accelerator
from torchvision.transforms import ToTensor



from models.discriminator import PatchDiscriminator
from models.surfacenet import SurfaceNet
from utils.infinite_dataloader import InfiniteDataLoader
from utils.losses import msssim_loss, rmse_loss
from utils.vgg_loss import compute_vgg_loss
from utils.utils import make_plot_maps
from my_datasets.utils import texture_maps
from TEED.inference import get_sketch_from_image_batch
from render1.render import renderTexFromTensor

class Trainer:
    def __init__(self, args, accelerator: Accelerator, tracker, datasets):
        self.accelerator = accelerator
        self.tracker = tracker

        self.args = args
        self.device = torch.device(self.accelerator.device)

        self.epoch = 0

        # Optimizer params
        optim_params = {'lr': args.lr}
        if args.optim == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif args.optim == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}

        optim_class = getattr(torch.optim, args.optim)

        object_to_checkpoint = []

    
        net = SurfaceNet()
        net.to(self.device)
        
        optim = optim_class(params=[param for param in net.parameters() 
                                    if param.requires_grad], **optim_params)

        self.net = self.accelerator.prepare_model(net)
        self.optim = self.accelerator.prepare_optimizer(optim, device_placement=True)

        object_to_checkpoint.append(self.net)
        object_to_checkpoint.append(self.optim)

        # Setup adversarial training
        if args.train_adversarial:
            discr = PatchDiscriminator()
            discr.to(self.device)
            optim_discr = optim_class(params=[param for param in discr.parameters() 
                                              if param.requires_grad], **optim_params)

            self.discr = self.accelerator.prepare_model(discr)
            self.optim_discr = self.accelerator.prepare_optimizer(optim_discr, device_placement=True)

            object_to_checkpoint.append(self.discr)
            object_to_checkpoint.append(self.optim_discr)

        # Register objects to checkpoint
        self.accelerator.register_for_checkpointing(*object_to_checkpoint)
        if args.resume:
            self.accelerator.load_state(args.resume)
            self.epoch = int(args.resume.split("_")[-1])

        # Params
        self.alpha_m = 0.88
        self.alpha_adv = 0.015

        
        splits = list(datasets.keys())

        # Setup data loaders
        loader_train = DataLoader(
            datasets['train']['synth'], batch_size=args.batch_size,
            shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2
        )
        

        loader_test = DataLoader(
            datasets['test']['synth'], batch_size=args.batch_size,
            shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=2
        )

        loader_train = self.accelerator.prepare_data_loader(loader_train, device_placement=True)
        loader_test = self.accelerator.prepare_data_loader(loader_test, device_placement=True)

        self.loaders = {
            'train': {'synth': loader_train},
            'test': {'synth': loader_test},
        }


        if args.train_real:
            self.loaders['train']['real'] = DataLoader(
                datasets['train']['real'], batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers)

            self.loaders['test']['real'] = DataLoader(
                datasets['test']['real'], batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers)
            
        

        
    

    def train(self):
        # Start training
        for epoch in range(self.epoch, self.args.epochs):
            self.run_epoch(epoch, train=True)

            torch.cuda.empty_cache()

            with torch.no_grad():
                self.run_epoch(epoch, train=False)

            if epoch % self.args.save_every == 0:
                # Save the starting state
                logging_dir = Path(self.tracker.logging_dir)/"checkpoints"/f"checkpoint_{epoch}"
                self.accelerator.save_state(output_dir=logging_dir)

    def run_epoch(self, epoch_idx, train=True):
        if train:
            self.net.train()
            if self.args.train_adversarial:
                self.discr.train()
        else:
            self.net.eval()
            if self.args.train_adversarial:
                self.discr.eval()

        split = "train" if train else "test"

        # Training loop
        for batch_idx, batch in enumerate(tqdm(self.loaders[split]['synth'])):
            step_idx = epoch_idx * len(self.loaders[split]['synth']) + batch_idx
            losses, maps = self.forward_batch(batch, step_idx, train=train)
            

            for k, v in losses.items():
                self.tracker.log({f"{split}/{k}": v}, step_idx)

            # Prepare and log image maps
            if batch_idx % self.args.log_every == 0:
                log_maps = {}
                for k, v in maps.items():
                    image = make_grid(v, nrow=len(texture_maps) + 1)
                    log_maps[f"{split}/{k}"] = image.unsqueeze(0)

                self.tracker.log_images(log_maps, step_idx)
                    

    def forward_batch(self, batch, step_idx, train=True, real=False):
        inputs = batch["render"]
        sketch = batch["sketch"]  
        outputs = self.net(inputs,sketch)
        
        if not real:
            
            targets = {key: batch[key] for key in batch.keys()}
            sketch_target = targets.pop("sketch", None)

            maps = {
                'gen': make_plot_maps(inputs, outputs),
                'gt': make_plot_maps(inputs, targets)
            }


            if train:
                return self.train_batch(inputs, outputs, targets,sketch_target, step_idx), maps
            else:
                return self.eval_batch(inputs,outputs, targets,sketch_target), maps
        else:
            raise NotImplementedError("Real data not implemented yet.")
            return self.train_real(inputs,outputs), self.make_plot_maps(inputs,outputs)
    

    
    def train_batch(self, inputs, outputs, targets,sketch_target, step_idx):
        render_images = renderTexFromTensor(outputs)  
        if not isinstance(render_images, list):
            render_images = [render_images]

        render_image_tensors = torch.stack([ToTensor()(img).to(inputs.device) for img in render_images])
        
        sketch_out = get_sketch_from_image_batch(render_image_tensors)
        
        losses = {}
        loss = 0

        # Adversarial losses (if applicable)
        if self.args.train_adversarial:
            real_batch = torch.cat([inputs, *[targets[x] for x in texture_maps]], 1)
            fake_batch = torch.cat([inputs, *[outputs[x] for x in texture_maps]], 1)
            
            
            self.__train_discr(real_batch, fake_batch.detach())

            # Train generator
            if step_idx >= self.args.adv_start:
                gen_batch = torch.cat([inputs, *[outputs[x] for x in texture_maps]], 1)

                pred_patch = self.discr(gen_batch)
                discr_loss = self.alpha_adv * mse_loss(pred_patch, torch.ones(pred_patch.shape, device=self.device))
                losses[f'adv_loss'] = discr_loss.item()

                loss += discr_loss

        
        vgg_l = compute_vgg_loss(render_image_tensors, inputs)
        sketch_l = l1_loss(sketch_out, sketch_target)

        loss += vgg_l + sketch_l
        losses['vgg_loss'] = vgg_l.item()
        losses['sketch_loss'] = sketch_l.item()


        for key in outputs.keys():
            out, tar = outputs[key], targets[key]

            if out.size(1) != tar.size(1):
                if out.size(1) == 1:  
                    out = out.repeat(1, 3, 1, 1)
                elif tar.size(1) == 1:  
                    tar = tar.repeat(1, 3, 1, 1)

 
            l_pix = l1_loss(out, tar)
            l_msssim = msssim_loss(out, tar)
            l_std = l1_loss(out.std(), tar.std())
           
            l_map = l_pix + self.alpha_m * l_msssim + l_std
            loss += l_map

    # Log individual losses
            losses[f'{key.lower()}_loss'] = l_map.item()
            losses[f'{key.lower()}_l1'] = l_pix.item()
            losses[f'{key.lower()}_rmse'] = rmse_loss(out, tar).item()
            losses[f'{key.lower()}_rmse_un'] = rmse_loss((out + 1) / 2, (tar + 1) / 2).item()


        # Perform the backward pass
        self.optim.zero_grad()
        self.accelerator.backward(loss)
        self.optim.step()
        
        
        losses[f'loss'] = loss.item()

        

        return losses




    def eval_batch(self, inputs, outputs, targets,sketch_target):
        
        render_images=renderTexFromTensor(outputs)

        if not isinstance(render_images, list):
            render_images = [render_images]

        
        render_image_tensors = torch.stack([ToTensor()(img).to(inputs.device) for img in render_images])
        
        sketch_out = get_sketch_from_image_batch(render_image_tensors)
        

        sketch_out=get_sketch_from_image_batch(render_image_tensors) 

        losses = {}
        loss=0
        vgg_l = compute_vgg_loss(render_image_tensors, inputs)  
        sketch_l = l1_loss(sketch_out, sketch_target)  

        
        loss += vgg_l  
        loss += sketch_l  

        
        losses['vgg_loss'] = vgg_l.item()
        losses['sketch_loss'] = sketch_l.item()

        
        for key in outputs.keys():
            out, tar = outputs[key], targets[key]

            l_pix = l1_loss(out, tar)
            l_msssim = msssim_loss(out, tar)
            l_std = l1_loss(out.std(), tar.std())

            l_map = l_pix + self.alpha_m * l_msssim + l_std

            loss += l_map

            losses[f'{key.lower()}_loss'] = l_map.item()
            losses[f'{key.lower()}_l1'] = l_pix.item()
            losses[f'{key.lower()}_rmse'] = rmse_loss(out, tar)
            losses[f'{key.lower()}_rmse_un'] = rmse_loss((out + 1) / 2, (tar + 1) / 2)

        return losses

    def __train_discr(self, real_batch, fake_batch):
        pred_real = self.discr(real_batch)
        loss_real = mse_loss(pred_real, torch.ones(pred_real.shape, device=self.device))

        pred_fake = self.discr(fake_batch)
        loss_fake = mse_loss(pred_fake, torch.zeros(pred_fake.shape, device=self.device))

        loss_discr = 0.5 * (loss_real + loss_fake)

        self.optim_discr.zero_grad()
        self.accelerator.backward(loss_discr)
        self.optim_discr.step()

        return loss_discr

    