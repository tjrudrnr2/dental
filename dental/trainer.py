import os
import time
from matplotlib.pyplot import gray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from torchvision import transforms
from PIL import Image
import loss.hingeloss as hingeloss
import loss.r1 as r1
import wandb


class Trainer:
    def __init__(self,G,F,D_y,D_x,loader,target_loader,eval_loader,device,args):
        self.G=G.to(device)
        self.F=F.to(device)
        self.D_y=D_y.to(device)
        self.D_x=D_x.to(device)
        self.args=args
        self.device=device
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.adam_beta1, 0.999))
        self.F_optimizer = optim.Adam(self.F.parameters(), lr=args.lr, betas=(args.adam_beta1, 0.999))
        self.D_x_optimizer = optim.Adam(self.D_x.parameters(), lr=args.lr, betas=(args.adam_beta1, 0.999))
        self.D_y_optimizer = optim.Adam(self.D_y.parameters(), lr=args.lr, betas=(args.adam_beta1, 0.999))
        
        self.lambda_cycle=args.lambda_cycle
        self.lambda_identity=args.lambda_identity
        # dataloader
        self.loader=loader
        self.target_loader=target_loader
        self.eval_loader=eval_loader
        
        self.target_dir=args.target_dir
        self.ckpt_save_path=os.path.join(args.save_path, f'{args.run_name}')
        #loss
        if self.args.loss == "cyclegan":
            self.GAN_criterion=nn.MSELoss().to(device)
        elif self.args.loss == "hingeloss":
            self.GAN_criterion=hingeloss.HingeLoss().to(device)
            
        self.Cycle_criterion=nn.L1Loss().to(device)
        self.Identity_criterion=nn.L1Loss().to(device)
        if self.args.r1:
            self.R1_reg = r1.R1(self.args).to(device)

        # pool
        self.generated_x_images = ImagePool(args,50)
        self.generated_y_images = ImagePool(args,50)
        
        # initialization
        if args.initialization_epochs>=0:
            self.Initialization_criterion=nn.L1Loss().to(device)
            self.lambda_initialization=args.content_loss_weight
            G.apply(he_init)
            F.apply(he_init)
            D_x.apply(he_init)
            D_y.apply(he_init)
        
        self.use_initialization = args.initialization_epochs>=0
        
        self.num_epochs=args.num_epochs
        self.curr_epoch=0
        self.print_every=args.print_every
        self.generated_image_save_path=os.path.join(args.generated_image_save_path, f'{self.args.run_name}')
        
        self.init_loss_hist = []
        self.loss_D_x_hist = []
        self.loss_D_y_hist = []
        self.loss_G_GAN_hist = []
        self.loss_F_GAN_hist = []
        self.loss_cycle_hist = []
        self.loss_identity_hist = []
        self.image_test=args.image_test
        
        
    def train(self,num_epochs,initialization_epochs,save_path):
        
        for init_epoch in tqdm(range(self.curr_epoch,initialization_epochs),desc="init_epochs",mininterval=0.1):
            start=time.time()
            epoch_loss=0
            devide=0
            for ix,((img,_),target_img) in enumerate(zip(self.loader,self.target_loader)):
                img=img.to(self.device)
                target_img=target_img.to(self.device)
                
                loss=self.initialize_step(img,target_img)
                self.init_loss_hist.append(loss)
                epoch_loss+=loss
                devide+=1
            if self.image_test and self.curr_epoch%5==0:
                generate_and_save_images(self.G,self.loader,self.F,self.target_loader,
                                         save_path,self.curr_epoch,self.args.save,self.device)
            self.curr_epoch+=1
            if self.args.save:
                wandb.log({"loss_D_x_hist": loss/devide}) 

            print("Initialization Phase [{0}/{1}], {2:.4f} seconds".format(init_epoch + 1, initialization_epochs,
                                                                           time.time() - start))
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            print ('Error: Creating directory. ' + save_path)
        best_fid=1000
        best_epoch=0
        ## training start
        print("training real epoch start...")
        for epoch in tqdm(range(self.curr_epoch, num_epochs),desc="epochs",mininterval=0.01):
            start=time.time()
            epoch_loss_D_x = 0
            epoch_loss_D_y = 0
            epoch_loss_G_GAN = 0
            epoch_loss_F_GAN = 0
            epoch_loss_cycle = 0
            epoch_loss_identity = 0

            devide=0
            for ix,((img,_),target_img) in enumerate(zip(self.loader,self.target_loader)):
                img=img.to(self.device)
                target_img=target_img.to(self.device)
                # train_step
                loss_D_x, loss_D_y, loss_G_GAN, loss_F_GAN, loss_cycle, loss_identity,target_Dy,generated_Dy,target_Dx,generated_Dx = self.train_step(img,target_img)
                
                # hist
                self.loss_D_x_hist.append(loss_D_x)
                self.loss_D_y_hist.append(loss_D_y)
                self.loss_G_GAN_hist.append(loss_G_GAN)
                self.loss_F_GAN_hist.append(loss_F_GAN)
                self.loss_cycle_hist.append(loss_cycle)
                self.loss_identity_hist.append(loss_identity)

                epoch_loss_D_x += loss_D_x
                epoch_loss_D_y += loss_D_y
                epoch_loss_G_GAN += loss_G_GAN
                epoch_loss_F_GAN += loss_F_GAN
                epoch_loss_cycle += loss_cycle
                epoch_loss_identity += loss_identity
                step_target_Dy=target_Dy
                step_generated_Dy=generated_Dy
                step_target_Dx=target_Dx
                step_generated_Dx=generated_Dx
                devide+=1
                # print progress
                if (ix + 1) % self.print_every == 0:
                    print("Training Phase Epoch {0} Iteration {1}: loss_D_x: {2:.4f} loss_D_y: {3:.4f} loss_G: {4:.4f} loss_F: {5:.4f} "
                          "loss_cycle: {6:.4f} loss_identity: {7:.4f}".format(epoch + 1, ix+1, epoch_loss_D_x / (ix + 1), epoch_loss_D_y / (ix + 1),
                                                                              epoch_loss_G_GAN / (ix + 1), epoch_loss_F_GAN / (ix + 1),
                                                                              epoch_loss_cycle / (ix + 1), epoch_loss_identity / (ix + 1)))  # print progress    
                if self.args.save:
                    wandb.log({
                                "step_target_Dy" : step_target_Dy,
                                "step_generated_Dy" : step_generated_Dy,
                                "step_target_Dx" : step_target_Dx,
                                "step_generated_Dx" : step_generated_Dx
                    
                })
            if self.args.save:              
                fid=calculate_fid(self.G, self.eval_loader,os.path.join(self.generated_image_save_path,"test"),self.target_dir,device=self.device)
                wandb.log({"FID" : fid})
                if fid<=best_fid:
                    self.save_checkpoint(os.path.join(self.ckpt_save_path, 'best.ckpt'))
                    best_fid=fid
                    best_epoch=self.curr_epoch
                    wandb.run.summary["best_fid"]=fid
                    wandb.run.summary["best_epoch"]=best_epoch
                wandb.log({"loss_D_x_hist": epoch_loss_D_x/devide,
                      "loss_D_y_hist" : epoch_loss_D_y/devide,
                      "loss_G_GAN_hist" : epoch_loss_G_GAN/devide,
                      "loss_F_GAN_hist" : epoch_loss_F_GAN/devide,
                      "loss_cycle_hist" : epoch_loss_cycle/devide,
                      "loss_identity_hist" : epoch_loss_identity/devide,
                      "fid" : fid,
                      "epoch" :self.curr_epoch
                      }) 

            if self.image_test and self.curr_epoch%5==0:
                generate_and_save_images(self.G,self.loader,self.F,self.target_loader,
                                         save_path,self.curr_epoch,self.args.save,self.device)
            self.curr_epoch += 1
            if self.curr_epoch%10==0:
                self.save_checkpoint(os.path.join(self.ckpt_save_path, 'checkpoint-epoch-{0}.ckpt'.format(self.curr_epoch)))
            print("Training Phase [{0}/{1}], {2:.4f} seconds".format(self.curr_epoch, num_epochs, time.time() - start))

        self.save_checkpoint(os.path.join(self.ckpt_save_path, 'checkpoint-epoch-{0}.ckpt'.format(num_epochs)))

        return self.loss_D_x_hist, self.loss_D_y_hist, self.loss_G_GAN_hist, self.loss_F_GAN_hist, \
               self.loss_cycle_hist, self.loss_identity_hist


    def train_step(self, img, target_img):
        # photo images are X, animation images are Y

        self.D_x_optimizer.zero_grad()
        self.D_y_optimizer.zero_grad()
        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()

        # Generate images and save them to image buffers
        generated_y = self.G(img) #G(x)
        self.generated_y_images.save(generated_y.detach())

        generated_x = self.F(target_img) # F(y)
        self.generated_x_images.save(generated_x.detach())

        # train D_y with gray_images and generated_y
        target_img.requires_grad = True
        target_output = self.D_y(target_img) 
        target_target = torch.ones_like(target_output)
        loss_animation = self.GAN_criterion(target_output, target_target)
        loss_D_y = loss_animation

        generated_y_sample = self.generated_y_images.sample(self.args.batch_size)
        generated_output = self.D_y(generated_y_sample)
        if self.args.loss == 'cyclegan':
            generated_target = torch.zeros_like(generated_output)
        elif self.args.loss == 'hingeloss':
            generated_target = torch.full_like(generated_output, -1)
            
        loss_generated = self.GAN_criterion(generated_output, generated_target)
        loss_D_y += loss_generated
        loss_D_y /= 2
        
        # R1 regularizatoin term for Discriminator_y
        if self.args.r1:
            r1_reg_y = self.R1_reg(target_output, target_img)
            loss_D_y += r1_reg_y

        loss_D_y.backward()
        self.D_y_optimizer.step()

        # train D_x with img and generated_x
        img.requires_grad = True
        photo_output = self.D_x(img)
        photo_target = torch.ones_like(photo_output)
        loss_photo = self.GAN_criterion(photo_output, photo_target)
        loss_D_x = loss_photo

        generated_x_sample = self.generated_x_images.sample(self.args.batch_size)
        generated_output = self.D_x(generated_x_sample)
        if self.args.loss == 'cyclegan':
            generated_target = torch.zeros_like(generated_output)
        elif self.args.loss == 'hingeloss':
            generated_target = torch.full_like(generated_output, -1)
        loss_generated = self.GAN_criterion(generated_output, generated_target)
        loss_D_x += loss_generated
        loss_D_x /= 2
        
        # R1 regularizatoin term for Discriminator_x
        if self.args.r1:
            r1_reg_x = self.R1_reg(photo_output, img)
            loss_D_x += r1_reg_x

        loss_D_x.backward()
        self.D_x_optimizer.step()

        # time to train G and F
        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()

        # 1. GAN loss
        generated_y_output = self.D_y(generated_y)
        generated_y_target = torch.ones_like(generated_y_output)
        loss_G_GAN = self.GAN_criterion(generated_y_output, generated_y_target)

        generated_x_output = self.D_x(generated_x)
        generated_x_target = torch.ones_like(generated_x_output)
        loss_F_GAN = self.GAN_criterion(generated_x_output, generated_x_target)

        # 2. Cycle-Consistency loss
        cycle_x = self.F(generated_y)  # X -> Y -> X
        cycle_y = self.G(generated_x)  # Y -> X -> Y

        loss_cycle = self.lambda_cycle * self.Cycle_criterion(cycle_x, img)
        loss_cycle += self.lambda_cycle * self.Cycle_criterion(cycle_y, target_img)

        # 3. identity loss
        G_y = self.G(target_img)
        F_x = self.F(img)
        loss_identity = self.lambda_identity * self.Identity_criterion(G_y, target_img)
        loss_identity += self.lambda_identity * self.Identity_criterion(F_x, img)

        generator_losses = loss_G_GAN + loss_F_GAN + loss_cycle + loss_identity
        generator_losses.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()

        return loss_D_x.detach().item(), loss_D_y.detach().item(), loss_G_GAN.detach().item(), loss_F_GAN.detach().item(), \
               loss_cycle.detach().item(), loss_identity.detach().item(),torch.mean(target_output).detach().item(),torch.mean(generated_y_output).detach().item(),torch.mean(photo_output).detach().item(),torch.mean(generated_x_output).detach().item()
                
                
    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                'G_state_dict': self.G.state_dict(),
                'F_state_dict': self.F.state_dict(),
                'D_x_state_dict': self.D_x.state_dict(),
                'D_y_state_dict': self.D_y.state_dict(),
                'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                'F_optimizer_state_dict': self.F_optimizer.state_dict(),
                'D_x_optimizer_state_dict': self.D_x_optimizer.state_dict(),
                'D_y_optimizer_state_dict': self.D_y_optimizer.state_dict(),
                'loss_D_x_hist': self.loss_D_x_hist,
                'loss_D_y_hist': self.loss_D_y_hist,
                'loss_G_GAN_hist': self.loss_G_GAN_hist,
                'loss_F_GAN_hist': self.loss_F_GAN_hist,
                'loss_cycle_hist': self.loss_cycle_hist,
                'loss_identity_hist': self.loss_identity_hist,
                'init_loss_hist': self.init_loss_hist,
                'curr_epoch': self.curr_epoch

            }, checkpoint_path
        )
    
    def initialize_step(self, img, target_img):
        # TODO
        # Train only using cycle-consistency and identity loss
        self.G_optimizer.zero_grad()
        self.F_optimizer.zero_grad()
        generated_y = self.G(img)
        generated_x = self.F(target_img)
        cycle_x = self.F(generated_y)  # X -> Y -> X
        cycle_y = self.G(generated_x)  # Y -> X -> Y
        loss_cycle = self.lambda_cycle * self.Cycle_criterion(cycle_x, img)
        loss_cycle += self.lambda_cycle * self.Cycle_criterion(cycle_y, target_img)

        G_y = self.G(target_img)
        F_x = self.F(img)
        loss_identity = self.lambda_identity * self.Identity_criterion(G_y, target_img)
        loss_identity += self.lambda_identity * self.Identity_criterion(F_x, img)

        initialization_loss = loss_cycle + loss_identity
        initialization_loss.backward()
        self.G_optimizer.step()
        self.F_optimizer.step()

        return initialization_loss.detach().item()

    def load_checkpoint(self, checkpoint_path):
        print("loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.F.load_state_dict(checkpoint['F_state_dict'])
        self.D_x.load_state_dict(checkpoint['D_x_state_dict'])
        self.D_y.load_state_dict(checkpoint['D_y_state_dict'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.F_optimizer.load_state_dict(checkpoint['F_optimizer_state_dict'])
        self.D_x_optimizer.load_state_dict(checkpoint['D_x_optimizer_state_dict'])
        self.D_y_optimizer.load_state_dict(checkpoint['D_y_optimizer_state_dict'])
        self.loss_D_x_hist = checkpoint['loss_D_x_hist']
        self.loss_D_y_hist = checkpoint['loss_D_y_hist']
        self.loss_G_GAN_hist = checkpoint['loss_G_GAN_hist']
        self.loss_F_GAN_hist = checkpoint['loss_F_GAN_hist']
        self.loss_cycle_hist = checkpoint['loss_cycle_hist']
        self.loss_identity_hist = checkpoint['loss_identity_hist']
        self.init_loss_hist = checkpoint['init_loss_hist']
        try:
            self.curr_epoch = checkpoint['curr_epoch']
        except:
            self.curr_epoch = int(checkpoint_path.split('-')[-1].split('.')[0])

def generate_and_save_images(generator_X, test_image_X_loader, generator_Y, test_image_Y_loader,
                             save_path, epoch, save, device):
    # 이미지 test + 생성으로 저장.
    generator_X.eval()
    generator_Y.eval()
    torch_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])
    
    ## Save X to Y image
    image_ix = 0
    for test_images,_ in test_image_X_loader:
        test_images = test_images.to(device)
        generated_images = generator_X(test_images).detach().cpu()
        # Gray scale일 때 아래 실행 코드
        if (test_images.shape)[1]==1:
            generated_images=torch.cat([generated_images,generated_images,generated_images],1)
            test_images=torch.cat([test_images,test_images,test_images],1)

        for i in range(len(generated_images)):
            test_image=test_images[i]
            test_image=torch_to_image(test_image)
            image = generated_images[i]
            image = torch_to_image(image)
            new_img=Image.new('RGB',(2*((test_image.size)[0]),(test_image.size)[1]))
            new_img.paste(test_image,(0,0))
            new_img.paste(image,((test_image.size)[0],0))
            new_img.save(os.path.join(save_path, f'XtoY_{epoch}_{image_ix}.jpg'))
            if save:
                wandb.log({"Generator_X Samples": [wandb.Image(f'./%s/XtoY_{epoch}_{image_ix}.jpg' %
                            (save_path))]}, commit=True)
            image_ix += 1
        break
    
    ## Save Y to X image
    image_ix = 0
    for test_images in test_image_Y_loader:
        test_images = test_images.to(device)
        generated_images = generator_Y(test_images).detach().cpu()
        # Gray scale일 때 아래 실행 코드
        if (test_images.shape)[1]==1:
            generated_images=torch.cat([generated_images,generated_images,generated_images],1)
            test_images=torch.cat([test_images,test_images,test_images],1)

        for i in range(len(generated_images)):
            test_image=test_images[i]
            test_image=torch_to_image(test_image)
            image = generated_images[i]
            image = torch_to_image(image)
            new_img=Image.new('RGB',(2*((test_image.size)[0]),(test_image.size)[1]))
            new_img.paste(test_image,(0,0))
            new_img.paste(image,((test_image.size)[0],0))
            new_img.save(os.path.join(save_path, f'YtoX_{epoch}_{image_ix}.jpg'))
            if save:
                wandb.log({"Generator_Y Samples": [wandb.Image(f'./%s/YtoX_{epoch}_{image_ix}.jpg' %
                            (save_path))]}, commit=True)
            image_ix += 1
        break


                

class ImagePool:
    def __init__(self,args, maxlen=50):
        self.buffer = []
        self.maxlen = maxlen
        self.args=args

    def save(self, images):
        for image in images:
            if len(self.buffer) >= self.maxlen:
                idx = np.random.randint(0, self.maxlen)
                self.buffer[idx] = image
            else:
                self.buffer.append(image)

    def sample(self, sample_size=8):
        idxs = np.random.choice(len(self.buffer), sample_size)
        return torch.stack([self.buffer[idx] for idx in idxs])